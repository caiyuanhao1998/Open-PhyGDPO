import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.distributed.tensor import distribute_tensor
from safetensors import safe_open

GROUP = 5


def checkpoint_attention(forward):
    def apply(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(
                forward, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return forward(*args, **kwargs)

    return apply


def tokenwise_mlp(forward):
    def apply(x):
        # Wan's feed-forward network has no attention, token mixing, or dropout.
        outputs = []
        for tokens in x.split(4096, dim=1):
            if torch.is_grad_enabled():
                result = checkpoint(
                    forward, tokens, use_reentrant=False, preserve_rng_state=False
                )
            else:
                result = forward(tokens)
            outputs.append(result)
        return torch.cat(outputs, dim=1)

    return apply


def reduce_block_memory(block):
    block.attn1.forward = checkpoint_attention(block.attn1.forward)
    block.attn2.forward = checkpoint_attention(block.attn2.forward)
    block.ffn.forward = tokenwise_mlp(block.ffn.forward)


def preference_coefficient(
    winner, rejected, reference_winner, reference_rejected, beta
):
    with torch.no_grad():
        logit = beta * ((winner - reference_winner) - (rejected - reference_rejected))
        loss = torch.nn.functional.softplus(logit)
        coefficient = beta * torch.sigmoid(logit)
    return loss, coefficient


class BlockGroup(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, context, timestep, rotary):
        for block in self.layers:
            if torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    context,
                    timestep,
                    rotary,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                x = block(x, context, timestep, rotary)
        return x


def group_checkpoint(group, x, context, timestep, rotary):
    boundary_shape = x.shape

    def pack(t):
        if t.is_cuda and t.ndim == 3 and t.shape == boundary_shape:
            return (t.device, t.detach().cpu())
        return t

    def unpack(item):
        return item[1].to(item[0]) if isinstance(item, tuple) else item

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        return checkpoint(
            group,
            x,
            context,
            timestep,
            rotary,
            use_reentrant=False,
            preserve_rng_state=False,
        )


@torch.no_grad()
def install_adapter(path, parameters, mesh):
    with safe_open(str(path), framework="pt") as src:
        if set(src.keys()) != set(parameters):
            raise ValueError("Checkpoint tensors do not match the evaluation model")
        for key, p in parameters.items():
            full = src.get_tensor(key).to(dtype=p.dtype, device=p.device)
            if full.shape != p.shape:
                raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
            shard = distribute_tensor(
                full, device_mesh=mesh, placements=p.placements, src_data_rank=None
            )
            p.to_local().copy_(shard.to_local())
