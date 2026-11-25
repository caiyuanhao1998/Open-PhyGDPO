import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from accelerate.checkpointing import load_accelerator_state, save_accelerator_state
from accelerate.utils import broadcast_object_list, gather_object, load_fsdp_optimizer, save_fsdp_optimizer
from filelock import FileLock
from safetensors import safe_open
from torch.distributed.tensor import DTensor, distribute_tensor


LORA_CHECKPOINT_FORMAT = "phygdpo-lora-v1"


@contextmanager
def checkpoint_read_lock(accelerator, directories):
    with ExitStack() as locks:
        if accelerator.is_main_process:
            for directory in sorted(set(directories)):
                lock = directory / ".checkpoint.lock"
                if lock.is_file() and os.access(lock, os.W_OK):
                    locks.enter_context(FileLock(lock))
        yield


@contextmanager
def staged_checkpoint(accelerator, adapter, state_dir):
    """Publish a matching adapter and state only after all ranks finish writing."""
    with ExitStack() as stack:
        shared_path = [None]
        if accelerator.is_main_process:
            staging = stack.enter_context(TemporaryDirectory(prefix=f".{adapter.stem}-", dir=adapter.parent))
            (Path(staging) / "state").mkdir()
            shared_path[0] = staging
        stage = Path(broadcast_object_list(shared_path)[0])
        yield stage / "adapter.safetensors", stage / "state"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Serialize publications. An interrupted retry can leave an older
            # state directory, including frozen weights from the legacy format.
            with FileLock(adapter.parent / ".checkpoint.lock"):
                previous_state = stage / "previous-state"
                previous_adapter = stage / "previous-adapter"
                if state_dir.exists() or state_dir.is_symlink():
                    state_dir.replace(previous_state)
                try:
                    if adapter.exists() or adapter.is_symlink():
                        adapter.replace(previous_adapter)
                    (stage / "adapter.safetensors").replace(adapter)
                    (stage / "state").replace(state_dir)
                except OSError:
                    if previous_adapter.exists() or previous_adapter.is_symlink():
                        previous_adapter.replace(adapter)
                    if previous_state.exists() or previous_state.is_symlink():
                        previous_state.replace(state_dir)
                    raise
        accelerator.wait_for_everyone()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persistent_buffers(model):
    return {
        ".".join(filter(None, (prefix, name))): value
        for prefix, module in model.named_modules()
        for name, value in module.named_buffers(recurse=False)
        if name not in module._non_persistent_buffers_set
    }


def _restore_tensors(path, targets):
    with safe_open(path, framework="pt", device="cpu") as weights:
        if set(targets) != set(weights.keys()):
            raise ValueError(f"Model tensor names differ from {path}. Check the LoRA configuration.")
        for name, target in targets.items():
            value = weights.get_tensor(name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Shape or training dtype differs for {name} in {path}.")
            if isinstance(target, DTensor):
                # Slice the shared CPU file for this rank and preserve the optimizer's Parameter.
                value = distribute_tensor(value, target.device_mesh, target.placements, src_data_rank=None)
                target.to_local().copy_(value.to_local())
            else:
                target.copy_(value)


class LoRATrainingState:
    """Resume from the inference adapter and the matching frozen base."""

    def __init__(self, accelerator, model, optimizer, scheduler, cursor, base_sources, prefix):
        self.accelerator = accelerator
        self.model = model
        self.module = accelerator.unwrap_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cursor = cursor
        self.base_sources = base_sources
        self.prefix = prefix or ""
        self.backend = "fsdp2" if accelerator.is_fsdp2 else "data_parallel"
        self.base_sha256 = None

    def base_digest(self):
        if self.base_sha256 is None:
            # Hash the loaded weights once, including each FSDP shard. Moving the
            # same base to another directory does not invalidate a checkpoint.
            local = []
            if self.accelerator.is_fsdp2 or self.accelerator.is_main_process:
                digest = hashlib.sha256()
                for name, value in self.module.named_parameters():
                    if value.requires_grad:
                        continue
                    tensor = value.to_local() if isinstance(value, DTensor) else value
                    digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
                    flat = tensor.detach().reshape(-1)
                    for part in flat.split(max(1, 16 * 1024**2 // flat.element_size())):
                        digest.update(part.cpu().contiguous().view(torch.uint8).numpy())
                local.append(digest.hexdigest())
            self.base_sha256 = hashlib.sha256("".join(gather_object(local)).encode()).hexdigest()
        return self.base_sha256

    def save(self, directory):
        acc = self.accelerator
        buffers = {}
        for name, buffer in _persistent_buffers(self.module).items():
            value = buffer.full_tensor() if isinstance(buffer, DTensor) else buffer
            if acc.is_main_process:
                buffers[name] = value.detach().to(device="cpu", copy=True).contiguous()
        if buffers:
            acc.save(buffers, str(directory / "buffers.safetensors"), safe_serialization=True)
        optimizers = [self.optimizer]
        if acc.is_fsdp2:
            save_fsdp_optimizer(acc.state.fsdp_plugin, acc, self.optimizer, self.model, str(directory))
            optimizers = []
        save_accelerator_state(directory, [], optimizers, [self.scheduler], [], acc.process_index, acc.step, acc.scaler)
        if acc.is_main_process:
            self.cursor.save([], [], directory)
        return dict(format=LORA_CHECKPOINT_FORMAT, backend=self.backend, world_size=acc.num_processes,
                    base_sources=self.base_sources, base_sha256=self.base_digest(), prefix=self.prefix)

    @torch.no_grad()
    def restore(self, directory, adapter, record):
        acc = self.accelerator
        if record["backend"] != self.backend or record["world_size"] != acc.num_processes:
            raise ValueError(f"Resume {directory} with its original backend and GPU count.")
        expected_checksum = record["adapter_sha256"]
        valid = [False]
        if acc.is_main_process:
            try:
                valid[0] = file_sha256(adapter) == expected_checksum
            except OSError:
                pass
        if not broadcast_object_list(valid)[0]:
            raise OSError(f"Adapter checksum mismatch: {adapter}")
        if self.base_digest() != record["base_sha256"]:
            raise ValueError(f"Frozen base weights differ from those used for {adapter}. "
                             f"Load the original base: {record['base_sources']}")

        parameters = {name.removeprefix(record["prefix"]): p for name, p in self.module.named_parameters()
                      if p.requires_grad}
        _restore_tensors(adapter, parameters)
        buffers = _persistent_buffers(self.module)
        buffer_path = directory / "buffers.safetensors"
        if buffers or buffer_path.exists():
            _restore_tensors(buffer_path, buffers)

        optimizers = [self.optimizer]
        if acc.is_fsdp2:
            shards = directory / "optimizer_0"
            load_fsdp_optimizer(acc.state.fsdp_plugin, acc, self.optimizer.optimizer, self.model,
                                str(shards if shards.is_dir() else directory))
            optimizers = []
        state = load_accelerator_state(directory, [], optimizers, [self.scheduler], [], acc.process_index,
                                       acc.scaler, map_location="cpu")
        acc.step = state["step"]
