import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset_Physics_DPO_Reward_Group, ModelLogger_Steps_Resume, launch_training_task_steps_resume, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pdb import set_trace as stx
import copy
from contextlib import contextmanager
from PIL import Image
from torch.nn import functional as F

ADAPTER_NAME = "dpo"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        skip_download=True,
        dpo_beta=0.1,
        alpha_min=0.5,
        k_alpha=5.0,
        b_alpha=0.5,
        k_gamma=3.0,
        b_gamma=0.5,
        lambda_gamma=0.5,
    ):
        super().__init__()
        # Load models
        model_configs = []
        self.lora_modules = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, skip_download=skip_download)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)


        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model_dpo(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                adapter_name=ADAPTER_NAME
            )
            setattr(self.pipe, lora_base_model, model)
            # 保存引用，后续开/关 LoRA 用
            self.lora_modules.append(getattr(self.pipe, lora_base_model))
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.dpo_beta = dpo_beta

        # lora on/off self-check
        try:
            with self.without_lora():
                pass
            print("[LoRA toggle self-check] OK", flush=True)
        except Exception as e:
            print("[LoRA toggle self-check] FAILED:", repr(e), flush=True)

    @contextmanager
    def without_lora(self):
        # 关闭：把指定 adapter 的 scale 置 0
        for m in self.lora_modules:
            for sub in m.modules():
                if hasattr(sub, "set_scale"):
                    sub.set_scale(ADAPTER_NAME, 0.0)
        try:
            yield
        finally:
            # 恢复：scale 置 1
            for m in self.lora_modules:
                for sub in m.modules():
                    if hasattr(sub, "set_scale"):
                        sub.set_scale(ADAPTER_NAME, 1.0)


    def dpo_loss(self, l_model_w, l_model_l, l_ref_w, l_ref_l):
        """
        Direct Preference Optimization (JMLR 2024) 的标量损失：
            L = -log σ[ β * ((l_model_w - l_ref_w) - (l_model_l - l_ref_l)) ]
        这里 l_* 可以是 “越小越好” 的 loss，也可以是 -log prob。
        """
        advantage = (l_model_w - l_ref_w) - (l_model_l - l_ref_l)
        return -torch.log(torch.sigmoid(-self.dpo_beta * advantage)).mean()


    def dpo_loss_reward(self,
                    l_model_w, l_model_l, l_ref_w, l_ref_l,
                    sa_score, pc_score):
        """
        Physics-aware DPO (logsigmoid sign consistent with your code).
        Uses alpha(v) in (- inside logsigmoid) and gamma(v) as multiplicative weight.
        """

        device = getattr(self.pipe, "device", l_model_w.device)

        # ---------- advantage ----------
        advantage = (l_model_w - l_ref_w) - (l_model_l - l_ref_l)   # [B], same as yours

        # ---------- physics-aware α, γ from VLM (no advantage used here) ----------
        # violation v ∈ [0,1]: larger = worse (lower sa/pc)
        v = 1.0 - (sa_score / 5.0 + pc_score / 5.0) / 2.0
        v = torch.as_tensor(v, dtype=torch.float32, device=device)

        # hyperparams (safe defaults; all overridable on self)
        # alpha_min=0.5,
        # k_alpha=5.0,
        # b_alpha=0.5,
        # k_gamma=3.0,
        # b_gamma=0.5,
        # lambda_gamma=0.5,
        alpha_min     = getattr(self, "alpha_min", 0.5)     # (0,1]
        k_alpha       = getattr(self, "k_alpha", 5.0)       # slope for α's sigmoid
        b_alpha       = getattr(self, "b_alpha", 0.5)       # threshold for α in [0,1]
        k_gamma       = getattr(self, "k_gamma", 3.0)       # slope for γ's sigmoid
        b_gamma       = getattr(self, "b_gamma", 0.5)       # threshold for γ
        lambda_gamma  = getattr(self, "lambda_gamma", 0.5)  # ≥ 0

        # α(v) ∈ (alpha_min, 1]
        alpha = alpha_min + (1.0 - alpha_min) * torch.sigmoid(k_alpha * (v - b_alpha))
        alpha = torch.clamp(alpha, min=1e-4, max=1.0)

        # γ(v) ≥ 1/α  (guarantees the LSE→logsigmoid upper bound condition)
        gamma = (1.0 / alpha) * (1.0 + lambda_gamma * torch.sigmoid(k_gamma * (v - b_gamma)))

        # ---------- DPO 基础项（与您的号位一致：-logsigmoid(- ... )） ----------
        beta = getattr(self, "dpo_beta", 1.0)
        logits = beta * alpha * advantage
        base_loss = -F.logsigmoid(-logits)    # 保持与你原代码相同的负号与位置

        # ---------- 最终损失：gamma 做样本权重 ----------
        loss = (gamma * base_loss).mean()
        return loss

        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi_win = {"prompt": data["prompt"]}
        inputs_nega_win = {}
        inputs_posi_lose = {"prompt": data["prompt"]}
        inputs_nega_lose = {}
        # RuntimeError: manual_seed expected a long, but got Tensor
        seed_win = torch.randint(0, 1000000, (1,)).item()
        seed_lose = torch.randint(0, 1000000, (1,)).item()
        
        # CFG-unsensitive parameters
        inputs_shared_win = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video_positive"],
            "height": data["video_positive"][0].size[1],
            "width": data["video_positive"][0].size[0],
            "num_frames": len(data["video_positive"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "seed": seed_win,
        }

        inputs_shared_lose = {
            "input_video": data["video_negative"],
            "height": data["video_negative"][0].size[1],
            "width": data["video_negative"][0].size[0],
            "num_frames": len(data["video_negative"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "seed": seed_lose,
        }

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared_win["input_image"] = data["video_positive"][0]
                inputs_shared_lose["input_image"] = data["video_negative"][0]
            elif extra_input == "end_image":
                inputs_shared_win["end_image"] = data["video_positive"][-1]
                inputs_shared_lose["end_image"] = data["video_negative"][-1]
            else:
                inputs_shared_win[extra_input] = data[extra_input]
                inputs_shared_lose[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        # self.pipe 是 WanVideoPipeline 类，self.pipe.units 是 WanVideoPipeline 类中若干数据预处理的管线
        for unit in self.pipe.units:
            inputs_shared_win, inputs_posi_win, inputs_nega_win = self.pipe.unit_runner(unit, self.pipe, inputs_shared_win, inputs_posi_win, inputs_nega_win)
            inputs_shared_lose, inputs_posi_lose, inputs_nega_lose = self.pipe.unit_runner(unit, self.pipe, inputs_shared_lose, inputs_posi_lose, inputs_nega_lose)
        return {**inputs_shared_win, **inputs_posi_win}, {**inputs_shared_lose, **inputs_posi_lose}

    
    def forward(self, data):
        # type(data) = dict_keys(['video_positive', 'video_negative', 'prompt'])
        # data.keys() = dict_keys(['video_positive', 'video_negative', 'sa_score', 'pc_score', 'joint_score', 'prompt'])
        # stx()
        inputs_win, inputs_lose = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        # stx()
        loss_win = self.pipe.training_loss_timestep(timestep_id, **models, **inputs_win)            # 正常的 eps loss
        # stx()
        loss_lose = self.pipe.training_loss_timestep(timestep_id, **models, **inputs_lose)          # 正常的 eps loss

        # reference model 就是关闭 lora 的 pipe
        with torch.no_grad(), self.without_lora():
            # stx()
            loss_ref_win = self.pipe.training_loss_timestep(timestep_id, **models, **inputs_win)
            # stx()
            loss_ref_lose = self.pipe.training_loss_timestep(timestep_id, **models, **inputs_lose)
            # stx()
            loss_ref_win = loss_ref_win.detach()
            loss_ref_lose = loss_ref_lose.detach()

        loss = self.dpo_loss_reward(loss_win, loss_lose, loss_ref_win, loss_ref_lose, data["sa_score"], data["pc_score"])
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # 1. 加载数据集
    # dict_keys(['video', 'prompt'])  -->  dict_keys(['video_positive', 'video_negative', 'prompt'])
    dataset = VideoDataset_Physics_DPO_Reward_Group(args=args)

    # stx()

    # 需要给 dataset 增加几个 VLM 的字段


    # stx()
    '''
        for physics dataset training:
        len(dataset) = 16840
        dataset[0] = dict_keys(['video', 'prompt']) - video 长度为 81 - list of PIL.Image.Image
        for extended: prompt - The person, wearing a glove, pours green liquid from 
        a pot into a jar. Gravity pulls the liquid downward as it flows. Friction from 
        the glove helps grip the containers, while the liquid's viscosity controls its flow. 
        The normal force from the floor keeps everything stable, resulting in the liquid 
        transferring into the jar and some splashing onto the floor.

        for not extended: prompt - In the video, a person is seen wearing a glove and holding a jar.
        The jar is being filled with a green liquid that is being poured from a pot into the jar. 
        The liquid appears to be some kind of food or beverage. The person is standing in a kitchen, 
        and there are various kitchen utensils and appliances visible in the background. 
        The lighting in the room is bright, and the walls are painted white. 
        The person is focused on pouring the liquid into the jar, and there is no one else visible in the video.
    '''

    # 2. 加载模型
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        skip_download=args.skip_download,
        dpo_beta=args.dpo_beta,
        alpha_min=args.alpha_min,
        k_alpha=args.k_alpha,
        b_alpha=args.b_alpha,
        k_gamma=args.k_gamma,
        b_gamma=args.b_gamma,
        lambda_gamma=args.lambda_gamma,
    )

    # 3. 加载模型日志
    model_logger = ModelLogger_Steps_Resume(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps
    )

    # 4. 加载优化器
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)

    # 5. 加载学习率调度器
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # 6. 启动训练任务
    launch_training_task_steps_resume(
        dataset, model, model_logger, optimizer, scheduler,
        total_training_steps=args.total_training_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )