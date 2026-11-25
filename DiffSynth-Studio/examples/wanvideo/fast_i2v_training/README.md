# Wan-2.1-Fast-14B 训练与评测

在 `DiffSynth-Studio/` 下运行以下命令。请先按说明下载模型。[模型下载与目录结构](../../../README.md#21-download-models)

## 训练

```sh
python downloads/download_training_data.py fast-14b
bash train_fast_14B.sh
```

训练使用真实视频的首帧和 caption，分辨率为 832×480，共 81 帧。默认八张卡，每张卡每次读一条样本，累积八次更新梯度，有效 batch 为 64。第一阶段每处理 512 个正例保存一次 checkpoint。进入无评分 PhyGDPO 训练后，每处理 256 对正负例保存一次。各阶段也会保存最后一次更新。

两个阶段的 checkpoint 分别保存在 `models/train/Wan2.1-Fast-TI2V-14B_PhyGDPO/` 下的 `sft/checkpoints/` 和 `dpo/checkpoints/`。重复运行同一命令会从该目录续训。要继续已有的旧实验，可将 `PHYGDPO_TRAIN_DIR` 指向原来包含 `sft/` 和 `dpo/` 的目录。分开保存的两个旧实验可以分别链接为该目录下的 `sft/` 和 `dpo/`。`bash train_fast_14B.sh --plan` 可查看保存时间表。原实验的 checkpoint 可以使用下载命令生成的对应清单续训，初始化 LoRA 按文件 SHA-256 核对。可以通过 `PHYGDPO_CACHE_DIR` 指定预处理特征的位置。

若要直接从发布的 Wan-2.1-Fast-14B 权重开始无评分 PhyGDPO 训练，可单独运行

```sh
PHYGDPO_INIT_ADAPTER=models/pretrained/Wan2.1-Fast-TI2V-14B_PhyGDPO/epoch-0.safetensors \
  python examples/wanvideo/fast_i2v_training/run.py dpo --skip-eval
```

## 生成负例

原实验为每个正视频配一个已发布的 Wan2.1-T2V-14B 负例。也可以用 Wan-2.1-Fast-14B 从正视频首帧和 caption 生成新的负例。

```sh
python downloads/download_training_data.py fast-14b --positives-only
bash test_dpo_losing_case_fast_14B.sh
```

每个负视频按训练帧率 15 fps 保存。默认训练复现原实验，使用 Wan2.1-T2V-14B 的发布负例。要使用自生成负例，运行

```sh
bash train_fast_14B.sh --negative-source generated
```

显式指定的负例来源优先。生成了多个 seed 时，可用 `PHYGDPO_PAIRS_CSV` 指定其中一份自生成清单。自生成数据的 checkpoint 和特征按 seed 分目录保存。

## 无评分 PhyGDPO 的目标函数

正负视频使用同一段 caption、正视频首帧和时间步。策略模型和冻结的参考模型对同一分支使用相同的噪声。记 $d_w=L_\theta^w-L_{\mathrm{ref}}^w$、$d_l=L_\theta^l-L_{\mathrm{ref}}^l$，其中 $L$ 为去噪 MSE。取 $\alpha=\gamma=1$ 时，训练目标为

$$
\mathcal L_{\mathrm{DPO}}=\operatorname{softplus}\!\left(1000(d_w-d_l)\right).
$$

策略和参考从相同权重初始化，初始偏好损失为 $\log 2$。

## 手动评测

同一运动的四个场景使用不同的源视频、首帧和 prompt。每个场景跨 checkpoint 固定首帧、prompt 和 seed。默认设置为 832×480、81 帧、4 步、CFG 1 和 24 fps，VAE 以 FP32 解码。以下命令单独评测发布权重的 Gymnastics-1，视频和同名 JSON 保存在 `runs/wan-fast/evaluation/phygdpo/`。

```sh
python examples/wanvideo/fast_i2v_training/evaluate.py check --case gymnastics-p1
torchrun --standalone --nproc_per_node=8 examples/wanvideo/fast_i2v_training/evaluate.py features --case gymnastics-p1
torchrun --standalone --nproc_per_node=8 examples/wanvideo/fast_i2v_training/evaluate.py generate --case gymnastics-p1 \
  --label phygdpo --adapter models/pretrained/Wan2.1-Fast-TI2V-14B_PhyGDPO/epoch-0.safetensors
```

## 基座权重

原始 CausVid rank-32 LoRA 以 0.95 合并到 Wan I2V 基座，来源版本记录在 `fusion.json`。重建基座可用 `fuse_model.py` 的 `--base`、`--causvid` 和 `--out` 指定输入权重和输出目录。可通过 `PHYGDPO_MODEL_DIR` 选择本地融合模型。[原 Space](https://huggingface.co/spaces/ysharma/wan2-1-fast/blob/18146426e964a4cd009f596ec17da34793df48c3/app.py) · [发布权重](https://huggingface.co/CaiYuanhao/PhyGDPO/tree/main/Wan2.1-Fast-TI2V-14B_PhyGDPO)
