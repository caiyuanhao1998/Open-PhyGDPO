# 按训练清单下载数据

在 `DiffSynth-Studio/` 下运行以下命令。下载工具使用 Python 3.10+ 和 `requests`，按所选模型的清单从压缩包中提取训练视频。三个模型默认共用 `data/`，校验通过的正视频会复用。使用其他目录时，下载、生成负例和训练均可通过 `PHYGDPO_DATA_DIR` 指定数据根目录。

```sh
python downloads/download_training_data.py fast-14b --plan
python downloads/download_training_data.py fast-14b
python downloads/download_training_data.py fast-14b --verify-only
```

`--plan` 可离线预览。下载器默认使用四个线程，接受服务端的 HTTP 206 字节范围响应，并校验范围、文件长度和 CRC32。通过校验的文件会从临时路径移到目标位置。可用 `--workers` 调整并发数。

## 一键下载训练正例

在 `DiffSynth-Studio/` 下运行一次，即可准备三个模型需要的真实正例和 caption。共享文件按视频 ID 去重，两个 T2V 模型各自生成 `prompts.json`，Wan-2.1-Fast-14B 生成带 caption 的 `positives.csv`。

```sh
python downloads/download_training_data.py all-positives
```

准备好正例后，可自行生成负例，也可为所选模型运行带 `--negatives-only` 的下载命令，补齐已发布的负例和训练清单。[负例生成与下载命令](../README.md#31-prepare-data)

只需单个模型的正例时，任选下列一条命令。

```sh
# Wan2.1-T2V-1.3B
python downloads/download_training_data.py original-1.3b --positives-only

# Wan-2.1-Fast-14B
python downloads/download_training_data.py fast-14b --positives-only

# Wan2.1-T2V-14B
python downloads/download_training_data.py original-14b --positives-only
```

正视频写入 `data/T2V_data/VIDGEN-1M_unzip/`。例如，原版 14B 的清单保存在 `data/manifests/original-14b-positives-only/positives.csv`，其中的媒体路径相对于数据根目录。为原版 T2V 生成负例时，在正例命令末尾加上 `--unscored` 即可同时下载 caption 并生成 `prompts.json`。[生成负例](../README.md#31-prepare-data)

## 清单与抽样

- 运动视频清单保留实验顺序、prompt、运动类别、原始宽高和固定负例 seed。以正视频为准筛选真人实拍、无水印、无台标、无字幕的素材，像素数至少为 832×480。每个正视频固定配一个 Wan2.1-T2V-14B 负例。
- `manifests/members.csv.gz` 记录所需 ZIP 成员的分片名、偏移、压缩方式、大小和 CRC32。VidGen 固定在 `9158f21`，PhyGDPO 负例固定在 `f68effe`。[VidGen-1M](https://huggingface.co/datasets/Fudan-FUXI/VIDGEN-1M/tree/9158f218b6b81694a7809b6ee74e30003d58a70e) · [PhyGDPO 负例](https://huggingface.co/datasets/CaiYuanhao/PhyGDPO/tree/f68effe3613d3a6cb2e5a69db53fa7f187adc4a3)
- `--ids` 接受每行一个视频 ID 的文本文件。`--limit` 取各模型清单开头的样本，在两个下载步骤中设置相同的上限即可配对。子集清单写入单独目录，下载器会打印实际路径。对子集训练 Wan-2.1-Fast-14B 时，将 `PHYGDPO_POSITIVE_CSV` 和 `PHYGDPO_PAIRS_CSV` 分别设为该目录中的正例和配对清单。

## 原版有评分和无评分训练

发布的原版负例使用 seed 0、1、3。下载器优先使用本项目 `PhyAugPipe/videophy/evaluated_score` 下的 CSV，也支持通过 `--scores` 指定目录。缺少本地评分文件时，会从固定的 GitHub 版本取回，并按 seed 写出与所选视频 ID 对齐的 `final.csv`。以下是原版 14B 的有评分训练命令。

```sh
python downloads/download_training_data.py original-14b
bash train_14B.sh
```

无评分模式按 ID 下载原始 caption，同一个视频 ID 只取一份。下载完整无评分配对后，启动脚本会自动找到清单。

```sh
python downloads/download_training_data.py original-14b --unscored
PHYGDPO_SCORE_CSV="" bash train_14B.sh
```

有评分子集用 `PHYGDPO_SCORE_CSV` 指定该子集 seed 0 的评分文件。无评分子集用 `PHYGDPO_PAIRS_CSV` 指定该子集的 `pairs.csv` 并设置 `PHYGDPO_SCORE_CSV=""`。旧数据目录也可按同名 MP4 和非空 TXT 组对。原版 T2V 已整理在 `PhyAugPipe/` 的数据也可通过 `PHYGDPO_DATA_DIR=../PhyAugPipe` 使用。

完整配对下载会生成 `prompts.json`。若需 Gemini 扩写的 prompt，可从固定版本的 `json_file.zip` 取出 `json_file/gemini/` 中的原始分片。[JSON 存档](https://huggingface.co/datasets/CaiYuanhao/PhyGDPO/blob/f68effe3613d3a6cb2e5a69db53fa7f187adc4a3/json_file.zip)
