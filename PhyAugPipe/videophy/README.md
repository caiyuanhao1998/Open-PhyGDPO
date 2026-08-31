&nbsp;

<div align="center">

<p align="center"> <img src="../img/logo.png" width="250px"> </p>

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2512.24551)
[![project](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/PhyGDPO/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/CaiYuanhao/PhyGDPO)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/CaiYuanhao/PhyGDPO)
<h3>PhyGDPO: Physics-Aware Groupwise Direct Preference <br> Optimization for Physically Consistent Text-to-Video Generation</h3> 

</div>

&nbsp;

### Introduction
This part of code develops the [videophy2](https://videophy2.github.io/) codebase to score the videos for physics rewarding of our PhyGDPO and quantitatic evaluation. We write the data parallel version to speed up the scoring process. Please refer to [the original github repo of videophy](https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2) for detailed instruction.


### 1. Installation

#### 1.1 Creating conda environment
```python
conda create -n videophy python=3.10
conda activate videophy
```

#### 1.2 Install dependencies
```python
python -m pip install -r requirements.txt
```

### 2. Score the generated videos as the losing cases for DPO training
```sh
cd VIDEOPHY2
. eval_all_train_part_0_14B.sh
```

`Note:` This environment pins `peft==0.4.0`, which is compatible with its
`transformers==4.28.1`. Newer PEFT releases fail to import with that Transformers
version. Losing-case scores are written under `evaluated_score`.

### Citation

If this code is useful for your research, please cite our ECCV 2026 paper:

```bibtex
@inproceedings{phygdpo,
  title={PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation},
  author={Cai, Yuanhao and Li, Kunpeng and Jia, Menglin and Wang, Jialiang and Sun, Junzhe and Liang, Feng and Chen, Weifeng and Juefei-Xu, Felix and Wang, Chu and Thabet, Ali and Dai, Xiaoliang and Ju, Xuan and Yuille, Alan and Hou, Ji},
  booktitle={ECCV},
  year={2026}
}
```
