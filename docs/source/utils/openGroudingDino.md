## OpenGroudingDino微调

简介：
1. 微调地址： https://github.com/longzw1997/Open-GroundingDino
2. 论文：https://arxiv.org/pdf/2303.05499
3. 项目：https://github.com/IDEA-Research/GroundingDINO


### 1、微调准备

1. 下载预训练模型：https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-1.8m-odvg.pth
2. 配置模型超参数：config/cfg_odvg.yaml(可选)
3. BERT预训练模型：git clone https://hf-mirror.com/google-bert/bert-base-uncased

> huggingFace配置: https://hf-mirror.com/

> 1. 安装要注意transformer库版本太高会使用torch2的接口，可能会报错
> 2. 显卡的指定要注意py文件和终端不要冲突

---

