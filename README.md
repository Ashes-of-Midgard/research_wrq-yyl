# Research_WRQ-YYL

## 1. INSTALLATION

### Requirements

- Python 3.7
- PyTorch 1.12.1
- MMCV 1.7.0
- MMDetection 2.25.2
- MMSegmentation 0.29.1

注：原文档标注需要MMCV 1.7.0，但是经过实际安装测试，MMCV 1.7.0和MMSegmentation 0.29.1不兼容，安装MMSegmentation 0.29.1时会自动安装MMCV 1.6.2

### A from-scratch setup script

```shell
conda create -n deepir python=3.7 -y
conda activate deepir

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install openmim
mim install mmcv-full
mim install mmdet==2.25.2
mim install mmsegmentation==0.29.1

pip install -r requirements.txt

git clone https://github.com/Ashes-of-Midgard/research_wrq-yyl.git
cd research_wrq-yyl
python setup.py develop
```

## 2. Getting Started
### Data
The dataset can be downloaded [here](https://github.com/YimianDai/open-sirst-v2).
```shell
cd research_wrq-yyl
mkdir data
cd data
git clone https://github.com/YimianDai/open-sirst-v2.git
```


### Train
AAL+FGSM
```shell
python tools/train_det.py configs/aal/ssd512_r34/ssd512_r34_sirst_aal.py --gpu-id 0 --work-dir work_dirs/ssd512_r34_sirst_aal
```

AAL
```shell
python tools/train_det.py configs/aal/ssd512_r34/ssd512_r34_sirst_sa.py --gpu-id 0 --work-dir work_dirs/ssd512_r34_sirst_sa
```

Clean
```shell
python tools/train_det.py configs/aal/ssd512_r34/ssd512_r34_sirst_clean.py --gpu-id 0 --work-dir work_dirs/ssd512_r34_sirst_clean
```

FGSM
```shell
python tools/train_det.py configs/aal/ssd512_r34/ssd512_r34_sirst_fgsm.py --gpu-id 0 --work-dir work_dirs/ssd512_r34_sirst_fgsm
```

### Result

| Model | mNoCoAP |
|---|---|
| Clean | 58.86 |
| FGSM | 61.30 |
| AAL+FGSM | 58.33 |
| AAL | 62.48 |
