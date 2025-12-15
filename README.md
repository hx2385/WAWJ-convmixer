# Patches Are All You Need? 
This repository contains a reproduction of ConvMixer for the ICLR 2022 submission ["Patches Are All You Need?"](https://openreview.net/forum?id=TVHS5Y4dNvM) by Haipei Xu and Weixi Huang from Columbia University

## Data
In this project, we used two Dataset, ImageNet-1K and CIFAR-10》 The former can be downloaded via Huggingface through this [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), while the latter will be automatically downloaded by train.py in the CIFAR folder. Due to the computation limitation, we only use the first 100 classes (10%) of the original ImageNet-1K dataset in our experiments.

## Code overview
The most important code is in `convmixer.py`, which is implemented by original authors. The `/CIFAR-10` directory contains files used for training and plotting results on the CIFAR-10 dataset. The `/ImageNet` directory includes scripts for training, figure generation, as well as training arguments and evaluation results on ImageNet-1K.
The `/pytorch-image-models` directory contains the source code implemented by the original author. 

For illustration purposes, we include a `demo.ipynb` notebook demonstrating how to train a ConvMixer model on the CIFAR-10 dataset.



## Training
Since we only have one card (T4), so we trained ImageNet-1K as following (The file train.py is located at /CIFAR-10/train.py):

```
python train.py [/path/to/ImageNet1k] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model convmixer_768_32 
    -b 32 
    -j 4 
    --opt adamw 
    --epochs 150 
    --sched onecycle 
    --amp 
    --input-size 3 224 224
    --lr 0.01 
    --aa rand-m9-mstd0.5-inc1 
    --cutmix 0.5 
    --mixup 0.5 
    --reprob 0.25 
    --remode pixel 
    --num-classes 100 
    --warmup-epochs 0 
    --opt-eps=1e-3 
    --clip-grad 1.0
```

__**Note:**__ While training on `ConvMixer-1536/20`, in order to not causing `OOM`, we used batch size (-b) of 16 instead.

## Organization of this directory

```   
WAWJ-convmixer
├── convmixer.py
├── Demo.ipynb
├── E4040.2025Fall.WAWJ.report.hx2385.wh2610.pdf
├── LICENSE
├── README.md
├── CIFAR-10
│   ├── Hyperparameter tuning.ipynb
│   ├── README.md
│   ├── run_CIFAR.ipynb
│   ├── train.py
│   └── figures
│       ├── convmixer_ks_ps_comparison.png
│       ├── test_acc_compare.png
│       ├── time_per_epoch_compare.png
│       └── train_acc_compare.png
├── Figures
│   ├── WAWJ_gcp_work_example_screenshot_1.png
│   ├── WAWJ_gcp_work_example_screenshot_2.png
│   └── WAWJ_gcp_work_example_screenshot_3.png
├── ImageNet
│   ├── Convmix_plot_ImageNet.ipynb
│   ├── Convmix_plot_ImageNet_768_32.ipynb
│   ├── get-training.ipynb
│   ├── ImageNet_val_first100.ipynb
│   ├── README.md
│   ├── Requirements.txt
│   ├── .ipynb_checkpoints
│   │   └── Validation_ImageNet-checkpoint.ipynb
│   ├── figures
│   │   ├── convmixer_1536_20.png
│   │   └── convmixer_768_32.png
│   └── Models
│       ├── ConvMixer-1536-20
│       │   ├── args.yaml
│       │   └── summary.csv
│       └── ConvMixer-768-32
│           ├── args.yaml
│           └── summary.csv
└── pytorch-image-models
    └── (third-party library; omitted)

10 directories, 29 files
```

