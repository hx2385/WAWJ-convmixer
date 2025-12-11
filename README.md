# Patches Are All You Need? 
This repository contains a reproduction of ConvMixer for the ICLR 2022 submission ["Patches Are All You Need?"](https://openreview.net/forum?id=TVHS5Y4dNvM) by Haipei Xu and Weixi Huang

## Data
In this project, we used two Dataset, ImageNet-1K and CIFAR-10》 The former can be downloaded via Huggingface through this [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), while the latter will be automatically downloaded by train.py in the CIFAR folder. Due to the computation limitation, we only use the first 100 classes (10%) of the original ImageNet-1K dataset in our experiments.

## Code overview
The most important code is in `convmixer.py`, which is implemented by original authors.

## Evaluation

## Training
Since we only have one card (T4), so we trained ImageNet-1K as following (The file train.py is located at /CIFAR-10/train.py):

```
python train.py [/path/to/ImageNet1k] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model convmixer_1536_20 
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

__**Note:**__ While training on `ConvMixer-1536/20`, in order to not cauing `OOM`, we used batch size (-b) of 16 instead.


