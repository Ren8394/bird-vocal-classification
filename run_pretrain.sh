#!bin/bash

export CUDA_VISIBLE_DEVICES=0,1

python pretrain.py --train --mask_ratio 0.8 --batch_size 64 --lr 0.0002 --weight_decay 0.0001 \
    --ckpt /media/birdsong/disk02/bird-vocal-classification/ckpt/pretrain/AudioMAE/mr08/lr0002_wd0001_b64_e32/best.pth.tar