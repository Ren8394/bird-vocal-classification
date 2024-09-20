#!bin/bash

export CUDA_VISIBLE_DEVICES=0

python finetune.py --train --inference \
    --weight /media/birdsong/disk02/bird-vocal-classification/weights/pretrain/AudioMAE/mr08/lr0002_wd0001_b64_e32/best_e16.pth \
    --batch_size 512 --lr 0.001 --weight_decay 0.0005