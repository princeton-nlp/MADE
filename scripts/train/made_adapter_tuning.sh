#!/bin/bash

set -e

SRC=$1

python run.py \
    --train_on $SRC \
    --adapter \
    --adapter_name $SRC \
    --epochs 10 \
    --max_train_examples 75000 \
    --max_dev_examples 1000 \
    --learning_rate 1e-4 \
    --criterion "f1" \
    --negative_examples \
    --bucket_sampler \
    --load_from "output/train/made" \
    --freeze_transformer \
    --save \
    --full_eval_after_training \
    --name "train/made_adapter_tuning/${SRC}";
