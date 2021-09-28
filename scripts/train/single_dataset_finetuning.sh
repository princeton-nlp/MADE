#!/bin/bash

set -e

SRC=$1

python run.py \
    --train_on $SRC \
    --epochs 10 \
    --max_train_examples 75000 \
    --max_dev_examples 1000 \
    --learning_rate 1e-5 \
    --criterion "f1" \
    --negative_examples \
    --bucket_sampler \
    --full_eval_after_training \
    --save \
    --name "train/single_dataset_ft/${SRC}";
