#!/bin/bash

set -e

SRC=$1

srun python src/run.py \
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
    --freeze_transformer \
    --full_eval_after_training \
    --save \
    --name "train/single_dataset_adapters/${SRC}";
