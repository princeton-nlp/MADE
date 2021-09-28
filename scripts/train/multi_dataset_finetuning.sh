#!/bin/bash

python run.py \
    --train_on SQuAD HotpotQA TriviaQA NewsQA SearchQA NaturalQuestions \
    --epochs 3 \
    --max_train_examples 75000 \
    --max_dev_examples 1000 \
    --learning_rate 1e-5 \
    --criterion "f1" \
    --negative_examples \
    --dataset_mixer_sampler \
    --full_eval_after_training \
    --save \
    --name "train/multi_dataset_ft";
