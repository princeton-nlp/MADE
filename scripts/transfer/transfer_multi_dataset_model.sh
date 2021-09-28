#!/bin/bash

for k in 16 64 256; do
    for t in BioASQ DROP DuoRC RACE RelationExtraction TextbookQA; do
	srun python run.py \
	    --train_on ${t} \
	    --learning_rate 1e-5 \
	    --epochs 10 \
	    --steps 200 \
	    --patience 10 \
	    --eval_before_training \
	    --full_eval_after_training \
	    --few_shot \
	    --max_train_examples ${k} \
	    --criterion "loss" \
	    --negative_examples \
	    --save \
	    --seeds 7 19 29 \
	    --load_from "MADE/multi_dataset_ft" \
	    --name "transfer/multi_dataset_ft/${t}/${k}"
    done;
done;
