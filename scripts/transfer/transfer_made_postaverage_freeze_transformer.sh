#!/bin/bash

for k in 16 64 256; do
    for t in BioASQ DROP DuoRC RACE RelationExtraction TextbookQA; do
	srun python run.py \
	    --train_on ${t} \
	    --adapter_names SQuAD HotpotQA TriviaQA NewsQA SearchQA NaturalQuestions \
	    --made \
	    --parallel_adapters \
	    --weighted_average_after_training \
	    --train_batch_size 1 \
	    --gradient_accumulation_steps 8 \
	    --learning_rate 1e-5 \
	    --adapter_learning_rate 1e-5 \
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
	    --freeze_transformer \
	    --separate_adapter_checkpoints \
	    --seeds 7 19 29 \
	    --load_from "MADE/made_transformer" \
	    --load_adapters_from "MADE/made_tuned_adapters" \
	    --name "transfer/made_postaverage_freeze_transformer"
    done;
done;
