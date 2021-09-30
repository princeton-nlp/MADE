# MADE (**M**ulti-**A**dapter **D**ataset **E**xperts)

This repository contains the implementation of MADE
(**M**ulti-**a**dapter **d**ataset **e**xperts), which is described in
the paper [Single-dataset Experts for Multi-dataset Question Answering](https://arxiv.org/pdf/2109.13880.pdf).

MADE combines a shared Transformer with a collection of adapters that are specialized to different reading comprehension datasets. See our paper for details.


## Quick links

* [Requirements](#requirements)
* [Download the data](#download-the-data)
* [Download trained models](#download-the-trained-models)
* [Run the model](#run-the-model)
  * [Train](#train)
  * [Evaluate](#evaluate)
  * [Transfer](#transfer)
* [Bugs or questions?](#bugs-or-questions)
* [Citation](#citation)


## Requirements

The code uses Python 3.8, PyTorch, and the
[adapter-transformers](https://github.com/Adapter-Hub/adapter-transformers)
library. Install the requirements with:
```
pip install -r requirements.txt
```

## Download the data

You can download the datasets used in the paper from the repository
for the [MRQA 2019 shared
task](https://github.com/mrqa/MRQA-Shared-Task-2019#download-scripts).

The datasets should be stored in directories ending with `train` or
`dev`. For example, download the in-domain training datasets to a
directory called `data/train/` and download the in-domain development
datasets to `data/dev/`.

For zero-shot and few-shot experiments, download the MRQA
out-of-domain development datasets to a separate directory and split
them into training and development splits using
[scripts/split_datasets.py](scripts/split_datasets.py).
For example, download the datasets to `data/transfer/` and run
```sh
ls data/transfer/* -1 | xargs -l python scripts/split_datasets.py
```
Use the default random seed (13) to replicate the splits used in the paper.


## Download the trained models

The trained models are stored on the HuggingFace model hub at this
URL: <https://huggingface.co/princeton-nlp/MADE>. 
All of the models are based on the RoBERTa-base model. They are:
* [MADE Transformer](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_transformer/model.pt)
* MADE adapters (with and without separately tuning the adapters on each dataset).
  * SQuAD ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/SQuAD/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/SQuAD/model.pt))
  * HotpotQA ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/HotpotQA/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/HotpotQA/model.pt))
  * TriviaQA ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/TriviaQA/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/TriviaQA/model.pt))
  * NewsQA ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/NewsQA/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/NewsQA/model.pt))
  * SearchQA ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/SearchQA/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/SearchQA/model.pt))
  * NaturalQuestions ([with adapter tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/NaturalQuestions/model.pt) | [without adapter tuning](https://huggingface.co/princeton-nlp/MADE/blob/main/made_adapters/NaturalQuestions/model.pt))
* [Multi-dataset fine-tuning](https://huggingface.co/princeton-nlp/MADE/resolve/main/multi_dataset_ft/model.pt)
* Single-dataset fine-tuning
  * [SQuAD](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/SQuAD/model.pt)
  * [HotpotQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/HotpotQA/model.pt)
  * [TriviaQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/TriviaQA/model.pt)
  * [NewsQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/NewsQA/model.pt)
  * [SearchQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/SearchQA/model.pt)
  * [NaturalQuestions](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_ft/NaturalQuestions/model.pt)
* Single-dataset adapters
  * [SQuAD](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/SQuAD/model.pt)
  * [HotpotQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/HotpotQA/model.pt)
  * [TriviaQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/TriviaQA/model.pt)
  * [NewsQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/NewsQA/model.pt)
  * [SearchQA](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/SearchQA/model.pt)
  * [NaturalQuestions](https://huggingface.co/princeton-nlp/MADE/resolve/main/single_dataset_adapters/NaturalQuestions/model.pt)

To download just the MADE Transformer and adapters:
```sh
mkdir made_transformer
wget https://huggingface.co/princeton-nlp/MADE/resolve/main/made_transformer/model.pt -O made_transformer/model.pt

mkdir made_tuned_adapters
for d in SQuAD HotpotQA TriviaQA SearchQA NewsQA NaturalQuestions; do
  mkdir "made_tuned_adapters/${d}"
  wget "https://huggingface.co/princeton-nlp/MADE/resolve/main/made_tuned_adapters/${d}/model.pt" -O "made_tuned_adapters/${d}/model.pt"
done;
```

You can download all of the models at once by cloning the repository
(first installing [Git LFS](https://git-lfs.github.com/)):
```
git lfs install
git clone https://huggingface.co/princeton-nlp/MADE
mv MADE models
```

# Run the model

The scripts in [scripts/train/](./scripts/train/) and
[scripts/transfer/](./scripts/transfer/) provide examples of how to
run the code. For more details, see the descriptions of the command
line flags in [run.py](./run.py).

## Train

You can use the scripts in [scripts/train/](./scripts/train/) to train
models on the MRQA datasets. For example, to train MADE:
```
./scripts/train/made_training.sh
```
And to tune the MADE adapters separately on individual datasets:
```
for d in SQuAD HotpotQA TriviaQA SearchQA NewsQA NaturalQuestions; do
  ./scripts/train/made_adapter_tuning.sh $d
done;
```
See [run.py](run.py) for details about the command line arguments.

## Evaluate

A single fine-tuned model:
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --load_from multi_dataset_ft \
    --output_dir output/zero_shot/multi_dataset_ft
```

An individual MADE adapter (e.g. SQuAD):
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --load_from made_transformer \
    --load_adapters_from made_tuned_adapters \
    --adapter \
    --adapter_name SQuAD \
    --output_dir output/zero_shot/made_tuned_adapters/SQuAD
```

An individual single-dataset adapter (e.g. SQuAD):
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --load_adapters_from single_dataset_adapters/ \
    --adapter \
    --adapter_name SQuAD \
    --output_dir output/zero_shot/single_dataset_adapters/SQuAD
```

An ensemble of MADE adapters. This will run a forward pass through
every adapter [in
parallel](https://docs.adapterhub.ml/adapter_composition.html#parallel).
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --load_from made_transformer \
    --load_adapters_from made_tuned_adapters \
    --adapter_names SQuAD HotpotQA TriviaQA SearchQA NewsQA NaturalQuestions \
    --made \
    --parallel_adapters  \
    --output_dir output/zero_shot/made_ensemble
```

Averaging the parameters of the MADE adapters:
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --load_from made_transformer \
    --load_adapters_from made_tuned_adapters \
    --adapter_names SQuAD HotpotQA TriviaQA SearchQA NewsQA NaturalQuestions \
    --adapter \
    --average_adapters  \
    --output_dir output/zero_shot/made_avg
```

Running UnifiedQA:
```sh
python run.py \
    --eval_on BioASQ DROP DuoRC RACE RelationExtraction TextbookQA \
    --seq2seq \
    --model_name_or_path allenai/unifiedqa-t5-base \
    --output_dir output/zero_shot/unifiedqa
```


## Transfer

The scripts in [scripts/transfer/](./scripts/transfer/) provide
examples of how to run the few-shot transfer learning experiments
described in the paper. For example, the following command will repeat
for three random seeds: (1) sample 64 training examples from BioASQ,
(2) calculate the zero-shot loss of all the MADE adapters on the
training examples, (3) average the adapter parameters in proportion to
zero-shot loss, (4) hold out 32 training examples for validation data,
(5) train the adapter until performance stops improving on the 32
validation examples, and (6) evaluate the adapter on the full
development set.
```sh
python run.py \
    --train_on BioASQ \
    --adapter_names SQuAD HotpotQA TriviaQA NewsQA SearchQA NaturalQuestions \
    --made \
    --parallel_made \
    --weighted_average_before_training \
    --adapter_learning_rate 1e-5 \
    --steps 200 \
    --patience 10 \
    --eval_before_training \
    --full_eval_after_training \
    --max_train_examples 64 \
    --few_shot \
    --criterion "loss" \
    --negative_examples \
    --save \
    --seeds 7 19 29 \
    --load_from "made_transformer" \
    --load_adapters_from "made_tuned_adapters" \
    --name "transfer/made_preaverage/BioASQ/64"
```

# Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Dan Friedman (`dfriedman@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

# Citation

```bibtex
@inproceedings{friedman2021single,
   title={Single-dataset Experts for Multi-dataset QA},
   author={Friedman, Dan and Dodge, Ben and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```
