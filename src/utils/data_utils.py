import collections
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pickle
import random

import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from src.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class SpanAnswer:
    text: str
    l: int
    r: int


@dataclass
class Example:
    dataset: str
    qid: str
    question: str
    context: str
    valid_answers: List[str]
    detected_answers: List[Any]


@dataclass
class Feature:
    qid: str
    feature_id: str
    input_ids: List[int]
    labels: Any


def load_mrqa_examples(fn):
    path = Path(fn).with_suffix(".jsonl")
    logger.info(f"loading examples from {path}")
    examples = []
    dataset = path.stem
    with open(path, "r") as f:
        lines = f.readlines()[1:]
    ds = [json.loads(l) for l in lines]
    examples = []
    for i, d in enumerate(ds):
        context = d["context"]
        for qa in d["qas"]:
            question = qa["question"]
            valid_answers = list(set(qa["answers"]))
            detected_spans = [
                (a["text"], s[0], s[1])
                for a in qa["detected_answers"]
                for s in a["char_spans"]
            ]
            detected_answers = [
                asdict(SpanAnswer(t, l, r)) for t, l, r in set(detected_spans)
            ]
            examples.append(
                Example(
                    dataset=dataset,
                    qid=qa["qid"],
                    question=question,
                    context=context,
                    valid_answers=valid_answers,
                    detected_answers=detected_answers,
                )
            )
    return [asdict(e) for e in examples]


def char_span_to_token_span(offsets_mapping, l, r, context_start):
    tl = tr = None
    for i in range(context_start, len(offsets_mapping)):
        cl, cr = offsets_mapping[i]
        if l >= cl and l <= cr:
            tl = i
        if r >= cl and r <= cr:
            tr = i
            break
    return tl, tr


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    stride=128,
):
    logger.info(f"tokenizing {len(examples)} examples")
    features = []
    skip_count = 0
    is_uqa = "T5" in tokenizer.__class__.__name__
    for e in examples:
        try:
            q = e["question"].lower() + " \\" if is_uqa else e["question"]
            c = e["context"].lower() if is_uqa else e["context"]
            encoded = tokenizer(
                q,
                text_pair=c,
                padding=False,
                truncation="only_second",
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
        except Exception as ex:
            logger.warning(f"error tokenizing {e['qid']}: {ex}")
            skip_count += 1
            continue
        eos = tokenizer.eos_token_id
        for j, (input_ids, offsets) in enumerate(
            zip(encoded["input_ids"], encoded["offset_mapping"])
        ):
            context_start = input_ids.index(tokenizer.eos_token_id) + (
                2 if "Roberta" in tokenizer.__class__.__name__ else 1
            )
            if is_uqa:
                # Hack--final input is "{question} \\<sep>{context}",
                # remove <sep> between question and context and replace with n
                input_ids[context_start - 1] = tokenizer.convert_tokens_to_ids(
                    "n"
                )

            feature_id = f"{e['qid']}.{j}"
            starts, ends, text_labels = [], [], []
            for span in e["detected_answers"]:
                tl, tr = char_span_to_token_span(
                    offsets, span["l"], span["r"], context_start
                )
                if tl is not None and tr is not None:
                    starts.append(tl)
                    ends.append(tr)
                    text_labels.append(
                        tokenizer(span["text"].lower()).input_ids
                    )
                labels = {"start": starts, "end": ends}
                if text_labels:
                    labels["text"] = text_labels[0]
                else:
                    labels["text"] = tokenizer("").input_ids
            features.append(Feature(e["qid"], feature_id, input_ids, labels))
    logger.info(
        f"converted {len(examples) - skip_count}/{len(examples)} examples "
        f"into {len(features)} features"
    )
    return [asdict(f) for f in features]


def mrqa_tokens():
    return ["[TLE]", "[DOC]", "[PAR]"]


def cache_fn(name, split, tokenizer, cache_dir="cache"):
    return (Path(cache_dir) / split / name).with_suffix(
        "." + tokenizer.__class__.__name__ + ".pkl"
    )


def post_process(features, tokenizer):
    for f in features:
        context_start = f["input_ids"].index(tokenizer.eos_token_id) + (
            2 if "Roberta" in tokenizer.__class__.__name__ else 1
        )
        context_mask = torch.zeros(len(f["input_ids"]), dtype=torch.bool)
        context_mask[0] = True
        context_mask[context_start:] = True
        f["context_mask"] = context_mask
        if (
            f["labels"].get("text", [])
            and not f["labels"]["text"][-1] == tokenizer.eos_token_id
        ):
            f["labels"]["text"].append(tokenizer.eos_token_id)
    return features


def load_dataset(
    name,
    split,
    tokenizer,
    max_examples=None,
    negative_examples=False,
    overwrite_cache=False,
    seed=13,
    data_dir="data",
    cache_dir="cache",
):
    fn = Path(data_dir) / split / name
    cache = cache_fn(name, split, tokenizer)
    if cache_dir and cache.exists() and not overwrite_cache:
        logger.info(f"loading cached dataset from {cache}")
        with open(cache, "rb") as f:
            examples, features = pickle.load(f)
    else:
        examples = load_mrqa_examples(fn)
        features = convert_examples_to_features(examples, tokenizer)
        if cache_dir:
            logger.info(f"caching dataset to {cache}")
            cache.parent.mkdir(exist_ok=True, parents=True)
            with open(cache, "wb") as f:
                pickle.dump((examples, features), f)

    count = 0
    for f in features:
        no_answer = int(len(f["labels"]["start"]) == 0)
        count += no_answer
        f["labels"]["no_answer"] = no_answer
        if no_answer:
            f["labels"]["start"].append(0)
            f["labels"]["end"].append(0)
    if (max_examples or 0) > 0 and max_examples < len(examples):
        random.seed(seed)
        examples = random.sample(examples, max_examples)
        qids = set(e["qid"] for e in examples)
        features = [f for f in features if f["qid"] in qids]
        logger.info(
            f"picked {len(examples)} examples / {len(features)} features"
        )

    post_process(features, tokenizer)

    dataset = GenericDataset(examples, features)
    return dataset


def collate(batch):
    examples, features = zip(*batch)
    d = {"examples": examples}
    d.update(collate_(features))
    return d


def collate_(features):
    d = {}
    for k in features[0].keys():
        values = [f[k] for f in features]
        if type(values[0]) == str:
            d[k] = values
        elif type(values[0]) in (int, float):
            d[k] = torch.tensor(values)
        elif type(values[0]) == list:
            max_len = max(len(v) for v in values)
            mask_k = "attention_mask" if k == "input_ids" else f"{k}_mask"
            d[k] = torch.full((len(features), max_len), 0, dtype=torch.long)
            d[mask_k] = torch.full(
                (len(features), max_len), 0, dtype=torch.bool
            )
            for i, v in enumerate(values):
                t = torch.tensor(v, dtype=torch.long)
                m = torch.ones_like(t, dtype=torch.bool)
                d[k][i, : len(t)] = t
                d[mask_k][i, : len(m)] = m
        elif type(values[0]) == dict:
            d[k] = collate_(values)
        elif type(values[0]) == torch.Tensor:
            d[k] = pad_sequence(values, batch_first=True)
        else:
            raise NotImplementedError(type(values[0]))
    return d


class GenericDataset:
    def __init__(self, examples, features):
        self.name = examples[0]["dataset"]
        self.examples = examples
        self.features = features
        self.qid_to_e = {e["qid"]: e for e in examples}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        e = self.qid_to_e[f["qid"]]
        return e, f

    def split(self, train_negative_examples=False):
        qids = list(self.qid_to_e.keys())
        random.shuffle(qids)
        a_qids = set(qids[: len(qids) // 2])
        a_features = [f for f in self.features if f["qid"] in a_qids]
        if not train_negative_examples:
            prev = len(a_features)
            a_features = [f for f in a_features if not f["labels"]["no_answer"]]
            logger.info(
                f"filtering {prev - len(a_features)}/{prev} training features "
                f"with no answer"
            )
        a_examples = [e for e in self.examples if e["qid"] in a_qids]
        b_features = [f for f in self.features if f["qid"] not in a_qids]
        b_examples = [e for e in self.examples if e["qid"] not in a_qids]
        return GenericDataset(a_examples, a_features), GenericDataset(
            b_examples, b_features
        )


class GenericDatasets:
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = [0]
        for d in datasets:
            self.offsets.append(self.offsets[-1] + len(d))

    @property
    def examples(self):
        return (e for d in self.datasets for e in d.examples)

    @property
    def features(self):
        return (f for d in self.datasets for f in d.features)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if idx < self.offsets[i + 1]:
                return self.datasets[i][idx - self.offsets[i]]
        raise ValueError(idx)


def maybe_filter_negative_examples(dataset, do_filter=True):
    if hasattr(dataset, datasets):
        return GenericDatasets(
            [maybe_filter_negative_examples(d, do_filter) for d in d.datasets]
        )
    count = sum(int(f["labels"]["no_answer"]) for f in dataset.features)
    total = len(dataset.features)
    if negative_examples:
        logger.info(
            f"[{dataset.name}] including {count}/{total} features "
            " with no answer"
        )
        return dataset
    else:
        logger.info(f"filtering {count}/{total} features with no answer")
        features = [f for f in dataset.features if not f["labels"]["no_answer"]]
        return GenericDataset(dataset.examples, features)


def dynamic_sampling_weights():
    return {
        "SQuAD": 0.906 + 0.836,
        "HotpotQA": 0.797 + 0.664,
        "TriviaQA": 0.779 + 0.734,
        "NewsQA": 0.551 + 0.724,
        "SearchQA": 0.847 + 0.796,
        "NaturalQuestions": 0.815 + 0.704,
    }


class WeightedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets=1):
        assert hasattr(dataset, "datasets")
        self.dataset_names = [d.name for d in dataset.datasets]
        self.offsets = dict(zip(self.dataset_names, dataset.offsets))
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.weights = [1 for _ in range(len(self.dataset_names))]

    def set_weights(self, weights):
        self.weights = [weights[t] for t in self.train_on]

    def set_dynamic_sampling_weights(self, report):
        baseline = dynamic_sampling_weights()
        diffs = []
        for t in self.dataset_names:
            cur = report[t]["em"] + report[t]["f1"]
            diffs.append(abs(baseline[t] - cur))
        new_weights = [d / sum(diffs) for d in diffs]
        logger.info(f"adjusting dynamic sampling weights")
        logger.info(f"current: {dict(zip(self.dataset_names, self.weights))}")
        logger.info(f"new: {dict(zip(self.dataset_names, new_weights))}")
        self.weights = new_weights

    def current_weights(self):
        return dict(zip(self.dataset_names, self.weights))

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.length // self.batch_size


class DatasetMixerSampler(WeightedSampler):
    def __init__(self, dataset, batch_size, sort_datasets=True, num_buckets=1):
        super().__init__(dataset, batch_size, num_buckets)
        self.subsamplers = {
            subdataset.name: RandomSampler(subdataset)
            for subdataset in dataset.datasets
        }
        self.iterators = {
            name: iter(subsampler)
            for name, subsampler in self.subsamplers.items()
        }
        self.sort_datasets = sort_datasets

    def subsample(self, dataset_name):
        e = next(self.iterators[dataset_name], None)
        if e is None:
            self.iterators[dataset_name] = iter(self.subsamplers[dataset_name])
            e = next(self.iterators[dataset_name])
        return e + self.offsets[dataset_name]

    def __iter__(self):
        for _ in range(self.length // self.batch_size):
            datasets = random.choices(
                self.dataset_names, weights=self.weights, k=self.batch_size
            )
            if self.sort_datasets:
                datasets = sorted(datasets)
            batch = [self.subsample(d) for d in datasets]
            yield batch


class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets=8, offset=0):
        lens = [len(f["input_ids"]) for f in dataset.features]
        self.idxs = idxs = [
            offset + i for i in sorted(range(len(lens)), key=lambda i: lens[i])
        ]
        bucket_size = round(len(idxs) / num_buckets)
        self.buckets = [
            idxs[i : i + bucket_size] for i in range(0, len(idxs), bucket_size)
        ]
        self.batch_size = batch_size

    def __iter__(self):
        for bucket in self.buckets:
            random.shuffle(bucket)
        batches = []
        batch = []
        for i in (i for b in self.buckets for i in b):
            batch.append(i)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return round(len(self.idxs) / self.batch_size)
