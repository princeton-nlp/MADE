"""Adapted evaluation script from the MRQA Workshop Shared Task.
Adapted from the SQuAD v1.1 official evaluation script.
"""
import argparse
import collections
from collections import Counter
import json
import re
import string

import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def score_prediction(p, e=None):
    if e is None:
        e = p["e"]
    p["em"] = metric_max_over_ground_truths(
        exact_match_score, p["predicted"], e["valid_answers"]
    )
    p["f1"] = metric_max_over_ground_truths(
        f1_score, p["predicted"], e["valid_answers"]
    )
    return p


def score_predictions(all_predictions):
    expert_names = all_predictions[0].get("experts", None)
    qid_to_preds = collections.defaultdict(list)
    for p in all_predictions:
        e = p["e"]
        qid_to_preds[e["qid"]].append(score_prediction(p))
        if expert_names:
            for ep in p["experts"].values():
                score_prediction(ep, e=e)
    predictions = []
    for qid_preds in qid_to_preds.values():
        pred = max(qid_preds, key=lambda p: p["score"])
        if expert_names:
            experts = pred["experts"]
            for e in list(experts.keys()):
                e_preds = [p["experts"][e] for p in qid_preds]
                experts[e] = max(e_preds, key=lambda p: p["score"])
        predictions.append(pred)
    report = {}
    for k in ("em", "f1", "loss"):
        report[k] = np.mean([p[k] for p in predictions])
    if expert_names:
        report["experts"] = {}
        for e in expert_names:
            report["experts"][e] = {}
            for k in ("em", "f1", "loss"):
                report["experts"][e][k] = np.mean(
                    [p["experts"][e][k] for p in predictions]
                )
    return report, predictions


def average_dicts(dicts, short=False):
    avg = {}
    for k, v in dicts[0].items():
        vs = [d[k] for d in dicts if k in d]
        if type(v) == dict:
            avg[k] = average_dicts(vs)
        elif type(v) in (int, float, np.float64):
            avg[k] = np.mean(vs)
            avg[f"{k}_std"] = np.std(vs)
            if not short:
                avg[f"{k}_lst"] = vs
        elif type(vs[0]) == list and len(vs[0]) == 1:
            avg[k] = [v[0] for v in vs]
        else:
            avg[k] = vs
    return avg
