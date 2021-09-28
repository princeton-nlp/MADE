# Head-based models for QA.
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdapterConfig, RobertaModel, RobertaTokenizerFast
import transformers.adapters.composition as ac
from tqdm import tqdm

from src.utils import data_utils, logging, model_utils
from src.models.trainer import Trainer

logger = logging.get_logger(__name__)


@dataclass
class QAOutputs:
    loss: torch.Tensor = field(default=None)
    start_log_probs: torch.Tensor = field(default=None)
    end_log_probs: torch.Tensor = field(default=None)
    details: Any = field(default=None)
    name: Any = field(default=0)


class QAModel(nn.Module):
    def __init__(self, args, tokenizer, adapter=None):
        super().__init__()
        self.model = RobertaModel.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.head = nn.Linear(self.model.config.hidden_size, 2)
        if (adapter is not None and adapter) or (
            adapter is None and args.adapter
        ):
            logger.info(f"adding adapter")
            config = AdapterConfig.load("houlsby")
            self.model.add_adapter(args.adapter_name, config)
            self.model.train_adapter(args.adapter_name)
            self.model.set_active_adapters(args.adapter_name)
            if args.train_on and not args.freeze_transformer:
                logger.info(f"unfreezing transformer")
                self.model.freeze_model(freeze=False)

    def forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        model_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}
        model_outputs = self.model(**model_inputs, return_dict=True)
        logits = self.head(model_outputs.last_hidden_state)
        start_logits, end_logits = [
            l.squeeze(-1).masked_fill(~inputs["context_mask"], -1e9)
            for l in logits.split(1, dim=-1)
        ]
        start_log_probs = F.log_softmax(start_logits, dim=1)
        end_log_probs = F.log_softmax(end_logits, dim=1)
        outputs = QAOutputs(
            start_log_probs=start_log_probs, end_log_probs=end_log_probs
        )
        if labels is not None:
            starts = labels["start"]
            ends = labels["end"]
            p_starts = torch.gather(start_log_probs, 1, starts)
            p_ends = torch.gather(end_log_probs, 1, ends)
            p_answers = p_starts + p_ends
            mask = labels["start_mask"]
            reduced_log_probs = model_utils.marginal_ll(p_answers, mask)
            outputs.loss = -reduced_log_probs
        return outputs


class MADEModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.model = RobertaModel.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.names = args.adapter_names
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(self.model.config.hidden_size, 2)
                for name in self.names
            }
        )
        logger.info(f"adding {len(self.names)} adapters")
        config = AdapterConfig.load("houlsby")
        for name in self.names:
            self.model.add_adapter(name, config)
            self.model.train_adapter(name)
        self.parallel = args.parallel_adapters

    def forward(self, **inputs):
        if self.parallel:
            return self.parallel_forward(**inputs)
        return self.batch_forward(**inputs)

    def batch_forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        datasets = inputs.pop("datasets")

        model_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}

        batch_sizes = list(Counter(datasets).values())
        adapters = list(Counter(datasets).keys())
        if len(adapters) == 1:
            self.model.set_active_adapters(adapters[0])
        else:
            self.model.active_adapters = ac.BatchSplit(
                *adapters, batch_sizes=batch_sizes
            )

        model_outputs = self.model(**model_inputs, return_dict=True)

        parts = model_outputs.last_hidden_state.split(batch_sizes, dim=0)
        logit_lst = []
        for name, hidden_state in zip(adapters, parts):
            logit_lst.append(self.heads[name](hidden_state))
        logits = torch.cat(logit_lst, dim=0)

        start_logits, end_logits = [
            l.squeeze(-1).masked_fill(~inputs["context_mask"], -1e9)
            for l in logits.split(1, dim=-1)
        ]
        start_log_probs = F.log_softmax(start_logits, dim=1)
        end_log_probs = F.log_softmax(end_logits, dim=1)
        outputs = QAOutputs(
            start_log_probs=start_log_probs, end_log_probs=end_log_probs
        )
        if labels is not None:
            starts = labels["start"]
            ends = labels["end"]
            p_starts = torch.gather(start_log_probs, 1, starts)
            p_ends = torch.gather(end_log_probs, 1, ends)
            p_answers = p_starts + p_ends
            mask = labels["start_mask"]
            reduced_log_probs = model_utils.marginal_ll(p_answers, mask)
            outputs.loss = -reduced_log_probs
        return outputs

    def parallel_forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        model_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask")}

        adapters = self.names
        self.model.active_adapters = ac.Parallel(*adapters)
        model_outputs = self.model(**model_inputs, return_dict=True)
        B, N = inputs["input_ids"].shape
        K = len(adapters)
        hidden_states = model_outputs.last_hidden_state
        logit_lst = []
        # (B * K, N, 2) -> [(B, N, 2)] x K
        for adapter, hidden_state in zip(
            adapters, hidden_states.split(B, dim=0)
        ):
            logit_lst.append(self.heads[adapter](hidden_state))

        # (B, K, N, 2)
        logits = torch.stack(logit_lst, dim=1)

        start_logits, end_logits = [
            l.squeeze(-1).masked_fill(
                ~inputs["context_mask"].unsqueeze(1), -1e9
            )
            for l in logits.split(1, dim=-1)
        ]
        start_log_probs = F.log_softmax(start_logits, dim=-1)
        end_log_probs = F.log_softmax(end_logits, dim=-1)

        expert_outputs = [
            QAOutputs(
                start_log_probs=s.squeeze(1),
                end_log_probs=e.squeeze(1),
                name=a,
            )
            for s, e, a in zip(
                start_log_probs.split(1, dim=1),
                end_log_probs.split(1, dim=1),
                adapters,
            )
        ]

        gate_log_probs = math.log(1 / len(adapters))
        gated_start_log_probs = torch.logsumexp(
            start_log_probs + gate_log_probs, dim=1
        )
        gated_end_log_probs = torch.logsumexp(
            end_log_probs + gate_log_probs, dim=1
        )

        expert_outputs.append(
            QAOutputs(
                start_log_probs=gated_start_log_probs,
                end_log_probs=gated_end_log_probs,
            )
        )

        if labels is not None:
            lst = (
                expert_outputs
                if "is_prediction" in inputs
                else [expert_outputs[-1]]
            )
            for o in lst:
                starts = labels["start"]
                ends = labels["end"]
                p_starts = torch.gather(o.start_log_probs, 1, starts)
                p_ends = torch.gather(o.end_log_probs, 1, ends)
                p_answers = p_starts + p_ends
                mask = labels["start_mask"]
                reduced_log_probs = model_utils.marginal_ll(p_answers, mask)
                o.loss = -reduced_log_probs

        outputs = expert_outputs.pop()
        outputs.details = {
            "expert_outputs": expert_outputs,
        }

        return outputs


class QATrainer(Trainer):
    def initialize(self, args):
        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.model_name_or_path
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": data_utils.mrqa_tokens()}
        )
        if args.made:
            model = MADEModel(args, tokenizer)
        else:
            model = QAModel(args, tokenizer)
        return tokenizer, model

    def get_predictions(self, args, model, batch, outputs, tokenizer):
        predictions = []
        # Suppress no_answer prediction
        outputs.start_log_probs[:, 0] = -1e9
        outputs.end_log_probs[:, 0] = -1e9
        # (batch_size, 2)
        best_spans = model_utils.best_spans(
            outputs.start_log_probs, outputs.end_log_probs
        )
        details = []
        experts = []
        if outputs.details and "expert_outputs" in outputs.details:
            expert_predictions = [
                self.get_predictions(args, model, batch, o, tokenizer)
                for o in outputs.details["expert_outputs"]
            ]
            for i in range(len(batch["examples"])):
                es = {}
                for o, e in zip(
                    outputs.details["expert_outputs"], expert_predictions
                ):
                    es[o.name] = {k: v for k, v in e[i].items() if k != "e"}
                experts.append(es)

        for i, e in enumerate(batch["examples"]):
            p_start, p_end = model_utils.to_list(best_spans[i])

            p_text = tokenizer.decode(
                batch["input_ids"][i][p_start : p_end + 1],
                skip_special_tokens=True,
            )

            start, end = model_utils.expand_span(
                batch["input_ids"][i], p_start, p_end, tokenizer
            )
            text = tokenizer.decode(
                batch["input_ids"][i][start : end + 1], skip_special_tokens=True
            )

            score = (
                outputs.start_log_probs[i][p_start]
                + outputs.end_log_probs[i][p_end]
            ).item()

            p = {
                "e": e,
                "feature_id": batch["feature_id"][i],
                "predicted": text,
                "full": p_text,
                "score": score,
                "span": (p_start, p_end),
                "loss": outputs.loss[i].item(),
                "name": outputs.name,
            }
            if experts:
                p["experts"] = experts[i]
            if details:
                p["details"] = details[i]
            predictions.append(p)
        return predictions

    def get_inputs(self, args, batch, **kwargs):
        inputs = {
            k: batch[k]
            for k in ("input_ids", "attention_mask", "labels", "context_mask")
        }
        if args.made:
            inputs["datasets"] = [e["dataset"] for e in batch["examples"]]
        return inputs
