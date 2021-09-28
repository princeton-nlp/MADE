# Sequence-to-sequence model for QA.
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
import transformers.adapters.composition as ac
from tqdm import tqdm

from src.utils import data_utils, logging, model_utils
from src.models.trainer import Trainer

logger = logging.get_logger(__name__)


@dataclass
class SeqOutputs:
    loss: torch.Tensor = field(default=None)
    details: Any = field(default=None)


class SeqModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path
        )
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, **inputs):
        mask = inputs["labels"]["text_mask"]
        labels = inputs["labels"]["text"].masked_fill(~mask, -100)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            return_dict=True,
        )

        return SeqOutputs(loss=outputs.loss)

    def _forward(self, **inputs):
        mask = inputs["labels"]["text_mask"]
        labels = inputs["labels"]["text"].masked_fill(~mask, -100)
        batch_size = labels.shape[0]
        zero = torch.zeros(
            batch_size, 1, device=labels.device, dtype=torch.long
        )
        decoder_input_ids = torch.cat([zero, inputs["labels"]["text"]], dim=-1)

        one = torch.ones(batch_size, 1, device=labels.device, dtype=torch.bool)
        decoder_attention_mask = torch.cat([one, mask], dim=-1)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

        pad = torch.full(
            (batch_size, 1), -100, device=labels.device, dtype=torch.long
        )
        padded_labels = torch.cat([labels, pad], dim=-1)
        logits = outputs.logits
        unwrapped_labels = padded_labels.view(-1)
        unwrapped_logits = logits.view(-1, logits.shape[-1])
        unwrapped_token_loss = F.cross_entropy(
            unwrapped_logits, unwrapped_labels, reduction="none"
        )
        token_loss = unwrapped_token_loss.view_as(padded_labels)[:, :-1]
        sequence_losses = torch.sum(token_loss, dim=1)

        return SeqOutputs(loss=sequence_losses)


class SeqTrainer(Trainer):
    def initialize(self, args):
        tokenizer = T5TokenizerFast.from_pretrained("t5-base")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": data_utils.mrqa_tokens()}
        )
        model = SeqModel(args, tokenizer)
        return tokenizer, model

    def get_predictions(self, args, model, batch, outputs, tokenizer):
        predictions = []
        inputs = self.get_inputs(args, batch)
        model_utils.to_device(inputs)
        out = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=True,
            min_length=2,
        )
        sequences = tokenizer.batch_decode(
            out["sequences"], skip_special_tokens=True
        )
        detailed_sequences = tokenizer.batch_decode(
            out["sequences"], skip_special_tokens=False
        )
        labels = tokenizer.batch_decode(
            batch["labels"]["text"], skip_special_tokens=False
        )

        stacked_scores = torch.stack(out["scores"], dim=1)
        token_scores = torch.gather(
            stacked_scores, -1, out["sequences"][:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        token_scores = token_scores.masked_fill(
            out["sequences"][:, 1:] == tokenizer.pad_token_id, 0
        )
        sequence_scores = torch.sum(token_scores, dim=-1)

        for i, e in enumerate(batch["examples"]):
            predictions.append(
                {
                    "e": e,
                    "feature_id": batch["feature_id"][i],
                    "predicted": sequences[i],
                    "score": sequence_scores[i].item(),
                    "loss": outputs.loss.item(),
                    # "loss": outputs.loss[i].item(),
                    "full": detailed_sequences[i],
                    "labels": labels[i],
                }
            )
        return predictions

    def get_inputs(self, args, batch, **kwargs):
        inputs = {
            k: batch[k] for k in ("input_ids", "attention_mask", "labels")
        }
        return inputs
