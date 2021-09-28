import collections
import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from tqdm import tqdm
from transformers import (
    Adafactor,
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src.utils import data_utils, logging, metrics, model_utils
from src.utils.model_utils import device, to_device

logger = logging.get_logger(__name__)


def save(args, model, ckp, adapter=None):
    if args.save_every_best:
        name = f"model.{ckp}.pt"
    elif args.save_epoch_best:
        epoch = ckp.split(".")[0]
        name = f"model.{epoch}.pt"
    else:
        name = "model.pt"
    if adapter:
        p = Path(args.output_dir) / adapter
        if not p.exists():
            p.mkdir(parents=True)
        fn = Path(args.output_dir) / adapter / name
    else:
        fn = Path(args.output_dir) / name
    params = dict(list(model.named_parameters()))
    state_dict = collections.OrderedDict()
    for k, v in model.state_dict().items():
        if adapter and adapter not in k:
            continue
        if k in params and params[k].requires_grad:
            state_dict[k] = v
    logger.info(
        f"saving {adapter or 'model'} to {fn} "
        f"({len(state_dict)}/{len(params)} parameters)"
    )
    torch.save(state_dict, str(fn))


def asdict_(d):
    if type(d) == dict:
        return d
    return asdict(d)


class Trainer:
    def initialize(self, args):
        raise NotImplementedError

    def get_predictions(self, args, model, batch, outputs, tokenizer, **kwargs):
        raise NotImplementedError

    def get_inputs(self, args, batch, **kwargs):
        raise NotImplementedError

    def checkpoint(self, args, model, tokenizer, eval_datasets, ckp, best):
        report = self.evaluate(
            args,
            model,
            tokenizer,
            eval_datasets,
            ckp=ckp,
        )
        if args.criterion == "loss":
            compare = lambda a, b: a < b
        else:
            compare = lambda a, b: a > b

        if args.separate_adapter_checkpoints:
            new_best = False
            new_bests = []
            for a, a_best in best["experts"].items():
                a_report = report["experts"][a]
                a_new_best = False
                if compare(a_report[args.criterion], a_best[args.criterion]):
                    a_best["ckp"] = ckp
                    best["ckp"] = ckp
                    best["patience"] = 0
                    a_best.update(a_report)
                    logger.info(f"new best ({a}): {a_best}")
                    if args.save:
                        save(args, model, ckp, adapter=a)
                    new_best = True
                    a_new_best = True
                new_bests.append(a_new_best)

            if new_best:
                best["ckp"] = ckp
                best["patience"] = 0
                if any(new_bests) and args.separate_model_checkpoint:
                    save(args, model, ckp)
            else:
                best["patience"] += 1
        elif compare(report[args.criterion], best[args.criterion]):
            best["ckp"] = ckp
            best["patience"] = 0
            best.update(report)
            if len(args.train_on) == 1:
                log_report = {
                    k: v
                    for k, v in best.items()
                    if k not in args.train_on + ["experts"]
                }
            else:
                log_report = best
            logger.info(f"new best: {log_report}")
            if args.save:
                save(args, model, ckp)
        else:
            best["patience"] += 1
        stop_early = False
        if (
            args.patience not in (None, -1)
            and best["patience"] >= args.patience
        ):
            stop_early = True
            logger.info(
                f"{args.patience} checkpoints with no improvement," " stopping"
            )
        return best, stop_early, report

    def evaluate_one(
        self, args, model, tokenizer, dataset, eval_dataloader, ckp=""
    ):
        model.eval()
        logger.info(f"evaluating on {dataset}")
        with torch.no_grad():
            t = tqdm(eval_dataloader, desc=f"eval [{ckp}]")
            predictions = []
            eval_loss = 0
            for step, batch in enumerate(t):
                inputs = self.get_inputs(args, batch)
                to_device(inputs)
                inputs["is_prediction"] = True
                outputs = model(**inputs)
                predictions += self.get_predictions(
                    args, model, batch, outputs, tokenizer
                )
                loss = outputs.loss.mean()
                eval_loss += loss.item()
                postfix = {"loss": loss.item()}
                if outputs.details and "kl" in outputs.details:
                    kl = outputs.details["kl"].item()
                    postfix["kl"] = kl
                t.set_postfix(postfix)
            eval_loss = eval_loss / len(eval_dataloader)
            logger.info(f"avg eval loss: {eval_loss}")
        report, predictions = metrics.score_predictions(predictions)
        logger.info(f"{dataset} results at {ckp}: {report}")
        logger.info(f"writing results to {args.output_dir}")
        with open(
            Path(args.output_dir) / f"metrics.{dataset}.{ckp}.json", "w"
        ) as f:
            json.dump(report, f, indent=2)
        pckp = f"{ckp}." if (ckp and not ckp[0].isdigit()) else ""
        with open(
            Path(args.output_dir) / f"predictions.{dataset}.{pckp}json", "w"
        ) as f:
            json.dump(predictions, f, indent=2)
        return report

    def evaluate(self, args, model, tokenizer, eval_datasets, ckp=""):
        logger.info("evaluating")
        eval_dataloaders = {}
        for dataset in args.eval_on:
            eval_dataloader = DataLoader(
                eval_datasets[dataset],
                batch_size=args.eval_batch_size,
                collate_fn=data_utils.collate,
            )
            eval_dataloaders[dataset] = eval_dataloader
        reports = {}
        for dataset, dataloader in eval_dataloaders.items():
            reports[dataset] = self.evaluate_one(
                args, model, tokenizer, dataset, dataloader, ckp
            )
        report = metrics.average_dicts(list(reports.values()), short=True)
        logger.info(
            f"average eval {args.criterion} at {ckp}: {report[args.criterion]}"
        )
        report.update(reports)
        logger.info(f"writing results to {args.output_dir}")
        with open(Path(args.output_dir) / f"metrics.{ckp}.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    def load_from(self, args, path_or_paths, model, **kwargs):
        paths = (
            path_or_paths if type(path_or_paths) == list else [path_or_paths]
        )
        state_dict = {}
        for path in paths:
            if str(path).endswith(".pt"):
                fn = path
            else:
                fns = sorted(
                    Path(path).glob("model*"),
                    key=lambda p: p.lstat().st_ctime,
                )
                if len(fns) == 0:
                    raise ValueError(f"no model.pt in {path}")
                fn = fns[-1]
            logger.info(f"loading checkpoint from {fn}")
            p_state_dict = torch.load(fn, map_location=model_utils.device())
            path_name = Path(path).name
            # rename head parameters to heads.{path_name}
            remap_adapters = (
                args.made
                or args.average_adapters
                or args.weighted_average_before_training
            )
            if remap_adapters and path_name in args.adapter_names:
                remapped = {}
                for k, p in p_state_dict.items():
                    if "head." in k:
                        rk = k.replace("head.", f"heads.{path_name}.")
                        remapped[rk] = p
                logger.info(
                    f"remapping {len(remapped)} adapter parameters from {path}"
                )
                p_state_dict.update(remapped)
            state_dict.update(p_state_dict)

        if args.average_adapters:
            logger.info(f"averaging adapters")
            proportions = None
            avg_dict = model_utils.average_adapter_params(
                args, state_dict, proportions=proportions
            )
            logger.info(f"averaging {len(avg_dict)} parameters")
            state_dict.update(avg_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"{len(missing)} missing, {len(unexpected)} unexpected")
        logger.info(f"missing: {missing}")
        logger.info(f"unexpected: {unexpected}")
        missing = set(missing)
        missing_new = [
            k
            for k, p in model.named_parameters()
            if p.requires_grad and k in missing
        ]
        logger.info(f"missing parameters with requires_grad: {missing_new}")
        return model

    def set_frozen(self, args, model):
        total = frozen = 0
        for k, p in model.named_parameters():
            total += 1
            if model_utils.freeze(args, k):
                p.requires_grad = False
                frozen += 1
            else:
                p.requires_grad = True
        logger.info(f"froze {frozen}/{total} parameters")

    def train(self, args, model, tokenizer, train_dataset, eval_datasets):
        logger.info(f"writing args to {args.output_dir}")
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        if len(args.train_on) > 1:
            sampler = data_utils.DatasetMixerSampler(
                train_dataset,
                batch_size=args.train_batch_size,
            )
        elif args.bucket_sampler:
            sampler = data_utils.BucketSampler(
                train_dataset, batch_size=args.train_batch_size
            )
        else:
            sampler = BatchSampler(
                RandomSampler(train_dataset),
                args.train_batch_size,
                drop_last=False,
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=data_utils.collate,
        )

        model.to(device())

        self.set_frozen(args, model)

        is_adapter_param = lambda k: ".adapters." in k or "head" in k

        if args.adapter or args.made and args.adapter_learning_rate:
            adapter_param_names, adapter_params = zip(
                *[
                    (k, p)
                    for k, p in model.named_parameters()
                    if p.requires_grad and is_adapter_param(k)
                ]
            )
            base_params = [
                p
                for k, p in model.named_parameters()
                if p.requires_grad and not is_adapter_param(k)
            ]
            params = [
                {"params": base_params},
                {"params": adapter_params, "lr": args.adapter_learning_rate},
            ]
            print(
                f"setting lr to {args.adapter_learning_rate} for "
                f"{len(adapter_param_names)} adapter params"
            )
        else:
            params = [p for p in model.parameters() if p.requires_grad]

        if args.optimizer == "adafactor":
            optimizer = Adafactor(
                params,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=args.learning_rate,
            )
        else:
            optimizer = AdamW(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        scheduler = None
        steps_per_epoch = (
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_steps = max(args.steps, args.epochs * steps_per_epoch)
        if args.steps > (args.epochs * steps_per_epoch):
            args.epochs = int(args.steps // steps_per_epoch) + 1
        logger.info(f"training for {num_steps} steps / {args.epochs} epochs")
        warmup_steps = args.warmup_steps or args.warmup_ratio * num_steps
        if args.scheduler == "linear":
            logger.info(
                f"using linear lr schedule,  num_steps: {num_steps}, "
                f"warmup: {warmup_steps}"
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=args.epochs * len(train_dataloader),
            )
        elif args.scheduler == "constant":
            logger.info(
                f"using constant lr schedule,  num_steps: {num_steps}, "
                f"warmup: {warmup_steps}"
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
            )

        lr = args.learning_rate

        best = {
            "ckp": "",
            "em": 0,
            "f1": 0,
            "loss": float("inf"),
            "patience": 0,
            "lr": lr,
        }
        if args.criterion not in best:
            raise NotImplementedError(f"unknown criterion {args.criterion}")
        if args.separate_adapter_checkpoints:
            copies = [copy.copy(best) for _ in args.adapter_names]
            best["experts"] = {a: c for a, c in zip(args.adapter_names, copies)}

        stop_early = False
        global_step = 0

        if args.eval_before_training:
            logger.info(f"evaluating before training")
            best, _, _ = self.checkpoint(
                args,
                model,
                tokenizer,
                eval_datasets,
                ckp="0.0",
                best=best,
            )

        for epoch in range(args.epochs):
            epoch_loss = 0
            checkpoint_loss = 0
            logger.info(f"epoch: {epoch}")
            t = tqdm(train_dataloader, desc=f"train [{epoch}]")
            for step, batch in enumerate(t):
                model.train()
                inputs = self.get_inputs(args, batch)
                to_device(inputs)
                outputs = model(**inputs)
                loss = outputs.loss.mean() / args.gradient_accumulation_steps
                epoch_loss += loss.item()
                checkpoint_loss += loss.item()
                loss.backward()
                postfix = {"loss": loss.item()}
                if outputs.details and "kl" in outputs.details:
                    kl = outputs.details["kl"].item()
                    postfix["kl"] = kl
                t.set_postfix(postfix)

                if loss > 1000:
                    logger.warning(f"bad loss")
                    logger.info(f"{batch}")

                if step > 0 and step % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if step > 0 and step % args.eval_every == 0:
                    ckp = f"{epoch}.{int((epoch * steps_per_epoch) + step)}"
                    checkpoint_loss = checkpoint_loss / args.eval_every
                    logger.info(f"training loss (ckp {ckp}): {checkpoint_loss}")
                    checkpoint_loss = 0
                    lr = optimizer.param_groups[0]["lr"]
                    best["lr"] = lr
                    best, stop_early, report = self.checkpoint(
                        args, model, tokenizer, eval_datasets, ckp, best
                    )
                    if stop_early:
                        break
                    if args.dynamic_sampling:
                        seen_examples = (
                            global_step
                            * args.train_batch_size
                            * args.gradient_accumulation_steps
                        )
                        if seen_examples > args.dynamic_sampling_after:
                            sampler = train_dataloader.batch_sampler
                            sampler.set_dynamic_sampling_weights(report)

                if global_step > num_steps:
                    stop_early = True
                    break

            if stop_early:
                break

            logger.info(f"end of epoch {epoch}")
            epoch_loss /= len(train_dataloader)
            logger.info(f"average training loss: {epoch_loss}")
            ckp = f"{epoch}.{(epoch + 1) * len(train_dataloader)}"
            best, stop_early, report = self.checkpoint(
                args, model, tokenizer, eval_datasets, ckp, best
            )
            if stop_early:
                break
        return best
