"""Rename adapters"""
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("dest_dir")
    parser.add_argument(
        "--adapter_names",
        type=str,
        nargs="+",
        default=[
            "SQuAD",
            "HotpotQA",
            "TriviaQA",
            "SearchQA",
            "NewsQA",
            "NaturalQuestions",
        ],
    )
    return parser.parse_args()


def rename_adapters(args):
    for adapter in args.adapter_names:
        fn = Path(args.src_dir) / adapter / "model.pt"
        print(f"loading state dict from {fn}")
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        state_dict = torch.load(fn, map_location=device)
        remapped = {}
        for k, p in state_dict.items():
            if ".adapter." in k:
                rk = k.replace(".adapter.", f".{adapter}.")
                remapped[rk] = p
            else:
                remapped[k] = p
        fn_out = Path(args.dest_dir) / adapter / "model.pt"
        print(
            f"writing {len(remapped)}/{len(state_dict)} parameters "
            f"to {fn_out}"
        )
        if not fn_out.parent.exists():
            fn_out.parent.mkdir(parents=True)
        torch.save(remapped, fn_out)


if __name__ == "__main__":
    args = parse_args()
    rename_adapters(args)
