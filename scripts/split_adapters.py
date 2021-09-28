"""Write adapter parameters to separate files"""
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_fn")
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


def split_state_dict(args):
    print(f"loading state dict from {args.src_fn}")
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    state_dict = torch.load(args.src_fn, map_location=device)
    for adapter in args.adapter_names:
        sd = {}
        for k, v in state_dict.items():
            if f"heads.{adapter}" in k:
                rk = k.replace(f"heads.{adapter}.", "head.")
                sd[rk] = v
            elif adapter in k:
                sd[k] = v
        fn = Path(args.dest_dir) / adapter / "model.pt"
        print(f"writing {len(sd)} named parameters to {fn}")
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True)
        torch.save(sd, fn)


if __name__ == "__main__":
    args = parse_args()
    split_state_dict(args)
