# Split MRQA development datasets for few-shot transfer learning.
import argparse
import copy
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--dev_size", type=int, default=400)
    parser.add_argument(
        "--seed",
        help="The splits in the paper were created with seed=13",
        type=int,
        default=13,
    )
    return parser.parse_args()


def split_mrqa(args):
    fn = Path(args.path)
    with open(fn, "r") as f:
        lines = f.readlines()
    header = lines[0]
    examples = [json.loads(l) for l in lines[1:]]
    qids = [qa["qid"] for e in examples for qa in e["qas"]]
    random.seed(args.seed)
    dev_qids = set(random.sample(qids, k=args.dev_size))
    train_examples = []
    dev_examples = []

    for e in examples:
        dev_e = copy.deepcopy(e)
        train_e = copy.deepcopy(e)
        dev_e["qas"] = [q for q in e["qas"] if q["qid"] in dev_qids]
        train_e["qas"] = [q for q in e["qas"] if q["qid"] not in dev_qids]
        if dev_e["qas"]:
            dev_examples.append(dev_e)
        if train_e["qas"]:
            train_examples.append(train_e)

    print(f"loaded {len(examples)} passages/{len(qids)} questions from {fn}")
    train = [header.strip()] + [json.dumps(e) for e in train_examples]
    dev = [header.strip()] + [json.dumps(e) for e in dev_examples]

    train_dest = fn.parent.parent / "train" / fn.name
    dev_dest = fn.parent.parent / "dev" / fn.name
    print(f"writing {len(qids) - len(dev_qids)} questions to {train_dest}")
    with open(train_dest, "w") as f:
        f.write("\n".join(train))
    print(f"writing {len(dev_qids)} questions to {dev_dest}")
    with open(dev_dest, "w") as f:
        f.write("\n".join(dev))


if __name__ == "__main__":
    args = parse_args()
    split_mrqa(args)
