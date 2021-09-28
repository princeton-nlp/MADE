import collections
import string

import torch


def device():
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def to_device(d, device_=None):
    if device_ is None:
        device_ = device()
    for k in d:
        if type(d[k]) == dict:
            d[k] = to_device(d[k], device_)
        elif type(d[k]) == torch.Tensor:
            d[k] = d[k].to(device_)
    return d


def max_ll(log_probs, mask):
    return torch.max(log_probs.masked_fill(~mask, -1e9), dim=-1)[0]


def marginal_ll(log_probs, mask):
    return torch.logsumexp(log_probs.masked_fill(~mask, -1e9), dim=-1)


def best_spans(start_log_probs, end_log_probs):
    """Copied from https://git.io/JkTCr. Output: (batch_size, 2)"""
    if start_log_probs.dim() != 2 or end_log_probs.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = start_log_probs.size()
    device = start_log_probs.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = start_log_probs.unsqueeze(2) + end_log_probs.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower
    # triangle has entries where the span ends before it starts.
    span_log_mask = torch.triu(
        torch.ones((passage_length, passage_length), device=device)
    ).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best
    # span using argmax. We can recover the start and end indices from
    # this flattened list using simple modular arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


_G = "Ġ"


def special_token_ids(tokenizer):
    return set(
        {
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.sep_token,
            tokenizer.cls_token,
            tokenizer.pad_token,
            tokenizer.mask_token,
        }
    )


def expand_span(input_ids, start, end, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special = special_token_ids(tokenizer)
    l = start
    r = end

    left_boundary = lambda l: (
        tokens[l].startswith(_G)
        or tokens[l - 1] in special
        or tokens[l - 1][-1] in string.punctuation
        or tokens[l - 1][0] in string.punctuation
    )
    while l > 0 and not left_boundary(l):
        l -= 1

    right_boundary = lambda r: (
        tokens[r + 1].startswith(_G)
        or tokens[r + 1][0] in string.punctuation
        or tokens[r + 1] in special
    )
    while r + 1 < len(tokens) and not right_boundary(r):
        r += 1

    return l, r


def freeze(args, k):
    if "adapter" in k:
        return args.freeze_adapters
    if "head" in k:
        return args.freeze_heads
    return args.freeze_transformer


def average_adapter_params(args, state_dict, proportions=None):
    if proportions is None:
        proportions = {
            a: torch.tensor(1 / len(args.adapter_names))
            for a in args.adapter_names
        }
    param_lst = collections.defaultdict(list)
    for k, p in state_dict.items():
        for name in args.adapter_names:
            if f"adapters.{name}." in k:
                rk = k.replace(f".{name}.", f".adapter.")
                param_lst[rk].append(p * proportions[name])
            if f"heads.{name}." in k:
                rk = k.replace(f"heads.{name}.", "head.")
                param_lst[rk].append(p * proportions[name])
    avg_dict = {
        k: torch.sum(torch.stack(vs, dim=0), dim=0)
        for k, vs in param_lst.items()
    }
    return avg_dict
