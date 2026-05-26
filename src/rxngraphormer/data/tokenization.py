"""Sequence tokenization helpers for legacy graph-to-sequence datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
from tqdm import tqdm


def load_vocab(vocab_file: str | PathLike[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(vocab_file) as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            token = token.split()[0]
            vocab[token] = i
    return vocab


def get_token_ids(tokens: Sequence[str], vocab: Mapping[str, int], max_len: int) -> tuple[list[int], int]:

    token_ids: list[int] = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[: max_len - 1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens


def smi_tokenizer(smi: str) -> str:
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|Se?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)
    return " ".join(tokens)


def get_train_val_test_token_data(
    src: Sequence[str],
    tgt: Sequence[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    np.random.seed(random_seed)
    sf_idx = list(range(len(src)))
    np.random.shuffle(sf_idx)
    train_idx = sf_idx[: int(len(sf_idx) * (1 - val_ratio - test_ratio))]
    val_idx = sf_idx[
        int(len(sf_idx) * (1 - val_ratio - test_ratio)) : int(
            len(sf_idx) * (1 - test_ratio)
        )
    ]
    test_idx = sf_idx[int(len(sf_idx) * (1 - test_ratio)) :]
    src_token = [smi_tokenizer(smi) for smi in src]
    tgt_token = [smi_tokenizer(smi) for smi in tgt]

    train_src = [src_token[i] for i in train_idx]
    train_tgt = [tgt_token[i] for i in train_idx]
    val_src = [src_token[i] for i in val_idx]
    val_tgt = [tgt_token[i] for i in val_idx]
    test_src = [src_token[i] for i in test_idx]
    test_tgt = [tgt_token[i] for i in test_idx]

    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


def gen_vocab_map(all_token: Sequence[str], token_file: str | PathLike[str]) -> None:
    vocab_map: dict[str, int] = {}
    for line in tqdm(all_token):
        vocab_lst = line.split()
        for vocab in vocab_lst:
            if vocab not in vocab_map:
                vocab_map[vocab] = 1
            else:
                vocab_map[vocab] += 1
    vocab_inf_lst = ["_PAD", "_UNK", "_SOS", "_EOS"]
    for vocab in vocab_map:
        vocab_inf_lst.append(f"{vocab}    {vocab_map[vocab]}")
    with open(token_file, "w") as fw:
        fw.writelines("\n".join(vocab_inf_lst))


__all__ = [
    "gen_vocab_map",
    "get_token_ids",
    "get_train_val_test_token_data",
    "load_vocab",
    "smi_tokenizer",
]
