"""
Preprocessing script for HuggingFace datasets.
Usage:
    # Small slice for local testing
    python scripts/prepare_hf.py \
        --dataset roneneldan/TinyStories \
        --split train \
        --output data/train.bin

    # Full dataset on cluster
    python scripts/prepare_hf.py \
        --dataset togethercomputer/RedPajama-Data-V2 \
        --split train \
        --output data/train.bin \
        --streaming
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a HuggingFace dataset and write to a .bin file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g. roneneldan/TinyStories)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use. Supports HF slice notation e.g. train[:1%%]",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the column containing raw text (default: text)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output .bin file (e.g. data/train.bin)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        help="Tiktoken encoding to use (default: gpt2)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HuggingFace streaming mode for large datasets that don't fit in RAM",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel tokenization (default: 4)",
    )
    return parser.parse_args()


def tokenize(batch: dict, tokenizer: tiktoken.Encoding, text_column: str) -> dict:
    """
    Tokenize a batch of texts, appending EOT after each document.
    Returns a flat list of token ids per sample (not yet concatenated).
    """
    eot = tokenizer.eot_token
    ids = []
    for text in batch[text_column]:
        tokens = tokenizer.encode_ordinary(text)
        tokens.append(eot)
        ids.append(tokens)
    return {"ids": ids}


def prepare(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    tokenizer = tiktoken.get_encoding(args.encoding)
    print(f"Loaded tiktoken encoding: {args.encoding}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")

    print(f"Loading dataset: {args.dataset} (split={args.split}, streaming={args.streaming})")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming
    )

    print("Tokenizing...")
    if args.streaming:
        # In streaming mode we can't use .map() with multiprocessing,
        # so we iterate and tokenize on the fly
        all_ids = []
        for sample in dataset:
            tokens = tokenizer.encode_ordinary(sample[args.text_column])
            tokens.append(tokenizer.eot_token)
            all_ids.extend(tokens)
    else:
        tokenized = dataset.map(
            tokenize,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "text_column": args.text_column},
            num_proc=args.num_workers,
            desc="Tokenizing",
            remove_columns=dataset.column_names,
        )
        all_ids = []
        for sample in tokenized:
            all_ids.extend(sample["ids"])

    total_tokens = len(all_ids)
    print(f"Total tokens: {total_tokens:,}")

    # GPT-2 vocab fits in uint16 (max 65535, vocab size 50257)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(args.output)
    print(f"Saved {total_tokens:,} tokens to {args.output} ({arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = parse_args()
    prepare(args)