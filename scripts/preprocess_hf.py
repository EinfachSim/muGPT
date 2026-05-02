"""
Preprocessing script for HuggingFace datasets.

Tokenizes a HuggingFace text dataset and writes a flat binary file of token
ids to disk. Each document is separated by the tokenizer's EOT token so the
model sees document boundaries during training.

The output .bin file is a flat uint16 array that can be memory-mapped or
loaded directly during training.

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

    # Limit samples for quick iteration
    python scripts/prepare_hf.py \
        --dataset roneneldan/TinyStories \
        --split train[:1%] \
        --output data/train_small.bin
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
        streaming=args.streaming,
        trust_remote_code=True,
    )

    print("Tokenizing and writing...")
    total_tokens = 0

    if args.streaming:
        # streaming: tokenize and write sample by sample
        with open(args.output, "wb") as f:
            for sample in dataset:
                tokens = tokenizer.encode_ordinary(sample[args.text_column])
                tokens.append(tokenizer.eot_token)
                arr = np.array(tokens, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(tokens)
                if total_tokens % 10_000_000 == 0:
                    print(f"  {total_tokens:,} tokens written...")
    else:
        # non-streaming: use parallel .map() for tokenization, write incrementally
        tokenized = dataset.map(
            tokenize,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "text_column": args.text_column},
            num_proc=args.num_workers,
            desc="Tokenizing",
            remove_columns=dataset.column_names,
        )
        with open(args.output, "wb") as f:
            for sample in tokenized:
                arr = np.array(sample["ids"], dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(sample["ids"])
                if total_tokens % 10_000_000 == 0:
                    print(f"  {total_tokens:,} tokens written...")

    file_size = os.path.getsize(args.output) / 1e6
    print(f"Saved {total_tokens:,} tokens to {args.output} ({file_size:.1f} MB)")



if __name__ == "__main__":
    args = parse_args()
    prepare(args)