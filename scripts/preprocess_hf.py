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

    # With validation split
    python scripts/prepare_hf.py \
        --dataset openai/openwebtext \
        --split train \
        --output data/train.bin \
        --include_val
        # writes data/train.bin and data/val.bin
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
    parser.add_argument(
        "--include_val",
        action="store_true",
        help="Randomly sample 1%% of data as validation set and write alongside train",
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


def write_dataset(dataset, output_path: str, tokenizer: tiktoken.Encoding,
                  text_column: str, streaming: bool, num_workers: int,
                  label: str) -> None:
    """Tokenize a dataset split and write it to a .bin file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_tokens = 0
    print(f"Tokenizing and writing {label} -> {output_path} ...")

    if streaming:
        with open(output_path, "wb") as f:
            for sample in dataset:
                tokens = tokenizer.encode_ordinary(sample[text_column])
                tokens.append(tokenizer.eot_token)
                arr = np.array(tokens, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(tokens)
                if total_tokens % 10_000_000 == 0:
                    print(f"  {total_tokens:,} tokens written...")
    else:
        tokenized = dataset.map(
            tokenize,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "text_column": text_column},
            num_proc=num_workers,
            desc=f"Tokenizing {label}",
            remove_columns=dataset.column_names,
        )
        with open(output_path, "wb") as f:
            for sample in tokenized:
                arr = np.array(sample["ids"], dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(sample["ids"])
                if total_tokens % 10_000_000 == 0:
                    print(f"  {total_tokens:,} tokens written...")

    file_size = os.path.getsize(output_path) / 1e6
    print(f"Saved {total_tokens:,} tokens to {output_path} ({file_size:.1f} MB)")


def prepare(args: argparse.Namespace) -> None:
    tokenizer = tiktoken.get_encoding(args.encoding)
    print(f"Loaded tiktoken encoding: {args.encoding}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")

    print(f"Loading dataset: {args.dataset} (split={args.split}, streaming={args.streaming})")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=args.streaming,
    )

    if args.include_val:
        if args.streaming:
            raise ValueError("--include_val is not supported with --streaming. "
                             "Load the full dataset without streaming to perform the split.")
        splits = dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
        print(f"Split -> train: {len(splits['train']):,} docs, val: {len(splits['test']):,} docs")

        stem, ext = os.path.splitext(args.output)
        val_output = stem.replace("train", "val") if "train" in stem else f"{stem}_val"
        val_output = val_output + ext

        write_dataset(splits["train"], args.output, tokenizer,
                      args.text_column, False, args.num_workers, "train")
        write_dataset(splits["test"], val_output, tokenizer,
                      args.text_column, False, args.num_workers, "val")
    else:
        write_dataset(dataset, args.output, tokenizer,
                      args.text_column, args.streaming, args.num_workers, "train")


if __name__ == "__main__":
    args = parse_args()
    prepare(args)