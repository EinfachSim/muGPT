"""
Main training entrypoint.

Usage:
    # Quick local test (small model, few steps)
    python train.py --config configs/test.yaml

    # Full 100M run on cluster
    python train.py --config configs/train_100m.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from mugpt.models.transformer import DecoderOnlyTransformer, ModelConfig
from mugpt.data.datasets import BinDataset
from mugpt.loss import CrossEntropyLoss
from mugpt.logger import WandBLogger
from mugpt.training import VanillaTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # --- Model ---
    model_config = ModelConfig(**cfg["model"])
    model = DecoderOnlyTransformer(model_config)
    print(f"Model parameters: {model.num_parameters():,}")

    # --- Data ---
    train_dataset = BinDataset(cfg["data"]["train"], seq_len=model_config.seq_len)
    val_dataset   = BinDataset(cfg["data"]["val"],   seq_len=model_config.seq_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Loss ---
    loss_fn = CrossEntropyLoss()

    # --- Logger ---
    logger = WandBLogger(
        project=cfg["wandb"]["project"],
        run_name=cfg["wandb"].get("run_name", None),
        config={**cfg["model"], **cfg["trainer"]},
    )

    # --- Trainer ---
    trainer_config = TrainerConfig(**cfg["trainer"])
    trainer = VanillaTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        logger=logger,
        config=trainer_config,
    )

    trainer.train()
    logger.close()


if __name__ == "__main__":
    main()