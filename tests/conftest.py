import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mugpt.models.transformer import DecoderOnlyTransformer, ModelConfig
from mugpt.loss.losses import CrossEntropyLoss
from mugpt.logger.base import BaseLogger
from mugpt.training.trainer import TrainerConfig, VanillaTrainer


# ── constants ────────────────────────────────────────────────────────────────
VOCAB  = 256
EMB    = 64
HEADS  = 4
LAYERS = 2
SEQ    = 32


# ── stub logger ──────────────────────────────────────────────────────────────
class StubLogger(BaseLogger):
    def __init__(self):
        self.entries = []

    def log(self, metrics: dict, step: int) -> None:
        self.entries.append((step, metrics))

    def close(self) -> None:
        pass


# ── factory helpers ──────────────────────────────────────────────────────────
def make_model_config(**overrides) -> ModelConfig:
    cfg = dict(
        vocab_size=VOCAB,
        emb_dim=EMB,
        num_heads=HEADS,
        num_layers=LAYERS,
        seq_len=SEQ,
        dropout=0.0,
        bias=False,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


def make_model(**overrides) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(make_model_config(**overrides))


def make_trainer_config(**overrides) -> TrainerConfig:
    cfg = dict(
        lr=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_steps=2,
        max_steps=10,
        batch_size=2,
        device="cpu",
        log_every=1,
        eval_every=5,
        eval_batches=2,
        checkpoint_every=999,
        checkpoint_dir="checkpoints_test",
    )
    cfg.update(overrides)
    return TrainerConfig(**cfg)


def make_bin_file(num_tokens: int, seq_len: int, vocab_size: int = VOCAB) -> str:
    """Write a temporary .bin file with random uint16 token ids, return its path."""
    total = max(num_tokens, seq_len * 4 + 1)
    data = np.random.randint(0, vocab_size, size=total, dtype=np.uint16)
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    data.tofile(f)
    f.close()
    return f.name


def make_dataloader(path: str, seq_len: int, batch_size: int = 2, shuffle: bool = False) -> DataLoader:
    from mugpt.data.datasets import BinDataset
    ds = BinDataset(path, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── pytest fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def model_config():
    return make_model_config()


@pytest.fixture
def model():
    m = make_model()
    m.eval()
    return m


@pytest.fixture
def loss_fn():
    return CrossEntropyLoss()


@pytest.fixture
def stub_logger():
    return StubLogger()


@pytest.fixture
def bin_file():
    path = make_bin_file(SEQ * 10 + 1, SEQ)
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_checkpoint_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def trainer(bin_file, stub_logger):
    m   = make_model()
    dl  = make_dataloader(bin_file, SEQ, batch_size=2)
    cfg = make_trainer_config(max_steps=5, log_every=1, eval_every=3, eval_batches=2, checkpoint_every=999)
    return VanillaTrainer(m, dl, dl, CrossEntropyLoss(), stub_logger, cfg)