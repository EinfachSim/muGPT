import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mugpt.data.datasets import BinDataset

from conftest import SEQ, VOCAB, make_bin_file, make_dataloader


NUM_TOKENS = SEQ * 10 + 1  # 10 full non-overlapping chunks


@pytest.fixture
def bin_path():
    path = make_bin_file(NUM_TOKENS, SEQ)
    yield path
    os.unlink(path)


@pytest.fixture
def dataset(bin_path):
    return BinDataset(bin_path, seq_len=SEQ)


class TestBinDataset:
    def test_len(self, bin_path):
        ds = BinDataset(bin_path, seq_len=SEQ)
        raw_len = np.fromfile(bin_path, dtype=np.uint16).shape[0]
        assert len(ds) == (raw_len - SEQ) // SEQ

    def test_item_shapes(self, dataset):
        x, y = dataset[0]
        assert x.shape == (SEQ,)
        assert y.shape == (SEQ,)

    def test_x_y_offset_by_one(self, dataset):
        """y should be x shifted forward by one token."""
        x, y = dataset[0]
        assert torch.equal(x[1:], y[:-1])

    def test_non_overlapping_chunks(self, bin_path):
        """Consecutive indices must produce non-overlapping windows."""
        ds = BinDataset(bin_path, seq_len=SEQ)
        x0, _ = ds[0]
        x1, _ = ds[1]

        raw = torch.from_numpy(np.fromfile(bin_path, dtype=np.uint16).astype(np.int64))
        assert torch.equal(x0, raw[0:SEQ])
        assert torch.equal(x1, raw[SEQ: SEQ * 2])

    def test_dtype_is_int64(self, dataset):
        x, y = dataset[0]
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_all_indices_in_bounds(self, bin_path):
        ds = BinDataset(bin_path, seq_len=SEQ)
        raw_len = np.fromfile(bin_path, dtype=np.uint16).shape[0]
        for i in range(len(ds)):
            start = i * SEQ
            assert start + SEQ + 1 <= raw_len + 1

    def test_dataloader_batch_shape(self, bin_path):
        dl = make_dataloader(bin_path, SEQ, batch_size=2)
        x, y = next(iter(dl))
        assert x.shape == (2, SEQ)
        assert y.shape == (2, SEQ)