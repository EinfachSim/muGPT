import math

import pytest
import torch

from mugpt.loss.losses import CrossEntropyLoss

from conftest import VOCAB, SEQ


@pytest.fixture
def loss_fn():
    return CrossEntropyLoss()


class TestCrossEntropyLoss:
    def test_output_is_scalar(self, loss_fn):
        logits  = torch.randn(2, SEQ, VOCAB)
        targets = torch.randint(0, VOCAB, (2, SEQ))
        assert loss_fn(logits, targets).shape == ()

    def test_perfect_prediction_low_loss(self, loss_fn):
        targets = torch.zeros(2, SEQ, dtype=torch.long)
        logits  = torch.full((2, SEQ, VOCAB), -1e9)
        logits[:, :, 0] = 1e9
        assert loss_fn(logits, targets).item() < 0.01

    def test_uniform_logits_near_log_vocab(self, loss_fn):
        logits  = torch.zeros(2, SEQ, VOCAB)
        targets = torch.randint(0, VOCAB, (2, SEQ))
        assert abs(loss_fn(logits, targets).item() - math.log(VOCAB)) < 0.01

    def test_loss_is_positive(self, loss_fn):
        logits  = torch.randn(2, SEQ, VOCAB)
        targets = torch.randint(0, VOCAB, (2, SEQ))
        assert loss_fn(logits, targets).item() > 0

    def test_gradients_flow(self, loss_fn):
        logits  = torch.randn(2, SEQ, VOCAB, requires_grad=True)
        targets = torch.randint(0, VOCAB, (2, SEQ))
        loss_fn(logits, targets).backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)