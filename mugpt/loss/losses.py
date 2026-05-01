from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class BaseLoss(ABC):
    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        ...


class CrossEntropyLoss(BaseLoss):
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        batch_size, seq_len, vocab_size = logits.shape
        
        logits = logits.view(batch_size * seq_len, vocab_size)
        targets = targets.view(batch_size * seq_len)
        
        return F.cross_entropy(logits, targets)