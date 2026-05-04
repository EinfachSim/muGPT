import os
import torch
from ..loss import BaseLoss
from ..logger.base import BaseLogger
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
import math

@dataclass
class TrainerConfig:
    # optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    # training
    max_steps: int = 1000
    batch_size: int = 32
    device: str = "cuda"
    # logging + eval
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 50
    checkpoint_every: int = 200
    checkpoint_dir: str = "checkpoints"
    gradient_accumulation_steps: int = 1
    def __post_init__(self):
        self.lr = float(self.lr)

def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

class VanillaTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            loss_fn: BaseLoss,
            logger: BaseLogger,
            config: TrainerConfig
        ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.logger = logger
        self.config = config
        self.optimizer = self._configure_optimizer()
        self.device = self.config.device
        self.model.to(self.device)
        self.model = torch.compile(self.model)
    
    def _get_lr(self, step: int) -> float:
        # warmup phase
        if step < self.config.warmup_steps:
            return self.config.lr * step / self.config.warmup_steps
        
        # cosine decay phase
        progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
        return self.config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, resume_from: str = None):
        self.model.train()
        start_step = 0
        if resume_from is not None:
            print(self.config.max_steps)
            start_step = self.load_checkpoint(resume_from)

        print("Starting training", flush = True)
        print(f"Train dataset size: {len(self.train_dataloader.dataset):,}", flush=True)

        data_iter = cycle_loader(self.train_dataloader)

        for step in range(start_step, self.config.max_steps):

            self.optimizer.zero_grad()

            if step == 0:
                print(f"First batch fetched, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
            if step >= self.config.max_steps:
                break

            # LR scheduling
            lr = self._get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            batch_loss = 0
            for micro_step in range(self.config.gradient_accumulation_steps):

                x,y = next(data_iter)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                logits = self.model(x)
                loss = self.loss_fn(logits, y) / self.config.gradient_accumulation_steps
                loss.backward()
                batch_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            #Logging
            if step % self.config.log_every == 0:
                self.logger.log({"train_loss": batch_loss, "learning rate": lr}, step=step)
                print(f"TRAIN LOSS: {batch_loss}")

            if step % self.config.eval_every == 0:
                val_loss = self.evaluate(self.config.eval_batches)
                self.logger.log({"val_loss": val_loss}, step=step)

            if step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)
        
        self.save_checkpoint(self.config.max_steps)
        
    def evaluate(self, num_batches: int) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_dataloader):
                if i >= num_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_loss += self.loss_fn(logits, y).item()
        self.model.train()
        return total_loss / num_batches
    
    def _configure_optimizer(self):
        decay_params = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        #Sanity check
        all_params = set(self.model.parameters())
        grouped_params = set(decay_params + no_decay_params)
        assert all_params == grouped_params, "Some parameters are missing from optimizer groups"

        return torch.optim.AdamW(param_groups, lr=self.config.lr)

    def save_checkpoint(self, step: int) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.model.cfg),  # dict, not object
        }
        path = f"{self.config.checkpoint_dir}/ckpt_{step}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from step {checkpoint['step']}")
        return checkpoint["step"]  # so train() knows where to resume from