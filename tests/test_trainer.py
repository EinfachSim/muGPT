import os
import tempfile

import pytest
import torch

from mugpt.loss.losses import CrossEntropyLoss
from mugpt.training.trainer import TrainerConfig, VanillaTrainer

from conftest import (
    SEQ,
    StubLogger,
    make_bin_file,
    make_dataloader,
    make_model,
    make_trainer_config,
)


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def bin_file():
    path = make_bin_file(SEQ * 50, SEQ)
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_checkpoint_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def logger():
    return StubLogger()


@pytest.fixture
def trainer(bin_file, logger):
    model = make_model()
    dl    = make_dataloader(bin_file, SEQ, batch_size=2)
    cfg   = make_trainer_config(max_steps=5, log_every=1, eval_every=3, eval_batches=2, checkpoint_every=999)
    return VanillaTrainer(model, dl, dl, CrossEntropyLoss(), logger, cfg)


@pytest.fixture
def checkpoint_trainer(tmp_checkpoint_dir, logger):
    path  = make_bin_file(SEQ * 20, SEQ)
    model = make_model()
    dl    = make_dataloader(path, SEQ)
    cfg   = make_trainer_config(checkpoint_dir=tmp_checkpoint_dir, checkpoint_every=1, max_steps=3)
    t = VanillaTrainer(model, dl, dl, CrossEntropyLoss(), logger, cfg)
    yield t, path
    os.unlink(path)


# ════════════════════════════════════════════════════════════════════════════
# TrainerConfig
# ════════════════════════════════════════════════════════════════════════════

class TestTrainerConfig:
    def test_lr_cast_to_float(self):
        assert isinstance(TrainerConfig(lr=3e-4).lr, float)

    def test_lr_from_string(self):
        cfg = TrainerConfig(lr="3e-4")
        assert isinstance(cfg.lr, float)
        assert abs(cfg.lr - 3e-4) < 1e-10

    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.weight_decay == 0.1
        assert cfg.grad_clip == 1.0


# ════════════════════════════════════════════════════════════════════════════
# LR schedule
# ════════════════════════════════════════════════════════════════════════════

class TestLRSchedule:
    @pytest.fixture(autouse=True)
    def setup(self, bin_file, logger):
        model = make_model()
        dl    = make_dataloader(bin_file, SEQ)
        cfg   = make_trainer_config(warmup_steps=10, max_steps=100, lr=1e-3)
        self.trainer = VanillaTrainer(model, dl, dl, CrossEntropyLoss(), logger, cfg)

    def test_lr_zero_at_step_zero(self):
        assert self.trainer._get_lr(0) == 0.0

    def test_lr_peaks_at_warmup_end(self):
        assert abs(self.trainer._get_lr(10) - 1e-3) < 1e-9

    def test_lr_decays_after_warmup(self):
        assert self.trainer._get_lr(10) > self.trainer._get_lr(55) > self.trainer._get_lr(100)

    def test_lr_zero_at_max_steps(self):
        assert abs(self.trainer._get_lr(100)) < 1e-9

    def test_lr_non_negative_throughout(self):
        assert all(self.trainer._get_lr(s) >= 0.0 for s in range(101))


# ════════════════════════════════════════════════════════════════════════════
# Optimizer groups
# ════════════════════════════════════════════════════════════════════════════

class TestOptimizerGroups:
    def test_two_param_groups(self, trainer):
        assert len(trainer.optimizer.param_groups) == 2

    def test_decay_group_weight_decay(self, trainer):
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1

    def test_no_decay_group_zero_weight_decay(self, trainer):
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_all_params_covered(self, trainer):
        all_params = set(trainer.model.parameters())
        grouped = {p for g in trainer.optimizer.param_groups for p in g["params"]}
        assert all_params == grouped

    def test_1d_params_in_no_decay_group(self, trainer):
        no_decay = trainer.optimizer.param_groups[1]["params"]
        assert all(p.dim() < 2 for p in no_decay)


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

class TestTrainingLoop:
    def test_logger_receives_train_loss(self, trainer, logger):
        trainer.train()
        assert any("train_loss" in m for _, m in logger.entries)

    def test_logger_receives_val_loss(self, trainer, logger):
        trainer.train()
        assert any("val_loss" in m for _, m in logger.entries)

    def test_model_in_train_mode_after_training(self, trainer):
        trainer.train()
        assert trainer.model.training

    def test_stops_at_max_steps(self, trainer, logger):
        trainer.train()
        logged_steps = [s for s, m in logger.entries if "train_loss" in m]
        assert all(s < trainer.config.max_steps for s in logged_steps)


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint save / load
# ════════════════════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_checkpoint_dir_created(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        subdir = os.path.join(tmp_checkpoint_dir, "subdir")
        trainer.config.checkpoint_dir = subdir
        trainer.save_checkpoint(0)
        assert os.path.isdir(subdir)

    def test_checkpoint_file_written(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(42)
        assert os.path.isfile(os.path.join(tmp_checkpoint_dir, "ckpt_42.pt"))

    def test_checkpoint_contains_expected_keys(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(7)
        ckpt = torch.load(os.path.join(tmp_checkpoint_dir, "ckpt_7.pt"), map_location="cpu")
        for key in ("step", "model_state_dict", "optimizer_state_dict", "config"):
            assert key in ckpt

    def test_checkpoint_step_value(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(99)
        ckpt = torch.load(os.path.join(tmp_checkpoint_dir, "ckpt_99.pt"), map_location="cpu")
        assert ckpt["step"] == 99

    def test_load_checkpoint_restores_step(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(42)
        step = trainer.load_checkpoint(os.path.join(tmp_checkpoint_dir, "ckpt_42.pt"))
        assert step == 42

    def test_load_checkpoint_restores_weights(self, checkpoint_trainer, tmp_checkpoint_dir):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(0)
        original = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        with torch.no_grad():
            for p in trainer.model.parameters():
                p.add_(torch.randn_like(p))

        trainer.load_checkpoint(os.path.join(tmp_checkpoint_dir, "ckpt_0.pt"))
        for k, v in trainer.model.state_dict().items():
            assert torch.allclose(v, original[k]), f"Weight mismatch: {k}"

    def test_resume_skips_past_steps(self, checkpoint_trainer, tmp_checkpoint_dir, logger):
        trainer, _ = checkpoint_trainer
        trainer.save_checkpoint(2)
        logger.entries.clear()
        trainer.train(resume_from=os.path.join(tmp_checkpoint_dir, "ckpt_2.pt"))
        logged_steps = [s for s, m in logger.entries if "train_loss" in m]
        assert all(s >= 2 for s in logged_steps)