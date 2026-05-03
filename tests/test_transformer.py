import math

import pytest
import torch

from mugpt.models.transformer import (
    CausalAttentionBlock,
    DecoderOnlyTransformer,
    FeedForwardBlock,
    ModelConfig,
    TransformerBlock,
)
from mugpt.loss.losses import CrossEntropyLoss

from conftest import VOCAB, EMB, HEADS, LAYERS, SEQ, make_model, make_model_config


# ════════════════════════════════════════════════════════════════════════════
# ModelConfig
# ════════════════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_valid_config(self, model_config):
        assert model_config.emb_dim == EMB
        assert model_config.num_heads == HEADS

    def test_invalid_head_division_raises(self):
        with pytest.raises(AssertionError):
            ModelConfig(vocab_size=VOCAB, emb_dim=65, num_heads=4)  # 65 % 4 != 0

    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.vocab_size == 50257
        assert cfg.seq_len == 1024


# ════════════════════════════════════════════════════════════════════════════
# CausalAttentionBlock
# ════════════════════════════════════════════════════════════════════════════

class TestCausalAttentionBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.block = CausalAttentionBlock(EMB, HEADS, SEQ, bias=False, dropout=0.0)
        self.block.eval()

    def test_output_shape(self):
        x = torch.randn(2, SEQ, EMB)
        assert self.block(x).shape == (2, SEQ, EMB)

    def test_shorter_sequence(self):
        x = torch.randn(2, SEQ // 2, EMB)
        assert self.block(x).shape == (2, SEQ // 2, EMB)

    def test_causal_mask(self):
        """Future tokens must not influence past positions."""
        x = torch.randn(1, SEQ, EMB)
        x_masked = x.clone()
        x_masked[0, 1:] = 0.0

        with torch.no_grad():
            out_full   = self.block(x)
            out_masked = self.block(x_masked)

        assert torch.allclose(out_full[0, 0], out_masked[0, 0], atol=1e-5)

    def test_mask_buffer_registered(self):
        assert hasattr(self.block, "mask")
        assert self.block.mask.shape == (1, 1, SEQ, SEQ)

    def test_mask_is_lower_triangular(self):
        mask = self.block.mask[0, 0]
        assert torch.equal(mask, torch.tril(torch.ones(SEQ, SEQ)))


# ════════════════════════════════════════════════════════════════════════════
# FeedForwardBlock
# ════════════════════════════════════════════════════════════════════════════

class TestFeedForwardBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.block = FeedForwardBlock(EMB, dropout=0.0, bias=False)
        self.block.eval()

    def test_output_shape(self):
        x = torch.randn(2, SEQ, EMB)
        assert self.block(x).shape == (2, SEQ, EMB)

    def test_hidden_expansion(self):
        assert self.block.l1.out_features == 4 * EMB
        assert self.block.l2.in_features == 4 * EMB

    def test_no_bias(self):
        assert self.block.l1.bias is None
        assert self.block.l2.bias is None


# ════════════════════════════════════════════════════════════════════════════
# TransformerBlock
# ════════════════════════════════════════════════════════════════════════════

class TestTransformerBlock:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.block = TransformerBlock(EMB, HEADS, SEQ, bias=False, dropout=0.0)
        self.block.eval()

    def test_output_shape(self):
        x = torch.randn(2, SEQ, EMB)
        assert self.block(x).shape == (2, SEQ, EMB)

    def test_residual_connection(self):
        x = torch.randn(2, SEQ, EMB)
        with torch.no_grad():
            out = self.block(x)
        assert not torch.allclose(out, x)

    def test_has_two_layernorms(self):
        assert hasattr(self.block, "layernorm1")
        assert hasattr(self.block, "layernorm2")


# ════════════════════════════════════════════════════════════════════════════
# DecoderOnlyTransformer
# ════════════════════════════════════════════════════════════════════════════

class TestDecoderOnlyTransformer:
    def test_output_shape(self, model):
        x = torch.randint(0, VOCAB, (2, SEQ))
        assert model(x).shape == (2, SEQ, VOCAB)

    def test_output_shape_short_seq(self, model):
        x = torch.randint(0, VOCAB, (1, SEQ // 2))
        assert model(x).shape == (1, SEQ // 2, VOCAB)

    def test_weight_tying(self, model):
        assert model.head.weight is model.token_emb.weight

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0

    def test_num_parameters_excludes_non_trainable(self, model):
        total = model.num_parameters()
        model.token_emb.weight.requires_grad_(False)
        reduced = model.num_parameters()
        model.token_emb.weight.requires_grad_(True)
        assert reduced < total

    def test_num_blocks(self, model):
        assert len(model.blocks) == LAYERS

    def test_initial_loss_near_log_vocab(self, model, loss_fn):
        x = torch.randint(0, VOCAB, (4, SEQ))
        y = torch.randint(0, VOCAB, (4, SEQ))
        with torch.no_grad():
            loss = loss_fn(model(x), y)
        assert abs(loss.item() - math.log(VOCAB)) < 0.5

    def test_residual_projection_scaling(self):
        model = make_model()
        scale = 0.02 / math.sqrt(2 * LAYERS)
        for name, param in model.named_parameters():
            if name.endswith("attn_proj.weight") or name.endswith("l2.weight"):
                assert param.std().item() < scale * 5

    def test_no_bias_when_configured(self):
        import torch.nn as nn
        model = make_model(bias=False)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert module.bias is None, f"Linear layer {name} has bias when bias=False"

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_forward_various_batch_sizes(self, model, batch_size):
        x = torch.randint(0, VOCAB, (batch_size, SEQ))
        assert model(x).shape[0] == batch_size