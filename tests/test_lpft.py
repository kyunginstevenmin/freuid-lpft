"""LP-FT switch: only LoRA A/B matrices toggle; head params never do (train.set_lora_trainable)."""
import torch
import torch.nn as nn
from train import set_lora_trainable


class _Lora(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.ones(2, 3)); self.B = nn.Parameter(torch.zeros(3, 2))


def test_set_lora_trainable_touches_only_lora_matrices():
    m = nn.Module(); m.backbone = nn.Module(); m.backbone.qkv = _Lora(); m.backbone.norm = nn.LayerNorm(3)
    m.head = nn.Linear(3, 1); m.backbone.norm.weight.requires_grad_(False)
    assert set_lora_trainable(m, False) == 12
    assert not m.backbone.qkv.A.requires_grad and not m.backbone.qkv.B.requires_grad
    assert m.head.weight.requires_grad and m.head.bias.requires_grad
    assert not m.backbone.norm.weight.requires_grad          # untouched: neither head nor LoRA
    set_lora_trainable(m, True)
    assert m.backbone.qkv.A.requires_grad and m.backbone.qkv.B.requires_grad
