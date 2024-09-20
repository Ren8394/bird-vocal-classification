import math

from timm.models.vision_transformer import Block
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.randn(rank, out_features))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        return x @ self.A @ self.B

class LoRA_Linear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRA_Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.adapter = LoRA(in_features, out_features, rank)

    def forward(self, x):
        return self.linear(x) + self.adapter(x)

class LoRA_ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, lora_rank=4):
        super(LoRA_ViTBlock, self).__init__()
        self.block = Block(dim, num_heads, mlp_ratio, qkv_bias, qk_norm)
        self.adapter = LoRA(in_features=dim, out_features=dim, rank=lora_rank)

    def forward(self, x):
        x = self.block.norm1(x)
        x = self.block.attn(x)
        x = x + self.adapter(x)
        x = self.block.drop_path1(x) + x

        x = self.block.norm2(x)
        x = self.block.mlp(x)
        x = self.block.drop_path2(x) + x
        return x
