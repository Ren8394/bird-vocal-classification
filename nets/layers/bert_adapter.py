from timm.models.vision_transformer import Block
import torch
import torch.nn as nn

class Bert_Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size),
            nn.GELU(),
            nn.Linear(adapter_size, hidden_size)
        )

        self._init_weights()

    def forward(self, x):
        return x + self.adapter(x)

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class Bert_ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, adapter_size=32):
        super(Bert_ViTBlock, self).__init__()
        self.block = Block(dim, num_heads, mlp_ratio, qkv_bias, qk_norm)
        self.adapter1 = Bert_Adapter(dim, adapter_size)
        self.adapter2 = Bert_Adapter(dim, adapter_size)


    def forward(self, x):
        x = x + self.block.drop_path1(self.block.attn(self.block.norm1(x)))
        x = self.adapter1(x)

        x = self.block.drop_path2(self.block.mlp(self.block.norm2(x)))
        x = self.adapter2(x)
        return x