from einops import rearrange, repeat

import torch
import torch.nn as nn

from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block

from nets.layers.pos_encode import PositionEncoding_2Dto1D

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=None):
        super().__init__()
        self.block = Block(dim, num_heads, mlp_ratio, qkv_bias, qk_norm)

    def forward(self, x):
        return self.block(x)

class ViT(nn.Module):
    def __init__(
        self, in_shape, patch_size, num_classes=1000,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
    ):
        super().__init__()
        self.in_channel, self.in_height, self.in_width = in_shape
        self.patch_height, self.patch_width = patch_size
        self.encoder_embed_dim, self.encoder_depth, self.encoder_num_heads = encoder_embed_dim, encoder_depth, encoder_num_heads

        self.grid_row, self.grid_col = self.in_height // self.patch_height, self.in_width // self.patch_width

        self.patch_embed = PatchEmbed(
            img_size=(self.in_height, self.in_width),
            patch_size=(self.patch_height, self.patch_width),
            in_chans=self.in_channel,
            embed_dim=encoder_embed_dim,
        )

        # encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_pos_embed = PositionEncoding_2Dto1D(encoder_embed_dim, (self.grid_row, self.grid_col), cls_token=True, requires_grad=False)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=encoder_embed_dim, num_heads=encoder_num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=None)
            for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # classifier head
        self.head = nn.Linear(encoder_embed_dim, num_classes)

        # initialize weights
        self.apply(self._init_weights)

    def forward_encoder(self, x: torch.Tensor):
        B = x.shape[0]
        # patch embedding
        x = self.patch_embed(x)
        
        # add cls token
        cls_token = repeat(self.cls_token, "() n d -> b n d", b=B)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder_pos_embed.pe

        # encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        latent = x[:, 0]

        return latent

    def forward(self, x):
        latent = self.forward_encoder(x)
        y = self.head(latent)

        return y

    def _init_weights(self, layer):
        # initialize patch embedding as nn.Linear
        patch_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_weight.view(patch_weight.size(0), -1))

        # initialize token
        nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

