from einops import rearrange, repeat

import numpy
import torch
import torch.nn as nn

def get_2d_sincos_encoding(grid_height, grid_width, d_model, cls_token=False):
    grid_h = torch.arange(grid_height, dtype=torch.float32) # (H, )
    grid_w = torch.arange(grid_width, dtype=torch.float32)  # (W, )
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")                   # (H, W)
    grid = torch.stack(grid)                                # (2, H, W)
    grid = rearrange(grid, "c h w -> c 1 h w")              # (2, 1, H, W)

    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D SinCos encoding"
    half_d_model = d_model // 2
    emb_h = get_1d_sincos_encoding(half_d_model, grid[0])   # (H*W, D/2)
    emb_w = get_1d_sincos_encoding(half_d_model, grid[1])   # (H*W, D/2)
    emb = torch.cat([emb_h, emb_w], dim=1)                  # (H*W, D)

    if cls_token:
        emb = torch.cat([torch.zeros(1, d_model), emb], dim=0)

    return emb

def get_1d_sincos_encoding(d_model, total_grid):
    assert d_model % 2 == 0, "d_model must be divisible by 2 for 1D SinCos encoding"
    omega = torch.arange(d_model // 2, dtype=torch.float32)  # (D/2, )
    omega = 1. / (10000 ** (2 * omega / d_model))            # (D/2, )

    total_grid = total_grid.view(-1)                         # (H*W, )
    output = torch.einsum("i,j->ij", total_grid, omega)      # (H*W, D/2)

    emb_sin = torch.sin(output)                              # (H*W, D/2)
    emb_cos = torch.cos(output)                              # (H*W, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)               # (H*W, D)

    return emb

class PositionEncoding_2Dto1D(nn.Module):
    def __init__(self, d_model, grid_size, cls_token=False, requires_grad=True):
        """
        Learnable positional encoding layer.

        Args:
            d_model (int): model dimension, embedding dimension, feature dimension (D)
            grid_size (tuple): size of the grid (H, W)
            cls_token (bool): whether to use cls token or not
            requires_grad (bool): whether the positional encoding is learnable or not
        Returns:
            torch.Tensor: positional encoding matrix (H*W, D)
        """
        super().__init__()
        self.d_model = d_model
        self.grid_height, self.grid_width = grid_size
        self.total_grid = self.grid_height * self.grid_width

        if cls_token:
            self.total_grid += 1
        self.pe = nn.Parameter(torch.zeros(1, self.total_grid, d_model, dtype=torch.float32), requires_grad=requires_grad)   # (1, max_len, d_model)
        self._init_weights_2d_sincos(cls_token=cls_token)

    def _init_weights_2d_sincos(self, cls_token=False):
        pe = get_2d_sincos_encoding(self.grid_height, self.grid_width, self.d_model, cls_token=cls_token)
        pe = rearrange(pe, "n d -> 1 n d")
        self.pe.data.copy_(pe)
