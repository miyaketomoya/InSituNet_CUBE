# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Self-attention block architecture

# Important!!!! The self-attention is not used in the paper because we cannot
# get good results out from it. Hence, the following code is not guaranteed to
# be correct. Users interested in this could try and see how the attention
# mechanisms make the differences, and if you identify any bugs in the following
# code, please let us know.

import torch
import torch.nn as nn
from torch.nn import functional as F

# class SelfAttention(nn.Module):
#   def __init__(self, in_channels):
#     super(SelfAttention, self).__init__()

#     self.conv_theta = nn.Conv2d(in_channels, in_channels // 8,
#                                 kernel_size=1, stride=1, padding=0)
#     self.conv_phi = nn.Conv2d(in_channels, in_channels // 8,
#                               kernel_size=1, stride=1, padding=0)
#     self.conv_g = nn.Conv2d(in_channels, in_channels // 2,
#                             kernel_size=1, stride=1, padding=0)
#     self.conv_o = nn.Conv2d(in_channels // 2, in_channels,
#                             kernel_size=1, stride=1, padding=0)

#     self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#     self.softmax = nn.Softmax(dim=-1)

#     self.gamma = nn.Parameter(torch.zeros(1))

#   def forward(self, x):
#     batch_size, c, h, w = x.size()

#     theta = self.conv_theta(x).view(batch_size, -1, h * w).permute(0, 2, 1)
#     phi = self.maxpool(self.conv_phi(x)).view(batch_size, -1, h * w // 4)

#     attn = torch.bmm(theta, phi)
#     attn = self.softmax(attn)

#     g = self.maxpool(self.conv_g(x)).view(batch_size, -1, h * w // 4)

#     o = torch.bmm(g, attn.permute(0, 2, 1))
#     o = o.view(batch_size, -1, h, w)
#     o = self.conv_o(o)

#     o = self.gamma * o + x

#     return o

class SelfAttention(nn.Module):
    r"""Self-Attention Module for calculating long-dependancy on feature map

	Args:
        in_channels (int): the dimension size of the input feature map.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.in_channels // 8,
                               kernel_size=1)
        self.key   = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.in_channels // 8,
                               kernel_size=1)
        self.value = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.in_channels,
                               kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        r"""
        Args:
          x (tensor): input feature map [B, C, H, W], N = (H*W)
        Returns:
          out (tensor): output feature map after self-attention apply 
        """
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H*W).permute(0, 2, 1) # [B, N, C]
        key   = self.key(  x).view(B, -1, H*W)                  # [B, C, N]
        energy = torch.bmm(query, key)                          # [B, N, N]
        attention = self.softmax(energy)                        # [B, N, N]

        value = self.value(x).view(B, -1, H*W)                  # [B, C, N]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x

        return out