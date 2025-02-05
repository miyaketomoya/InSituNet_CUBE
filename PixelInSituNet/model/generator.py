# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Generator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from PixelInSituNet.module.resblock import BasicBlockGenerator
from PixelInSituNet.module.self_attention import SelfAttention

class Generator(nn.Module):
  def __init__(self, dsp=1, dtp=1, dvp=3, dspe=512, dtpe=512, dvpe=512, ch=64,pixelshuffle = False):
    super(Generator, self).__init__()

    self.pixelshuffle = pixelshuffle
    self.dsp, self.dspe = dsp, dspe
    self.dtp, self.dtpe = dtp, dtpe
    self.dvp, self.dvpe = dvp, dvpe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # visualization operations subnet
    self.time_subnet = nn.Sequential(
      nn.Linear(dtp, dtpe), nn.ReLU(),
      nn.Linear(dtpe, dtpe), nn.ReLU()
    )

    # view parameters subnet
    self.vparams_subnet = nn.Sequential(
      nn.Linear(dvp, dvpe), nn.ReLU(),
      nn.Linear(dvpe, dvpe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe + dvpe + dtpe, ch * 16 * 4 * 4, bias=False),
      nn.ReLU()
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1,pixelshuffle=self.pixelshuffle),

      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1, upsample=False, pixelshuffle=False),# new

      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),

          # Another resblock at the new resolution
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1, upsample=False, pixelshuffle=False),

      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),
      #512用
      # SelfAttention(ch*4),

          # Additional resblock without upsampling
      BasicBlockGenerator(ch * 4, ch * 4, kernel_size=3, stride=1, padding=1, upsample=False, pixelshuffle=False),

      BasicBlockGenerator(ch * 4, ch * 4, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),
      
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),
      # SelfAttention(ch * 2),
          # One more resblock without upsampling
      BasicBlockGenerator(ch * 2, ch * 2, kernel_size=3, stride=1, padding=1, upsample=False, pixelshuffle=False),

      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1, pixelshuffle=self.pixelshuffle),
      nn.BatchNorm2d(ch),
      #nn.GroupNorm(ch//2,ch),
      nn.ReLU(),
      nn.Conv2d(ch, 4, kernel_size=3, stride=1, padding=1),
      nn.Tanh()
    )

  def forward(self, sp, vp, tp):
    sp = self.sparams_subnet(sp)
    tp = self.time_subnet(tp)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp,vp,tp), 1)
    mp = self.mparams_subnet(mp)

    x = mp.view(mp.size(0), self.ch * 16, 4, 4)
    x = self.img_subnet(x)

    return x
