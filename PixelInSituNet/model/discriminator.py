# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from PixelInSituNet.module.resblock import FirstBlockDiscriminator, BasicBlockDiscriminator
from PixelInSituNet.module.self_attention import SelfAttention

class Discriminator(nn.Module):
  def __init__(self, dsp=1, dtp=1, dvp=3, dspe=512, dtpe=512, dvpe=512, ch=64):
    super(Discriminator, self).__init__()

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
      nn.Linear(dspe + dvpe + dtpe, ch * 16),
      nn.ReLU()
    )

    # image classification subnet
    self.img_subnet = nn.Sequential(
      FirstBlockDiscriminator(4, ch, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3,
                              stride=1, padding=1),
      # SelfAttention(ch * 2),
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3,
                              stride=1, padding=1),
      SelfAttention(ch * 8),
      BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3,
                              stride=1, padding=1),
      
      #512用
      BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1),

      BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1, downsample=False),
      
      
      nn.ReLU()
    )

    # output subnets
    self.out_subnet = nn.Sequential(
      nn.Linear(ch * 16, 1)
    )

  def forward(self, sp, vp, tp,x):
    sp = self.sparams_subnet(sp)
    tp = self.time_subnet(tp)
    vp = self.vparams_subnet(vp)

    mp = torch.cat((sp, vp, tp), 1)
    mp = self.mparams_subnet(mp)

    x = self.img_subnet(x)
    x = torch.sum(x, (2, 3))

    out = self.out_subnet(x)
    out += torch.sum(mp * x, 1, keepdim=True)
    
    #
    #out = torch.sigmoid(out)

    return out
