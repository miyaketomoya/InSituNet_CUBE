import os
import pandas as pd
import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import json

class ImageDataset(Dataset):
  def __init__(self, root,dvp,dsp,dtp,data_len = 0,train = True, transform = None):
    self.root = root
    self.train = train
    self.transform = transform
    self.data_len = data_len
    self.params = self._get_params(root) 
    self.dvp = dvp
    self.dsp = dsp
    self.dtp = dtp
    self.view_list = self._get_view_list(root)
    #新しく,vparams,sparamsなどの次元数をしていするようにする

  def _get_params(self,root):
    if self.train:
      with open(os.path.join(root,"train","filtered_timestep.json"), "r") as f:
        params = json.load(f)
    else:
      with open(os.path.join(root,"test","filtered_timestep.json"), "r") as f:
        params = json.load(f)
    # 新しいリスト形式に変換
    params_list = [
      {
        "name": key,
        "depth": value["image_name_depth"],
        "view": value["view_num"],
        "time": value["time_step"],
        "sim": value["sim_param"],
    }
    for key, value in params.items()
    ]
    return params_list

  def _get_view_list(self,root):
    if self.train:
      with open(os.path.join(root,"train","viewpoint1202.json"), "r") as f:
        view_list = json.load(f)
    else:
      with open(os.path.join(root,"test","viewpoint1202.json"), "r") as f:
        view_list = json.load(f)
    return view_list

  # def _get_file_paths(self, root):
  #   if self.train:
  #     file_paths = (pd.read_csv(os.path.join(root,"train","filenames.txt"),sep = " ",header = None))
  #   else:
  #     file_paths = (pd.read_csv(os.path.join(root,"test","filenames.txt"),sep = " ",header = None))
  #   return file_paths

  # def _get_img_params(self,root):
  #   if self.train:
  #     img_params = (pd.read_csv(os.path.join(root,"train","params.csv"),sep = ",",header = None , skiprows = 1).values)
  #   else:
  #     img_params = (pd.read_csv(os.path.join(root,"test","params.csv"),sep = ",",header = None , skiprows = 1).values)
  #   return img_params

  def __getitem__(self,index):
    if self.train:
      img_name = os.path.join(self.root,"train","img",self.params[index]["name"])
      depth_name = os.path.join(self.root,"train","img",self.params[index]["depth"])
    else:
      img_name = os.path.join(self.root,"train","img",self.params[index]["name"])
      depth_name = os.path.join(self.root,"train","img",self.params[index]["depth"])
      
    image = io.imread(img_name)
    depth = io.imread(depth_name)

    view_num = self.params[index]["view"] #x,y,z
    vparams = self.view_list[view_num]
    sparams = [self.params[index]["sim"]] #V,T
    time = [self.params[index]["time"]]
    sample = {"image":image, "depth":depth, "vparams":vparams, "sparams":sparams, "time":time}
    if self.transform:
      sample = self.transform(sample)

    # 返すのは結合したimage
    return sample
  
  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.params)
  
class ImageDatasetPredict(Dataset):
  def __init__(self, root, dvp, dsp, dtp, data_len=0, transform=None):
    self.root = root
    self.transform = transform
    self.data_len = data_len
    self.params = self._get_params(root)
    self.dvp = dvp
    self.dsp = dsp
    self.dtp = dtp
    self.view_list = self._get_view_list(root)

  def _get_params(self, root):
    with open(os.path.join(root, "params.json"), "r") as f:
      params = json.load(f)
    params_list = [
      {
        "name": key,
        "depth": value["image_name_depth"],
        "view": value["view_num"],
        "time": value["time_step"],
        "sim": value["sim_param"],
      }
      for key, value in params.items()
    ]
    return params_list

  def _get_view_list(self, root):
    with open(os.path.join(root, "viewpoint182.json"), "r") as f:
      view_list = json.load(f)
    return view_list

  def __getitem__(self, index):
    img_name = os.path.join(self.root, "img", "blank.bmp")
    depth_name = os.path.join(self.root, "img", "blank.bmp")

    # image = io.imread(img_name)
    # depth = io.imread(depth_name)

    #dummy
    image = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    depth = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

    view_num = self.params[index]["view"]
    vparams = self.view_list[view_num]
    sparams = [self.params[index]["sim"]]
    time = [self.params[index]["time"]]

    sample = {"image": image, "depth": depth, "vparams": vparams, "sparams": sparams, "time": time}
    if self.transform:
      sample = self.transform(sample)

    sample["view_num"] = view_num
    return sample

  def __len__(self):
    if self.data_len:
      return self.data_len
    else:
      return len(self.params)



class Resize(object):
  # Resize(256) ← 数字1つ　or Resize((256,128)) ← taple ()のどちらかで宣言
  def __init__(self,size):
    assert isinstance(size,(int,tuple))
    self.size = size

  def __call__(self,sample):
    image = sample["image"]
    depth = sample["depth"]
    h,w = image.shape[:2]
    if isinstance(self.size, int):
      if h > w:
        new_h, new_w = self.size * h / w, self.size
      else:
        new_h, new_w = self.size, self.size * w / h
    else:
      new_h, new_w = self.size

    new_h, new_w = int(new_h), int(new_w)

    #半分ぐらい調べても出てこないパラメータ mode?
    image = transform.resize(
    image, (new_h, new_w), order=1, mode="reflect",
    preserve_range=True, anti_aliasing=True).astype(np.float32)
    depth = transform.resize(
    depth, (new_h, new_w), order=1, mode="reflect",
    preserve_range=True, anti_aliasing=True).astype(np.float32)
    return {"image": image,
            "depth":depth,
            "vparams":sample["vparams"],
            "sparams":sample["sparams"],
            "time":sample["time"]
            }
    
class Normalize(object):
  def __call__(self, sample):
    image = sample["image"]
    depth = sample["depth"]
    sparams = sample["sparams"]
    vparams = sample["vparams"]
    time = sample["time"]


    image = (image.astype(np.float32) - 127.5) / 127.5
    depth = (depth.astype(np.float32) - 127.5) / 127.5

    # sparams min [0.0]
    #         max [0.1]
    sparams = (sparams - np.array([0.05], dtype=np.float32)) / np.array([0.05], dtype=np.float32)

    # vparamsは正規化したものを元々パラメータにする　 0-1
    time =  [time[0]/500]

    return {"image": image,
            "depth":depth,
            "vparams": vparams,
            "sparams": sparams,
            "time":time}
    
class ToTensor(object):
  def __call__(self, sample):
    image = sample["image"].astype(np.float32)
    depth = sample["depth"].astype(np.float32)
    vparams = np.array(sample["vparams"], dtype=np.float32)  # リストをnumpy配列に変換
    sparams = np.array(sample["sparams"], dtype=np.float32)  # リストをnumpy配列に変換
    time = np.array(sample["time"], dtype=np.float32)        # リストをnumpy配列に変換

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    # depthの1チャンネルのみを使用
    depth = depth[:, :, 0]  # 例えば、最初のチャンネルを使用
    depth = depth[np.newaxis, :, :]  # チャンネル次元を追加

    # 結合後のチャンネル数が4になるように修正

    combined = np.concatenate((image, depth), axis=0)
    return {"image": torch.from_numpy(combined),
            "vparams": torch.from_numpy(vparams),
            "sparams": torch.from_numpy(sparams),
            "time": torch.from_numpy(time)}
    
    
