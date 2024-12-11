import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from mpas import ImageDatasetPredict, Normalize, ToTensor
from torchvision import transforms

import sys
sys.path.append("../")
from mpas import *

from PixelInSituNet.model.generator import Generator


def load_data(root, dvp, dsp, dtp, batch_size=1):
    # データセットの準備
    dataset = ImageDatasetPredict(
        root=root,
        dvp=dvp,
        dsp=dsp,
        dtp=dtp,
        transform=transforms.Compose([Normalize(), ToTensor()])
    )
    
    # データローダーの準備
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def load_model(checkpoint_path, device, args):
    # モデルの初期化
    g_model = Generator(dsp=args.dsp, dtp=args.dtp, dvp=args.dvp,
                        dspe=args.dspe, dtpe=args.dtpe, dvpe=args.dvpe,
                        ch=args.ch, pixelshuffle=args.pixelshuffle)
    
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    g_model.load_state_dict(checkpoint)
    g_model.to(device)
    g_model.eval()
    
    return g_model

def inference(g_model, sparams, vparams, time_step, device):
    # データをデバイスに移動
    sparams = sparams.to(device)
    vparams = vparams.to(device)
    time_step = time_step.to(device)
    
    # 推論
    with torch.no_grad():
        fake_image = g_model(sparams, vparams, time_step)
    
    return fake_image

def save_inference_results(fake_image, output_dir, sparams, vparams, time_step ,view_nun):
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(fake_image), 8)
    fake_rgb = fake_image[:, :3, :, :]  # RGB部分
    fake_depth = fake_image[:, 3:, :, :]  # Depth部分

    # パラメータを使ってファイル名を作成
    filename_prefix = f"sparams_{round(sparams.item()*0.04+0.05, 2)}_time_{round(time_step.item()*50)}_view_{round(view_nun.item())}"
    print(filename_prefix)
    save_image(((fake_rgb.cpu() + 1.) * .5),
               os.path.join(output_dir, f"{filename_prefix}_rgb.png"), nrow=n)
    save_image(((fake_depth.cpu() + 1.) * .5),
               os.path.join(output_dir, f"{filename_prefix}_depth.png"), nrow=n)


# 使用例
if __name__ == "__main__":
    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 引数の設定（例）
    class Args:
        dsp = 1
        dtp = 1
        dvp = 3
        dspe = 512
        dtpe = 512
        dvpe = 512
        ch = 64
        pixelshuffle = True

    args = Args()
    
    # データのロード
    root = "/home/tomoyam/ImageBasedSurrogate_CUBE/result/mu_0.05_v182_GT"
    dvp, dsp, dtp = 3, 1, 1  # 例としての次元数
    data_loader = load_data(root, dvp, dsp, dtp)
    
    # モデルのロード
    checkpoint_path = "/data/data2/tomoya/InSituNetTime/model/CUBE/v1202_pixel_addres/result/model_True_relu1_2_vanilla_80.pth"
    g_model = load_model(checkpoint_path, device, args)


    output_dir = "/home/tomoyam/ImageBasedSurrogate_CUBE/result/mu_0.05_v182_result"
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        vparams = sample["vparams"]
        sparams = sample["sparams"]
        time_step = sample["time"]
        view_num = sample["view_num"]
        fake_image = inference(g_model, sparams, vparams, time_step, device)
        save_inference_results(fake_image, output_dir, sparams, vparams, time_step, view_num)
