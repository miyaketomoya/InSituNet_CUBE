from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, Range1d, Label
from bokeh.plotting import figure
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import sys
import time

sys.path.append("../")
from PixelInSituNet.model.generator import Generator

def toXYZ(tht, phi, rp):
    x = rp * math.sin(math.radians(tht)) * math.sin(math.radians(phi))
    y = rp * math.cos(math.radians(tht))
    z = rp * math.sin(math.radians(tht)) * math.cos(math.radians(phi))
    return x, y, z

def parse_args():
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument("--ModelName", required=True, type=str,
                        help="modelPath")
    parser.add_argument("--dsp", type=int, default=3,
                        help="dimensions of the simulation parameters (default: 3)")
    parser.add_argument("--dtp", type=int, default=3,
                        help="dimensions of the time parameters (default: 3)")
    parser.add_argument("--dvp", type=int, default=3,
                        help="dimensions of the view parameters (default: 3)")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode (default: 512)")
    parser.add_argument("--dtpe", type=int, default=512,
                        help="dimensions of the time parameters' encode (default: 512)")
    parser.add_argument("--dvpe", type=int, default=512,
                        help="dimensions of the view parameters' encode (default: 512)")
    parser.add_argument("--ch", type=int, default=64,
                        help="channel multiplier (default: 64)")
    parser.add_argument("--sn", action="store_true", default=False,
                        help="enable spectral normalization")
    parser.add_argument("--resolution", type=int, default=512,
                        help="resolution of image")
    parser.add_argument("--pixelshuffle", action="store_true", default=False,
                        help="use pixelshuffle or not")
    return parser.parse_args()

def model_set(args, device):
    def add_sn(m):
        for name, c in m.named_children():
            m.add_module(name, add_sn(c))
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            return nn.utils.spectral_norm(m, eps=1e-4)
        else:
            return m
    g_model = Generator(dsp=args.dsp, dtp=args.dtp, dvp=args.dvp,
                        dspe=args.dspe, dtpe=args.dtpe, dvpe=args.dvpe,
                        ch=args.ch, pixelshuffle=args.pixelshuffle)
    if args.sn:
        g_model = add_sn(g_model)
    order_dict = torch.load(args.ModelName, map_location=torch.device(device))
    g_model.load_state_dict(order_dict)
    g_model.to(device)
    g_model.eval()
    return g_model

def generate_imageTuple(args, g_model, device, tht, phi, rp, v, t):
    x, y, z = toXYZ(tht, phi, rp)
    time_params = torch.tensor([[t / 500.]], dtype=torch.float32, device=device)
    sparams = torch.tensor([[(v - 0.05) / 0.05]], dtype=torch.float32, device=device)
    vparams = torch.tensor([[x, y, z]], dtype=torch.float32, device=device)

    start = time.time()
    fake_image = g_model(sparams, vparams, time_params)
    end = time.time()
    print("Inference time:", end - start)
    
    # RGBとDepth画像を分割
    rgb_image = fake_image[:, :3, :, :]
    depth_image = fake_image[:, 3:, :, :]

    # RGB画像をNumPy配列に変換
    rgb_image_np = rgb_image.detach().cpu().numpy()
    rgb_image_np = np.clip(rgb_image_np, -1, 1)
    rgb_image_np = (rgb_image_np * 127.5 + 127.5).astype(np.uint8)
    rgb_image_np = np.transpose(rgb_image_np[0], (1, 2, 0))

    # Depth画像をNumPy配列に変換
    depth_image_np = depth_image.detach().cpu().numpy()
    depth_image_np = np.clip(depth_image_np, -1, 1)
    depth_image_np = (depth_image_np * 127.5 + 127.5).astype(np.uint8)
    depth_image_np = np.squeeze(depth_image_np[0], axis=0)

    # RGBA形式の画像を格納する配列を作成
    img_rgba = np.zeros((args.resolution, args.resolution, 4), dtype=np.uint8)
    img_rgba[..., :3] = rgb_image_np
    img_rgba[..., 3] = 255

    return img_rgba.view(dtype=np.uint32).reshape((args.resolution, args.resolution)), depth_image_np, x, y, z

args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
g_model = model_set(args, device)

v = 0.0
t = 0
# RGB画像表示用Figure
p_rgb = figure(x_range=Range1d(start=0, end=args.resolution), y_range=Range1d(start=0, end=args.resolution), tools="zoom_in,zoom_out,reset")
p_rgb.xaxis.visible = False
p_rgb.yaxis.visible = False

# Depth画像表示用Figure
p_depth = figure(x_range=Range1d(start=0, end=args.resolution), y_range=Range1d(start=0, end=args.resolution), tools="zoom_in,zoom_out,reset")
p_depth.xaxis.visible = False
p_depth.yaxis.visible = False

# 初期画像データを設定
initial_img, initial_depth_img, x, y, z = generate_imageTuple(args, g_model, device, 23, -63, 1., v, t)
r_rgb = p_rgb.image_rgba(image=[initial_img], x=0, y=0, dw=args.resolution, dh=args.resolution)
r_depth = p_depth.image(image=[initial_depth_img], x=0, y=0, dw=args.resolution, dh=args.resolution, palette="Greys256")

label = Label(x=10, y=500, text=f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v:.2f}, t: {t:.2f}", text_font_size="10pt", text_color="white")
p_rgb.add_layout(label)

# スライダーを作成
slider1 = Slider(start=0.0, end=0.1, value=0.0, step=0.001, title="Mu", width=570)
slider2 = Slider(start=0, end=500, value=0, step=1, title="Time step", width=570)
slider3 = Slider(start=-180., end=180., value=23., step=1., title="θ", width=570)
slider4 = Slider(start=-180., end=180., value=-63., step=1., title="φ", width=570)

# コールバック関数を定義
def update_data(attrname, old, new):
    v = slider1.value
    t = slider2.value
    tht = slider3.value
    phi = slider4.value
    img, depth_img, x, y, z = generate_imageTuple(args, g_model, device, tht, phi, 1., v, t)

    # 画像データソースを更新
    r_rgb.data_source.data["image"] = [img]
    r_depth.data_source.data["image"] = [depth_img]

    # ラベルを更新
    label.text = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}, v: {v:.3f}, t: {t:.3f}"

# スライダーにコールバック関数を接続
for slider in [slider1, slider2, slider3, slider4]:
    slider.on_change("value", update_data)

# レイアウトを配置（RGBとDepth画像を横に並べる）
layout = column(slider1, slider2, slider3, slider4, row(p_rgb, p_depth))

# レイアウトを現在のドキュメントに追加
curdoc().add_root(layout)



# bokeh serve --show MyBoke.py --args --ModelName [path/to/model.pth] --dsp 1 --dtp 1 --dvp 3 --pixelshuffle
# /data/data2/tomoya/InSituNetTime/model/CUBE/v1202_pixel_addres/result/model_True_relu1_2_vanilla_80.pth

# bokeh serve --show MyBoke.py --args --ModelName /data/data2/tomoya/InSituNetTime/model/CUBE/v1202_pixel_addres/result/model_True_relu1_2_vanilla_80.pth --dsp 1 --dtp 1 --dvp 3 --pixelshuffle