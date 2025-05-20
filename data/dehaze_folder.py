import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

def dehaze_image(image_path, output_dir, model_path):
    data_hazy = Image.open(image_path).convert('RGB')
    data_hazy = (np.asarray(data_hazy) / 255.0).astype(np.float32)
    data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1).unsqueeze(0).cuda()
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))
    clean_image = dehaze_net(data_hazy)
    os.makedirs(output_dir, exist_ok=True)
    nombre_salida = os.path.join(output_dir, os.path.basename(image_path))
    torchvision.utils.save_image(clean_image, nombre_salida)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--haze_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)
    args = parser.parse_args()
    haze_images_list = glob.glob(os.path.join(args.haze_dir, '*'))
    for image_path in haze_images_list:
        dehaze_image(image_path, args.output_dir, args.weight_path)
        print(image_path, "procesada")

if __name__ == '__main__':
    main()
