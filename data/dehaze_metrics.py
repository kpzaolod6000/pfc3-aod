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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(orig, pred):
    """
    Calcula PSNR y SSIM entre la imagen original y la predicha.
    :param orig: Tensor de la imagen original (C, H, W).
    :param pred: Tensor de la imagen predicha (C, H, W).
    :return: Diccionario con PSNR y SSIM.
    """
    orig_np = orig.permute(1, 2, 0).cpu().numpy()  # Convertir a numpy (H, W, C)
    pred_np = pred.permute(1, 2, 0).cpu().numpy()

    psnr_value = peak_signal_noise_ratio(orig_np, pred_np, data_range=1.0)
    ssim_value = structural_similarity(orig_np, pred_np, data_range=1.0, multichannel=True)

    return {"PSNR": psnr_value, "SSIM": ssim_value}

def dehaze_image(image_path, metrics, model_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy)/255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))

    clean_image = dehaze_net(data_hazy)

    # Guardar resultados visuales
    torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])

    # Calcular métricas
    orig_path = image_path.replace("test_images", "ground_truth")  # Ruta de la imagen original
    if os.path.exists(orig_path):
        data_orig = Image.open(orig_path)
        data_orig = (np.asarray(data_orig)/255.0)
        data_orig = torch.from_numpy(data_orig).float().permute(2, 0, 1).cuda()
        metric_values = compute_metrics(data_orig, clean_image.squeeze(0))
        metrics.append(metric_values)
        print(f"Metrics for {image_path}: PSNR={metric_values['PSNR']:.4f}, SSIM={metric_values['SSIM']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth del modelo entrenado')
    args = parser.parse_args()

    test_list = glob.glob("test_images/*")
    metrics = []

    for image in test_list:
        dehaze_image(image, metrics, args.model_path)
        print(image, "done!")

    # Promedio de métricas
    if metrics:
        avg_psnr = np.mean([m["PSNR"] for m in metrics])
        avg_ssim = np.mean([m["SSIM"] for m in metrics])
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
