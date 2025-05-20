import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob
import net

def dehaze_image(image_path, model_path):
    # Cargar la imagen con neblina
    data_hazy = Image.open(image_path).convert("RGB")
    data_hazy = np.asarray(data_hazy, dtype=np.float32) / 255.0  # Normalizar [0,1]
    
    # Convertir a tensor PyTorch: [H,W,C] -> [C,H,W]
    data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1).unsqueeze(0).cuda()

    # Cargar el modelo
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))

    # Procesar la imagen
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)
    
    # Crear carpeta 'results' si no existe
    os.makedirs("/home/pytorch/data/results_gun/imagenes_selectivas/heavy_dehaze", exist_ok=True)
    # Guardar la imagen restaurada en la carpeta 'results'
    nombre_salida = os.path.join("/home/pytorch/data/results_gun/imagenes_selectivas/heavy_dehaze", os.path.basename(image_path))
    torchvision.utils.save_image(clean_image, nombre_salida)

    # Convertir de tensor [1,C,H,W] a NumPy [H,W,C]
    clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return clean_image_np

def calculate_metrics(hazy_image_path, gt_image_path, restored_image):
    # Cargar la imagen ground truth (GT)
    gt_image = np.asarray(Image.open(gt_image_path).convert("RGB"), dtype=np.float32) / 255.0

    # (Opcional) Cargar la imagen hazy si la necesitas para algo extra
    # hazy_image = np.asarray(Image.open(hazy_image_path).convert("RGB"), dtype=np.float32) / 255.0
    
    # Asegurarnos de que restored_image esté entre [0,1]
    restored_image = np.clip(restored_image, 0.0, 1.0)

    # Calcular PSNR
    psnr_value = psnr(gt_image, restored_image, data_range=1.0)

    # Calcular SSIM (recomendado usar channel_axis en versiones recientes de skimage)
    ssim_value = ssim(
        gt_image,
        restored_image,
        data_range=1.0,
        channel_axis=-1  # si tu skimage >= 0.19
    )

    return psnr_value, ssim_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Ruta al archivo .pth del modelo entrenado')
    parser.add_argument('--hazy_image_path', type=str, required=True, 
                        help='Ruta de la imagen con neblina (hazy)')
    parser.add_argument('--orig_image_path', type=str, required=True, 
                        help='Ruta de la imagen original (ground truth)')
    args = parser.parse_args()

    # Procesar la imagen y restaurarla
    restored_image = dehaze_image(args.hazy_image_path, args.model_path)

    # Calcular métricas
    psnr_value, ssim_value = calculate_metrics(args.hazy_image_path, 
                                               args.orig_image_path, 
                                               restored_image)

    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
