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
import netbest

def dehaze_image(image_path, model_path):
    # Cargar la imagen con neblina
    data_hazy = Image.open(image_path).convert("RGB")
    data_hazy = np.asarray(data_hazy, dtype=np.float32) / 255.0  # Normalizar [0,1]
    
    # Convertir a tensor PyTorch: [H,W,C] -> [C,H,W]
    data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1).unsqueeze(0).cuda()

    # Cargar el modelo
    dehaze_net = netbest.PAODNet().cuda()
    dehaze_net.load_state_dict(torch.load(model_path))

    # Procesar la imagen
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)
    
    # Crear carpeta 'results' si no existe
    # os.makedirs("results", exist_ok=True)
    # Guardar la imagen restaurada en la carpeta 'results'
    # nombre_salida = os.path.join("results", os.path.basename(image_path))
    # torchvision.utils.save_image(clean_image, nombre_salida)

    # Convertir de tensor [1,C,H,W] a NumPy [H,W,C]
    clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Asegurarse de que los valores estén en [0,1]
    clean_image_np = np.clip(clean_image_np, 0.0, 1.0)
    return clean_image_np

def calculate_metrics(gt_image_path, restored_image):
    # Cargar la imagen ground truth (GT)
    gt_image = Image.open(gt_image_path).convert("RGB")
    gt_image = np.asarray(gt_image, dtype=np.float32) / 255.0

    # Calcular PSNR
    psnr_value = psnr(gt_image, restored_image, data_range=1.0)

    # Calcular SSIM (usando channel_axis=-1 en lugar de multichannel para skimage >= 0.19)
    ssim_value = ssim(gt_image, restored_image, data_range=1.0, channel_axis=-1)

    return psnr_value, ssim_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Ruta al archivo .pth del modelo entrenado')
    parser.add_argument('--hazy_folder_path', type=str, required=True, 
                        help='Ruta de la carpeta con imágenes hazy')
    parser.add_argument('--gt_folder_path', type=str, required=True, 
                        help='Ruta de la carpeta con imágenes ground truth')
    args = parser.parse_args()

    # Crear lista de imágenes hazy
    # Asumiendo que son .jpg o .png, ajusta la extensión según tus archivos
    hazy_images = sorted(glob.glob(os.path.join(args.hazy_folder_path, '*.*')))

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # Iterar sobre cada imagen hazy
    for hazy_image_path in hazy_images:
        # Encontrar la imagen ground truth correspondiente
        # (asumimos que tienen el mismo nombre de archivo)
        filename = os.path.basename(hazy_image_path)  
        gt_image_path = os.path.join(args.gt_folder_path, filename)

        # Verificar que exista la imagen GT
        if not os.path.exists(gt_image_path):
            print(f"[ADVERTENCIA] No existe el GT para: {filename}. Se omite esta imagen.")
            continue

        # Restaurar imagen
        restored_image = dehaze_image(hazy_image_path, args.model_path)

        # Calcular métricas
        psnr_val, ssim_val = calculate_metrics(gt_image_path, restored_image)
        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1
    
        print(f"Number: {count}")
        print(f"Imagen: {filename}")
        print(f"  PSNR: {psnr_val:.2f}")
        print(f"  SSIM: {ssim_val:.4f}\n")

    # Calcular promedios
    if count > 0:
        avg_psnr = total_psnr / count   
        avg_ssim = total_ssim / count
        print(f"Promedio PSNR en {count} imágenes: {avg_psnr:.2f}")
        print(f"Promedio SSIM en {count} imágenes: {avg_ssim:.4f}")
    else:
        print("No se procesaron imágenes debido a falta de coincidencia o carpeta vacía.")
