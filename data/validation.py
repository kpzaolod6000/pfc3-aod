import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from net import dehaze_net  # Asegúrate de que tu modelo esté en este archivo
from tunning import DehazingDataset  # Dataset previamente definido


def validate_model(config, loss_type):
    # Cargar modelo
    model = dehaze_net().cuda()
    model.load_state_dict(torch.load(config.model_weights))
    model.eval()

    # Preprocesamiento de imágenes
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Cargar dataset de validación
    val_dataset = DehazingDataset(
        config.val_orig,
        [config.val_hazy_light, config.val_hazy_medium, config.val_hazy_heavy],
        transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Variables para métricas
    total_psnr = 0
    total_ssim = 0
    total_images = 0

    # Validación
    with torch.no_grad():
        for orig, hazy in val_loader:
            orig = orig.cuda()
            hazy = hazy.cuda()

            # Predicción del modelo
            output = model(hazy)

            # Convertir tensores a numpy para calcular PSNR y SSIM
            orig_np = orig.squeeze().permute(1, 2, 0).cpu().numpy()
            output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()

            # Asegurar que los valores estén en el rango [0, 1]
            orig_np = np.clip(orig_np, 0, 1)
            output_np = np.clip(output_np, 0, 1)

            # Calcular PSNR
            psnr_value = psnr(orig_np, output_np, data_range=1.0)
            total_psnr += psnr_value

            # Calcular SSIM
            ssim_value = ssim(orig_np, output_np, data_range=1.0, multichannel=True)
            total_ssim += ssim_value

            total_images += 1

    # Calcular promedios
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images

    print(f"Resultados para {loss_type}:")
    print(f"PSNR promedio: {avg_psnr:.4f}")
    print(f"SSIM promedio: {avg_ssim:.4f}")

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    import argparse

    # Argumentos de configuración
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_orig', type=str, default="/home/pytorch/data/images_gun/val2024")
    parser.add_argument('--val_hazy_light', type=str, default="/home/pytorch/data/results_gun/val2024/light_haze")
    parser.add_argument('--val_hazy_medium', type=str, default="/home/pytorch/data/results_gun/val2024/medium_haze")
    parser.add_argument('--val_hazy_heavy', type=str, default="/home/pytorch/data/results_gun/val2024/heavy_haze")
    parser.add_argument('--model_weights', type=str, default="/home/pytorch/data/snapshots/ms-ssim-l1/dehazer_final.pth")

    config = parser.parse_args()

    # Validar para cada tipo de pérdida
    # validate_model(config, loss_type="ms-ssim")
    validate_model(config, loss_type="ms-ssim+l1")
    # validate_model(config, loss_type="ms-ssim+l2")
    # validate_model(config, loss_type="l2")
