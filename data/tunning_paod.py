import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import torchvision.utils as vutils
from netbest import PAODNet  # Asegúrate de que el modelo está en `net.py`
from pytorch_msssim import MS_SSIM


# Dataset personalizado
class DehazingDataset(Dataset):
    def __init__(self, original_path, hazy_paths, transform=None):
        """
        original_path: Ruta con imágenes originales sin niebla.
        hazy_paths: Lista de rutas con imágenes con diferentes densidades de niebla (ligera, mediana, pesada).
        transform: Transformaciones a aplicar a las imágenes.
        """
        self.original_path = original_path
        self.hazy_paths = hazy_paths
        self.image_filenames = sorted(os.listdir(original_path))
        self.transform = transform

        # Comprobar que todas las rutas de niebla tienen el mismo número de imágenes
        for path in hazy_paths:
            assert len(self.image_filenames) == len(sorted(os.listdir(path))), \
                f"La carpeta {path} no tiene el mismo número de imágenes que la original."

    def __len__(self):
        # El total de muestras es el número de imágenes originales multiplicado por el número de niveles de niebla
        return len(self.image_filenames) * len(self.hazy_paths)

    def __getitem__(self, idx):
        # Cálculo de índices:
        #   hazy_idx determina qué nivel de niebla se usará.
        #   image_idx es el índice real de la imagen original.
        hazy_idx = idx % len(self.hazy_paths)
        image_idx = idx // len(self.hazy_paths)

        orig_img_path = os.path.join(self.original_path, self.image_filenames[image_idx])
        hazy_img_path = os.path.join(self.hazy_paths[hazy_idx], self.image_filenames[image_idx])

        orig_img = Image.open(orig_img_path).convert("RGB")
        hazy_img = Image.open(hazy_img_path).convert("RGB")

        if self.transform:
            orig_img = self.transform(orig_img)
            hazy_img = self.transform(hazy_img)

        return orig_img, hazy_img


def train(config):
    # Inicializar el modelo
    model = PAODNet().cuda()

    # Cargar pesos preentrenados si existen
    if os.path.exists(config.pretrained_weights):
        model.load_state_dict(torch.load(config.pretrained_weights))
        print(f"Pesos cargados desde {config.pretrained_weights}")

    # Preprocesamiento de imágenes
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ajusta esta resolución a tu necesidad
        transforms.ToTensor()
    ])

    # Cargar datasets con múltiples densidades de niebla
    train_dataset = DehazingDataset(
        config.train_orig,
        [config.train_hazy_light, config.train_hazy_medium, config.train_hazy_heavy],
        transform
    )
    val_dataset = DehazingDataset(
        config.val_orig,
        [config.val_hazy_light, config.val_hazy_medium, config.val_hazy_heavy],
        transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Definir funciones de pérdida
    ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3).cuda()
    l1_loss = nn.L1Loss().cuda()
    l2_loss = nn.MSELoss().cuda()

    # Coeficientes de combinación
    alpha_l1 = 0.025  # Para MS-SSIM + L1
    alpha_l2 = 0.1    # Para MS-SSIM + L2

    # Configurar optimizador
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Entrenamiento
    model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for i, (orig, hazy) in enumerate(train_loader):
            orig, hazy = orig.cuda(), hazy.cuda()

            # Forward
            output = model(hazy)

            # Cálculo de pérdidas individuales
            loss_ms_ssim = 1 - ms_ssim(output, orig)  # Convertir MS-SSIM en pérdida
            loss_l1_val = l1_loss(output, orig)       # L1
            loss_l2_val = l2_loss(output, orig)       # L2

            # Selección del tipo de pérdida según config
            if config.loss_type == "ms-ssim":
                loss = loss_ms_ssim
            elif config.loss_type == "ms-ssim+l1":
                loss = loss_ms_ssim + alpha_l1 * loss_l1_val
            elif config.loss_type == "ms-ssim+l2":
                loss = loss_ms_ssim + alpha_l2 * loss_l2_val
            elif config.loss_type == "l1":
                loss = loss_l1_val
            elif config.loss_type == "l2":
                loss = loss_l2_val
            else:
                raise ValueError("Tipo de pérdida no soportado. Usa: ms-ssim, ms-ssim+l1, ms-ssim+l2, l1 o l2.")

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}")

        # Guardar modelo al final de la época
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, f"dehaze_epoch_{epoch+1}.pth"))

        # Validación
        model.eval()
        with torch.no_grad():
            for j, (orig, hazy) in enumerate(val_loader):
                orig, hazy = orig.cuda(), hazy.cuda()
                output = model(hazy)
                comparison = torch.cat((hazy, output, orig), dim=0)
                vutils.save_image(comparison, os.path.join(config.sample_output_folder, f"val_epoch_{epoch+1}_sample_{j+1}.jpg"))
        model.train()

    # Guardar el modelo final
    torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "dehazer_final.pth"))
    print("Entrenamiento completo. Modelo guardado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directorios del dataset
    parser.add_argument('--train_orig', type=str, default="/home/pytorch/data/images_gun/train2024")
    parser.add_argument('--train_hazy_light', type=str, default="/home/pytorch/data/results_gun/train2024/light_haze")
    parser.add_argument('--train_hazy_medium', type=str, default="/home/pytorch/data/results_gun/train2024/medium_haze/")
    parser.add_argument('--train_hazy_heavy', type=str, default="/home/pytorch/data/results_gun/train2024/heavy_haze")
    parser.add_argument('--val_orig', type=str, default="images_gun/val2024")
    parser.add_argument('--val_hazy_light', type=str, default="/home/pytorch/data/results_gun/val2024/light_haze")
    parser.add_argument('--val_hazy_medium', type=str, default="/home/pytorch/data/results_gun/val2024/medium_haze")
    parser.add_argument('--val_hazy_heavy', type=str, default="/home/pytorch/data/results_gun/val2024/heavy_haze")

    # Configuración del entrenamiento
    parser.add_argument('--pretrained_weights', type=str, default="/home/pytorch/data/snapshots/dehazer.pth")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--loss_type', type=str, default="ms-ssim", 
                        choices=["ms-ssim", "ms-ssim+l1", "ms-ssim+l2", "l1", "l2"], 
                        help="Tipo de pérdida a usar: ms-ssim, ms-ssim+l1, ms-ssim+l2, l1 o l2.")

    config = parser.parse_args()

    # Crear carpetas si no existen
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.sample_output_folder, exist_ok=True)

    # Entrenar el modelo
    train(config)
