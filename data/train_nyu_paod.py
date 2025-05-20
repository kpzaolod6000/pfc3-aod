import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import argparse
import dataloader
import net
import netbest
from torchvision import transforms
from pytorch_msssim import MS_SSIM
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def weights_init(m):
    """
    Inicializador de pesos para capas Convolucionales y BatchNorm.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def gaussian_weighting(output, target, sigma_g_values=[0.5, 1, 2, 4, 8]):
    """
    Calcula el término G_G^M basado en una pirámide gaussiana simplificada.
    Aplica un suavizado Gaussiano con múltiples sigmas y promedia el MSE.
    Requiere PyTorch >= 1.10 (para F.gaussian_blur).
    """
    G_G_M = 0
    for sigma in sigma_g_values:
        # Aplicar suavizado gaussiano con F.gaussian_blur (PyTorch >= 1.10)
        smoothed_output = TF.gaussian_blur(output, kernel_size=5, sigma=sigma)
        smoothed_target = TF.gaussian_blur(target, kernel_size=5, sigma=sigma)
        
        # Calcular MSE entre las versiones suavizadas
        G_G_M += F.mse_loss(smoothed_output, smoothed_target, reduction='mean')
    
    # Promedio de los valores de sigma
    G_G_M /= len(sigma_g_values)
    return G_G_M


def get_loss_function(loss_type, alpha_l1=0.025, alpha_l2=0.1):
    """
    Devuelve una función de pérdida configurada según 'loss_type'.
    - 'ms-ssim'
    - 'ms-ssim+l1'
    - 'ms-ssim+l2'
    - 'l1'
    - 'l2'
    """
    ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3).cuda()
    l1_loss = nn.L1Loss().cuda()
    l2_loss = nn.MSELoss().cuda()
    
    def compute_loss(output, target):
        loss_ms_ssim = 1 - ms_ssim(output, target)
        loss_l1_val = l1_loss(output, target)
        loss_l2_val = l2_loss(output, target)

        if loss_type == "ms-ssim":
            return loss_ms_ssim
        
        elif loss_type == "ms-ssim+l1":
            return alpha_l1 * loss_ms_ssim + (1 - alpha_l1) * loss_l1_val
        
        elif loss_type == "ms-ssim+l2":
            # Ponderación con G_G^M:
            G_G_M = gaussian_weighting(output, target)
            return alpha_l2 * loss_ms_ssim + (1 - alpha_l2) * G_G_M * loss_l2_val
        
        elif loss_type == "l1":
            return loss_l1_val
        
        elif loss_type == "l2":
            return loss_l2_val
        
        else:
            raise ValueError("Tipo de pérdida no soportado.")

    return compute_loss


def train(config):
    # (Opcional) Acelera convoluciones en GPU si el input tiene dimensiones fijas:
    cudnn.benchmark = True  
    
    # Definir el modelo y aplicar la inicialización de pesos
    dehaze_net = netbest.PAODNet().cuda()
    dehaze_net.apply(weights_init)

    # Carga de datos (train y val)
    train_dataset = dataloader.dehazing_loader(
        config.orig_images_path, 
        config.hazy_images_path
    )
    val_dataset = dataloader.dehazing_loader(
        config.orig_images_path, 
        config.hazy_images_path, 
        mode="val"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.val_batch_size, 
        shuffle=False,  # Val típicamente no se baraja
        num_workers=config.num_workers, 
        pin_memory=True
    )

    # Definir la función de pérdida según config.loss_type
    loss_function = get_loss_function(config.loss_type)

    # Definir optimizador
    optimizer = optim.Adam(
        dehaze_net.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    # Pasamos el modelo a modo train
    dehaze_net.train()

    for epoch in range(config.num_epochs):
        # ---------------------------
        # 1) Etapa de entrenamiento
        # ---------------------------
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)
            loss = loss_function(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            # Clip de gradientes, si se desea
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            # Mostrar avance cada 'display_iter'
            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], "
                      f"Iteration [{iteration+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

            # Guardar snapshot cada 'snapshot_iter'
            if (iteration + 1) % config.snapshot_iter == 0:
                torch.save(
                    dehaze_net.state_dict(), 
                    os.path.join(config.snapshots_folder, f"Epoch{epoch+1}_Iter{iteration+1}.pth")
                )

        # --------------------------------
        # 2) Etapa de validación (inferencia)
        # --------------------------------
        dehaze_net.eval()  # Modo evaluación
        with torch.no_grad():
            for iter_val, (img_orig, img_haze) in enumerate(val_loader):
                img_orig = img_orig.cuda()
                img_haze = img_haze.cuda()

                clean_image = dehaze_net(img_haze)

                # Guardar muestra de validación
                torchvision.utils.save_image(
                    torch.cat((img_haze, clean_image, img_orig), 0),
                    os.path.join(config.sample_output_folder, f"val_{epoch+1}_{iter_val+1}.jpg")
                )
        # Regresamos a modo entrenamiento para la siguiente época
        dehaze_net.train()

    # Guardar último modelo al finalizar todas las épocas
    torch.save(
        dehaze_net.state_dict(), 
        os.path.join(config.snapshots_folder, "dehazer_final.pth")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parámetros de entrada
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/NYU/PAOD_MS_SSIM_L2_G")
    parser.add_argument('--sample_output_folder', type=str, default="samples/NYU/PAOD_MS_SSIM_L2_G")
    parser.add_argument('--loss_type', type=str, default="ms-ssim", 
                        choices=["ms-ssim", "ms-ssim+l1", "ms-ssim+l2", "l1", "l2"])

    config = parser.parse_args()

    # Creación de directorios con exist_ok=True para simplificar
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.sample_output_folder, exist_ok=True)

    # Llamada a la función de entrenamiento
    train(config)
