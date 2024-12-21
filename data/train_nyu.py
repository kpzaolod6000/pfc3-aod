import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import argparse
import dataloader
import net
from torchvision import transforms
from pytorch_msssim import MS_SSIM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_loss_function(loss_type, alpha_l1=0.025, alpha_l2=0.1):
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
            return loss_ms_ssim + alpha_l1 * loss_l1_val
        elif loss_type == "ms-ssim+l2":
            return loss_ms_ssim + alpha_l2 * loss_l2_val
        elif loss_type == "l1":
            return loss_l1_val
        elif loss_type == "l2":
            return loss_l2_val
        else:
            raise ValueError("Tipo de pérdida no soportado. Usa: ms-ssim, ms-ssim+l1, ms-ssim+l2, l1 o l2.")

    return compute_loss

def train(config):
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )

    loss_function = get_loss_function(config.loss_type)

    optimizer = optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            loss = loss_function(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            if (iteration + 1) % config.snapshot_iter == 0:
                torch.save(dehaze_net.state_dict(), os.path.join(config.snapshots_folder, f"Epoch{epoch+1}_Iter{iteration+1}.pth"))

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            torchvision.utils.save_image(
                torch.cat((img_haze, clean_image, img_orig), 0),
                os.path.join(config.sample_output_folder, f"val_{iter_val+1}.jpg")
            )

        torch.save(dehaze_net.state_dict(), os.path.join(config.snapshots_folder, "dehazer_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/NYU/L1")
    parser.add_argument('--sample_output_folder', type=str, default="samples/NYU/L1")
    parser.add_argument('--loss_type', type=str, default="ms-ssim", choices=["ms-ssim", "ms-ssim+l1", "ms-ssim+l2", "l1", "l2"])

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
