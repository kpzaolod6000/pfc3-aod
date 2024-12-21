import torch
import torch.nn as nn
import torch.nn.functional as F

class PfAAM(nn.Module):
    """Parameter-free Average Attention Module (PfAAM)"""
    def __init__(self):
        super(PfAAM, self).__init__()

    def forward(self, x):
        avg_hw = torch.mean(x, dim=(2, 3), keepdim=True)  # HW Average: 1x1xC
        avg_c = torch.mean(x, dim=1, keepdim=True)       # Channel Average: CxHxW

        elemWiseMultiply = torch.sigmoid(avg_hw.expand_as(x) * avg_c.expand_as(x))
        x = elemWiseMultiply * x  # Element-wise multiplication
        return x
    
class PAODNet(nn.Module):
    def __init__(self):
        super(PAODNet, self).__init__()

        # Convolucionales con distintos kernels
        self.e_conv1 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.e_conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.e_conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.e_conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.e_conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pfaam = PfAAM()

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.pfaam(self.relu(self.e_conv1(x)))
        x2 = self.pfaam(self.relu(self.e_conv2(x1)))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.pfaam(self.relu(self.e_conv3(concat1)))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.pfaam(self.relu(self.e_conv4(concat2)))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.pfaam(self.relu(self.e_conv5(concat3)))

        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


# Ejemplo de entrenamiento con datos de ejemplo:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PAODNet().to(device)

    # Ejemplo de una imagen de entrada (batch_size=1, channels=3, height=256, width=256)
    input_image = torch.randn(1, 3, 256, 256).to(device)
    output = model(input_image)
    print("Salida del modelo:", output.shape)  # Salida esperada: (1, 3, 256, 256)
