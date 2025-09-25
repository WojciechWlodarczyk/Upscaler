import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
    #    self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    #    self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # skip connection
        x += residual
        return F.relu(x)


class LiteCNNUpscaler(nn.Module):
    def __init__(self, input_size=(540, 960)):
        super().__init__()
        self.input_size = input_size

        # wejściowa konwolucja + batchnorm
        self.first_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    #    self.first_bn = nn.BatchNorm2d(16)

        # stos bloków residual
        self.res1 = ResidualBlock(16)
        self.res2 = ResidualBlock(16)
        self.res3 = ResidualBlock(16)
        self.res4 = ResidualBlock(16)

        # końcowa konwolucja + batchnorm
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    #    self.final_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        h, w = self.input_size

        # powiększanie
        x = F.interpolate(x, size=(h * 2, w * 2), mode='bilinear', align_corners=False)

        # ekstrakcja cech wejściowych
        x = F.relu(self.first_conv(x))

        # przepuszczanie przez bloki residual
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        # rekonstrukcja RGB
        x = self.final_conv(x)

        return x
        # torch.clamp(x, min=0.0, max=1.0)


def estimate_vram(model, batch_size=1, input_size=(3, 540, 960), dtype=torch.float32):
    device = 'cpu'
    dummy_input = torch.zeros((batch_size, *input_size), dtype=dtype, device=device)

    # Liczymy parametry
    param_mem = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2  # MB
    print(f"Parametry: {param_mem:.2f} MB")

    # Liczymy feature maps
    def hook(module, input, output):
        fl = output.numel() * 4 / 1024 ** 2  # MB
        feature_maps.append(fl)

    feature_maps = []
    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.MaxPool2d)):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        _ = model(dummy_input)
    for h in hooks: h.remove()

    fm_mem = sum(feature_maps)
    print(f"Feature maps (forward only): {fm_mem:.2f} MB")
    print(f"Przybliżona VRAM (forward + backward ~2x): {(param_mem + 2 * fm_mem):.2f} MB")


# Przykład użycia:
#model = LiteCNNUpscaler()
#estimate_vram(model, batch_size=1, input_size=(3, 540, 960))