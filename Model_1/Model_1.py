import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        return F.relu(x)

class LiteCNNUpscaler(nn.Module):
    def __init__(self, input_size=(540, 960)):
        super().__init__()
        # Mniej kanałów
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)

        # Jeden residual block zamiast dwóch
        self.res1 = ResidualBlock(16)

        self.final_conv = nn.Conv2d(16, 3, 3, padding=1)
        self.input_size = input_size

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.res1(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)


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
model = LiteCNNUpscaler()
estimate_vram(model, batch_size=1, input_size=(3, 540, 960))