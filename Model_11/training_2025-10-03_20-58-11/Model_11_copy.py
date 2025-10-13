import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block z residual scaling (EDSR/ESRGAN styl)
class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        out = self.conv2(out)
        return residual + self.res_scale * out


# Główny model
class LiteCNNUpscaler(nn.Module):
    def __init__(self, scale_factor=2, num_blocks=16, channels=64):
        super().__init__()

        # Ekstrakcja cech wejściowych
        self.first_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # Stos bloków residual
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Upsampling (PixelShuffle)
        upsample_layers = []
        for _ in range(int(scale_factor).bit_length() - 1):  # obsługa x2, x4, x8
            upsample_layers += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        # Końcowa rekonstrukcja RGB
        self.final_conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        return x
        # torch.clamp(x, min=0.0, max=1.0)


def estimate_vram(model, batch_size=1, input_size=(3, 540, 960), dtype=torch.float32):
    pass
    """
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
    print(f"Przybliżona VRAM (forward + backward ~2x): {(param_mem + 2 * fm_mem):.2f} MB"
    """


# Przykład użycia:
#model = LiteCNNUpscaler()
#estimate_vram(model, batch_size=1, input_size=(3, 540, 960))