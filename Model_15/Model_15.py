import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CNNUpscaler(nn.Module):
    def __init__(self, scale_factor=2, num_blocks=64, channels=64):
        super().__init__()

        self.first_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        upsample_layers = []
        for _ in range(int(scale_factor).bit_length() - 1):
            upsample_layers += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        self.res_blocks2 = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(2)]
        )

        self.final_conv = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.res_blocks2(x)
        x = self.final_conv(x)
        return x
        # torch.clamp(x, min=0.0, max=1.0)


def estimate_vram(model, batch_size=1, input_size=(3, 540, 960), dtype=torch.float32):
    pass