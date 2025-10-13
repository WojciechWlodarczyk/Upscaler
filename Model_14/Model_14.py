import torch
import torch.nn as nn
import torch.nn.functional as F

# Lekko zmodyfikowana, "ESRGAN-lite" implementacja generatora z RRDB, SEBlock i globalnym skip
# Zawiera:
# - DenseResidualBlock (wewnętrzne gęste połączenia)
# - RRDB (Residual-in-Residual Dense Block)
# - SEBlock (channel attention)
# - Global skip connection (long skip)
# - Upsample przez PixelShuffle
#
# Kod ma na celu być czytelny i łatwy do uruchomienia — nie zawiera części treningowej (loss/optimizers),
# ale na dole pliku dodano prosty test inicjalizacji i przepuszczenia losowego tensora.


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DenseResidualBlock(nn.Module):
    """Dense block similar to ESRGAN's DB but simplified.
    Produces output of same number of channels as input by a 1x1 projection at the end if needed.

    Architecture:
    x -> conv(growth) -> PReLU -> concat -> conv(growth) -> ... (5 layers)
    output = x + scale * projection(cat(outputs))
    """

    def __init__(self, channels, growth=32, n_layers=5, scale=0.2):
        super().__init__()
        self.channels = channels
        self.growth = growth
        self.n_layers = n_layers
        self.scale = scale

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch = channels + i * growth
            self.convs.append(nn.Conv2d(in_ch, growth, kernel_size=3, padding=1))

        # projection to match input channels (1x1 conv)
        self.proj = nn.Conv2d(channels + n_layers * growth, channels, kernel_size=1, padding=0)
        self.act = nn.PReLU()

    def forward(self, x):
        inputs = [x]
        for conv in self.convs:
            inp = torch.cat(inputs, dim=1)
            out = self.act(conv(inp))
            inputs.append(out)
        cat = torch.cat(inputs, dim=1)
        out = self.proj(cat)
        return x + self.scale * out


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: stack 3 DenseResidualBlock z jednym skip i skalowaniem.
    """
    def __init__(self, channels, growth=32, n_layers=5, scale=0.2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            DenseResidualBlock(channels, growth=growth, n_layers=n_layers, scale=scale)
            for _ in range(3)
        ])
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.blocks(x)


class RRDBNet(nn.Module):
    """Generator network inspired by ESRGAN / EDSR with improvements:
    - first conv
    - trunk of RRDB blocks
    - global residual skip (add trunk output to first_conv output)
    - upsampling via PixelShuffle
    - final reconstruction conv
    - optional SEBlock (channel attention)

    Args:
        in_ch: input channels (3 RGB)
        out_ch: output channels (3 RGB)
        channels: base feature maps
        num_rrdb: number of RRDB blocks in trunk
        growth: growth channels for dense blocks
        scale: scale factor (2,4,8)
        use_se: add SEBlock after trunk or inside blocks
    """

    def __init__(self, in_ch=3, out_ch=3, channels=64, num_rrdb=23, growth=32, scale=2, use_se=True):
        super().__init__()
        assert scale in (2, 4, 8), "scale must be 2, 4 or 8"

        self.scale = scale
        self.use_se = use_se

        # initial conv (feature extraction)
        self.first_conv = nn.Conv2d(in_ch, channels, kernel_size=3, padding=1)

        # trunk of RRDBs (feature-transform)
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(channels, growth=growth))
        self.trunk = nn.Sequential(*rrdb_blocks)

        # trunk conv (1 conv after trunk like original ESRGAN)
        self.trunk_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # optional channel attention
        if use_se:
            self.se = SEBlock(channels)
        else:
            self.se = None

        # upsampling layers using PixelShuffle
        upsample_layers = []
        # number of upsampling blocks is log2(scale)
        n_upsamples = {2: 1, 4: 2, 8: 3}[scale]
        for _ in range(n_upsamples):
            upsample_layers += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        # final reconstruction conv
        self.final_conv = nn.Conv2d(channels, out_ch, kernel_size=3, padding=1)

        # weight initialization
        self._initialize_weights()

    def forward(self, x):
        fea = self.first_conv(x)
        trunk = self.trunk(fea)
        trunk = self.trunk_conv(trunk)

        if self.se is not None:
            trunk = self.se(trunk)

        # global skip connection (long skip)
        fea = fea + trunk

        out = self.upsample(fea)
        out = self.final_conv(out)
        return out

    def _initialize_weights(self):
        # He initialization for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Opcjonalny prosty PatchGAN discriminator — przydatny jeśli chcesz trenować GAN
class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_channels=64):
        super().__init__()
        layers = []
        c = base_channels
        layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]

        for i, mult in enumerate([1, 2, 4, 8]):
            layers += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(c * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            c *= 2

        # final classifier
        layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1)


# VGG feature extractor for perceptual loss (użyj w treningu, poza modelem)
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35, use_input_norm=True, device=None):
        super().__init__()
        # import wewnątrz, żeby nie wymuszać zależności przy samym imporcie pliku
        from torchvision.models import vgg19
        vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])
        # zamrażamy parametry
        for p in self.features.parameters():
            p.requires_grad = False

        self.use_input_norm = use_input_norm
        if use_input_norm:
            # mean/std (ImageNet)
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            if device is not None:
                mean = mean.to(device)
                std = std.to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)


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