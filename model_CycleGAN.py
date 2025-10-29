from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, norm_layer: nn.Module) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """The residual generator used in the original CycleGAN paper."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")

        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf * mult, norm_layer)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(inplace=True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()

        kw = 4
        padw = 1
        sequence: List[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights(module: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    classname = module.__class__.__name__
    if hasattr(module, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        if init_type == "normal":
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "xavier":
            nn.init.xavier_normal_(module.weight.data, gain=gain)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight.data, gain=gain)
        else:
            raise NotImplementedError(f"initialization method '{init_type}' is not implemented")
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)


def build_generator(input_nc: int, output_nc: int, **kwargs) -> ResnetGenerator:
    net = ResnetGenerator(input_nc=input_nc, output_nc=output_nc, **kwargs)
    net.apply(init_weights)
    return net


def build_discriminator(input_nc: int, **kwargs) -> NLayerDiscriminator:
    net = NLayerDiscriminator(input_nc=input_nc, **kwargs)
    net.apply(init_weights)
    return net
