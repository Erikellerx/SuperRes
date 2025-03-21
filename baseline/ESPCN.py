import math

import torch
from torch import nn, Tensor

__all__ = [
    "ESPCN",
    "espcn_x2", "espcn_x3", "espcn_x4", "espcn_x8",
]


class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int,
    ) -> None:
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


def espcn_x2(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=2, **kwargs)

    return model


def espcn_x3(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=3, **kwargs)

    return model


def espcn_x4(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=4, **kwargs)

    return model


def espcn_x8(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=8, **kwargs)

    return model

if __name__ == "__main__":
    
    model = espcn_x4(in_channels=3, out_channels=3, channels=64)
    
    from torchsummary import summary
    
    summary(model, (3, 48, 48))