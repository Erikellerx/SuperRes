import torch
import torch.nn as nn
import math

class EDSRResBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, res_scaling=1):
        super(EDSRResBlock, self).__init__()

        self.res_scaling = res_scaling

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        out = self.conv1(x)   # Conv
        out = self.relu(out)  # ReLU
        out = self.conv2(out) # Conv
        out = out * self.res_scaling
        out = out + x         # Addition/Skip connection

        return out

class EDSR(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int = 64,
            num_res_blocks: int = 16,
            upscale_factor: int = 4,
    ) -> None:
        super(EDSR, self).__init__()

        # Adjust out_channels for the sub-pixel convolution layer (same as ESPCN)
        self.out_channels = int(out_channels * (upscale_factor ** 2))

        # Initial feature extraction layer
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[EDSRResBlock(channels=channels) for _ in range(num_res_blocks)])

        # Convolutional layer after residual blocks
        self.conv_mid = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Upsampling layer (sub-pixel convolution) for x4 scaling
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

        # Final output layer to adjust back to desired output channels
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Custom initialization (similar to ESPCN)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = self.conv_in(x)  # Initial convolution
        residual = x

        x = self.res_blocks(x)  # Pass through residual blocks
        x = self.conv_mid(x)  # Intermediate convolution
        x = x + residual  # Residual connection

        x = self.sub_pixel(x)  # Upsampling (x4)
        x = self.conv_out(x)  # Final output adjustment

        return torch.clamp(x, 0.0, 1.0)

# Helper function to create an EDSR model with x4 scaling
def edsr_x4(in_channels: int, out_channels: int, channels: int = 64) -> EDSR:
    return EDSR(in_channels=in_channels, out_channels=out_channels, channels=channels, upscale_factor=4)

# Test the model with a sample input
if __name__ == "__main__":
    model = edsr_x4(in_channels=3, out_channels=3, channels=64)
    
    from torchsummary import summary
    
    summary(model, (3, 48, 48))