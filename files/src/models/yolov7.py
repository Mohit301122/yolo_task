import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # YOLOv6 uses SiLU instead of LeakyReLU

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.use_identity = in_channels == out_channels

    def forward(self, x):
        main = self.conv1(x)
        branch = self.conv2(x)
        if self.use_identity:
            return main + branch + x
        return main + branch

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.down = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBlock(out_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(out_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        
        self.rep_blocks = nn.Sequential(*[
            RepBlock(mid_channels, mid_channels) for _ in range(num_blocks)
        ])
        
        self.concat_conv = ConvBlock(mid_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.down(x)
        x1 = self.conv1(x)
        x2 = self.rep_blocks(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)
        return self.concat_conv(x)

class YOLOv6(nn.Module):
    def __init__(self, grid_size=20, num_boxes=3, num_classes=80):
        super(YOLOv6, self).__init__()
        
        # Initial convolution
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=2, padding=1),
            CSPBlock(64, 128, num_blocks=1),
            CSPBlock(128, 256, num_blocks=2),
            CSPBlock(256, 512, num_blocks=3),
        )
        
        # Neck layers
        self.conv_layers = nn.Sequential(
            RepBlock(512, 256),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            RepBlock(512, 256),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            RepBlock(512, 256),
        )
        
        # Detection head (following similar format to original code)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (grid_size // 32) ** 2, 4096),
            nn.Dropout(0.5),
            nn.SiLU(),
            nn.Linear(4096, grid_size * grid_size * (num_boxes * 5 + num_classes)),
        )
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
        return x

# Example usage
def create_model(grid_size=20, num_boxes=3, num_classes=80):
    model = YOLOv6(grid_size=grid_size, num_boxes=num_boxes, num_classes=num_classes)
    return model
