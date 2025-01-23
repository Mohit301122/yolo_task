import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  
        
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels//2, kernel_size=1, stride=1, padding=0),
            ConvBlock(channels//2, channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPBlock, self).__init__()
        self.downsample = ConvBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.trans1 = ConvBlock(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.trans2 = ConvBlock(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels//2) for _ in range(num_blocks)])
        self.concat = ConvBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.downsample(x)
        x1 = self.trans1(x)
        x2 = self.blocks(self.trans2(x))
        return self.concat(torch.cat([x1, x2], dim=1))

class YOLOv8(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv8, self).__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            CSPBlock(64, 128, num_blocks=3),
            CSPBlock(128, 256, num_blocks=6),
            CSPBlock(256, 512, num_blocks=9),
            CSPBlock(512, 1024, num_blocks=3)
        )
        
        # Neck (FPN + PAN)
        self.fpn = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
        )
        
        self.pan = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )
        
        # Detection heads for different scales
        self.detect_1 = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, (num_classes + 4), kernel_size=1)  # cls + box
        )
        
        self.detect_2 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, (num_classes + 4), kernel_size=1)  # cls + box
        )
        
        self.detect_3 = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, (num_classes + 4), kernel_size=1)  # cls + box
        )

    def forward(self, x):
        # Backbone forward
        x1 = self.backbone[0:4](x)    # P3
        x2 = self.backbone[4:5](x1)   # P4
        x3 = self.backbone[5:](x2)    # P5
        
        # FPN forward
        fpn_out = self.fpn(x3)
        
        # PAN forward
        pan_out = self.pan(fpn_out)
        
        # Detection heads
        small = self.detect_1(pan_out)    # Small objects
        medium = self.detect_2(torch.cat([pan_out, x2], dim=1))  # Medium objects
        large = self.detect_3(torch.cat([medium, x3], dim=1))    # Large objects
        
        # Reshape outputs for each detection head
        batch_size = x.shape[0]
        small = small.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes + 4)
        medium = medium.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes + 4)
        large = large.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes + 4)
        
        return torch.cat([small, medium, large], dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)