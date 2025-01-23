import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLO, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(192, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_layers = nn.Sequential(
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * (grid_size // 32) ** 2, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
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
    
    
# "yolo"        : {"image_size" : (416, 416),  # YOLO typically uses 416x416 images
#                                         "normalize"  : {"mean" : [0.485, 0.456, 0.406], 
#                                                         "std"  : [0.229, 0.224, 0.225] 
#                                                        }
#                                        }