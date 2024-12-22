import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

bn_mom = 0.0003

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b0', downsample_factor=16):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        self.downsample_factor = downsample_factor

        # Adjust stride_list based on downsample_factor
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('EfficientNetBackbone: downsample_factor=%d is not supported.' % downsample_factor)

        # Define layers similar to Xception
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)

        # Adjust the number of channels for the first block
        first_block = self.model._blocks[0]
        in_channels = 32  # Based on the output channels of self.conv1
        out_channels = first_block._project_conv.weight.size(0)  # Output channels of the first block
        self.block1 = EfficientNetBlock(first_block, in_channels, out_channels, stride=stride_list[0])

        # Adjust the number of channels for the second block
        second_block = self.model._blocks[1]
        in_channels = out_channels
        out_channels = second_block._project_conv.weight.size(0)  # Output channels of the second block
        self.block2 = EfficientNetBlock(second_block, in_channels, out_channels, stride=stride_list[1])

        # Adjust the number of channels for the third block
        third_block = self.model._blocks[2]
        in_channels = out_channels
        out_channels = third_block._project_conv.weight.size(0)  # Output channels of the third block
        self.block3 = EfficientNetBlock(third_block, in_channels, out_channels, stride=stride_list[2])

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_feature_layer = x  # Store low feature layer

        x = self.block2(x)
        x = self.block3(x)

        return low_feature_layer, x

class EfficientNetBlock(nn.Module):
    def __init__(self, block, in_channels, out_channels, stride=1):
        super(EfficientNetBlock, self).__init__()
        self.block = block

        # Adjust the depthwise convolution
        self.block._depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                               padding=1, groups=in_channels, bias=False)

        # Adjust the project convolution
        self.block._project_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                             padding=0, bias=False)

    def forward(self, x):
        x = self.block(x)
        return x

if __name__ == '__main__':
    # Initialize a random input tensor
    batch_size = 4
    channels = 3
    height, width = 512, 512
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Create an instance of EfficientNetBackbone
    backbone = EfficientNetBackbone(model_name='efficientnet-b0', downsample_factor=16)

    # Forward pass through the backbone
    low_feature_layer, x = backbone(input_tensor)

    # Print the shapes of the output tensors
    print(f'Low-level feature layer shape: {low_feature_layer.shape}')
    print(f'Final output tensor shape: {x.shape}')
