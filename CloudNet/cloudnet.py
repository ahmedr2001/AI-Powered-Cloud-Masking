import torch
import torch.nn as nn
import torch.nn.functional as F

class EncConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if p_dropout:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1_bn(self.conv1(x)))
        x_concat = torch.cat((x, x1), dim=1)
        x2 = self.relu(self.conv2_bn(self.conv2(x)))
        x2 = self.conv3(x2)
        if self.dropout:
            x2 = self.dropout(x2)
        x2 = self.relu(self.conv3_bn(x2))
        x_add = x2 + x_concat
        x_add = self.relu(x_add)

        return x_add
    
class EncBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv5_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1_bn(self.conv1(x)))
        x_concat = torch.cat((x, x1), dim=1)
        x2 = self.relu(self.conv2_bn(self.conv2(x)))
        x2 = self.relu(self.conv3_bn(self.conv3(x2)))
        x5 = self.relu(self.conv5_bn(self.conv5(x2)))
        x2 = self.relu(self.conv4_bn(self.conv4(x2)))
        x_add = x2 + x_concat + x5
        x_add = self.relu(x_add)
        
        return x_add
    
class DecBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_skip_list):
        x_skips = []
        for i in range(len(x_skip_list)):
            x_skip = torch.cat([x_skip_list[i]] * pow(2, i), dim=1)
            x_skip = F.max_pool2d(x_skip, kernel_size=pow(2, i), stride=pow(2, i), padding=0)
            x_skips.append(x_skip)
        x_skip = x_skips[0]
        x_skips_cat = x_skip
        for i in range(1, len(x_skips)):
            x_skips_cat = torch.add(x_skips_cat, x_skips[i])
        x_skips_cat = self.relu(x_skips_cat)
        x = self.tconv1(x)
        x_concat = torch.cat((x, x_skips_cat), dim=1)
        x1 = self.relu(self.conv1_bn(self.conv1(x_concat)))
        x2 = self.relu(self.conv2_bn(self.conv2(x1)))
        x3 = self.relu(self.conv3_bn(self.conv3(x2)))
        x_add = x3 + x_skip + x
        x_add = self.relu(x_add)

        return x_add
    
class DecConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_skip_list):
        x_skips = []
        for i in range(len(x_skip_list)):
            x_skip = torch.cat([x_skip_list[i]] * pow(2, i), dim=1)
            x_skip = F.max_pool2d(x_skip, kernel_size=pow(2, i), stride=pow(2, i), padding=0)
            x_skips.append(x_skip)
        x_skip = x_skips[0]
        x_skips_cat = x_skip
        for i in range(1, len(x_skips)):
            x_skips_cat = torch.add(x_skips_cat, x_skips[i])
        x_skips_cat = self.relu(x_skips_cat)
        x = self.tconv1(x)
        x_concat = torch.cat((x, x_skips_cat), dim=1)
        x1 = self.relu(self.conv1_bn(self.conv1(x_concat)))
        x2 = self.relu(self.conv2_bn(self.conv2(x1)))
        x_add = x2 + x_skip + x
        x_add = self.relu(x_add)

        return x_add 

class CloudNet(nn.Module):
    def __init__(self, in_channels, out_channels, with_logits=True):
        super().__init__()
        self.with_logits = with_logits

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.enc1 = EncConvBlock(16, 32)
        self.enc2 = EncConvBlock(32, 64)
        self.enc3 = EncConvBlock(64, 128)
        self.enc4 = EncConvBlock(128, 256)
        self.enc_bottleneck = EncBottleneck(256, 512)
        self.enc_bridge = EncConvBlock(512, 1024)

        self.dec_bottleneck = DecBottleneck(1024, 512)
        self.dec1 = DecConvBlock(512, 256)
        self.dec2 = DecConvBlock(256, 128)
        self.dec3 = DecConvBlock(128, 64)
        self.dec4 = DecConvBlock(64, 32)

        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        conv_x1 = self.enc1(x)
        x1 = self.maxpool(conv_x1)
        conv_x2 = self.enc2(x1)
        x2 = self.maxpool(conv_x2)
        conv_x3 = self.enc3(x2)
        x3 = self.maxpool(conv_x3)
        conv_x4 = self.enc4(x3)
        x4 = self.maxpool(conv_x4)

        conv_x_bottleneck = self.enc_bottleneck(x4)
        x_bottleneck = self.maxpool(conv_x_bottleneck)
        x_bridge = self.enc_bridge(x_bottleneck)

        x_dec_bottleneck = self.dec_bottleneck(x_bridge, [conv_x_bottleneck, conv_x4, conv_x3, conv_x2, conv_x1])
        x_dec1 = self.dec1(x_dec_bottleneck, [conv_x4, conv_x3, conv_x2, conv_x1])
        x_dec2 = self.dec2(x_dec1, [conv_x3, conv_x2, conv_x1])
        x_dec3 = self.dec3(x_dec2, [conv_x2, conv_x1])
        x_dec4 = self.dec4(x_dec3, [conv_x1])

        out = self.conv2(x_dec4)

        if not self.with_logits:
            out = self.sigmoid(out)

        return out