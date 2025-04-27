import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile
import torchvision.transforms as T
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import csv
import argparse
import sys
import numpy as np

sys.path.insert(1, '../')
from profile_script import profile

def normalize_img(img):
    img = img.astype('float32')
    normalize_band = lambda band: (band - band.min()) / (band.max() - band.min() + 1e-6)
    for i in range(img.shape[2]):
        img[..., i] = normalize_band(img[..., i])
    return img

def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    
    Returns:
        str: RLE-encoded string.
    """
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format

def rle_decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """Decodes an RLE-encoded string into a binary mask with validation checks."""
    
    if not isinstance(mask_rle, str) or not mask_rle.strip() or mask_rle.lower() == 'nan':
        # Return all-zero mask if RLE is empty, invalid, or NaN
        return np.zeros(shape, dtype=np.uint8)
    
    try:
        s = list(map(int, mask_rle.split()))
    except:
        raise Exception("RLE segmentation must be a string and containing only integers")
    
    if len(s) % 2 != 0:
        raise Exception("RLE segmentation must have even-length (start, length) pairs")
    
    if any(x < 0 for x in s):
        raise Exception("RLE segmentation must not contain negative values")
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    starts, lengths = s[0::2], s[1::2]
    
    for start, length in zip(starts, lengths):
        if start >= mask.size or start + length > mask.size:
            raise Exception("RLE indices exceed image size")
        mask[start:start + length] = 1
    
    return mask.reshape(shape, order='F')  # Convert to column-major order

def generate_random_mask(shape, probability=0.5):
    """
    Generates a random binary mask.

    Args:
        shape (tuple): (height, width) of the mask.
        probability (float): Probability of a pixel being 1 (default is 0.5).

    Returns:
        np.ndarray: Random binary mask.
    """
    return (np.random.rand(*shape) < probability).astype(np.uint8)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CloudNet(in_channels=4, out_channels=1)
model.load_state_dict(torch.load("cloudnet.pth"))
num_ops, num_params = profile(model, (1, 4, 192, 192))

model = CloudNet(in_channels=4, out_channels=1)
model.load_state_dict(torch.load("cloudnet.pth", map_location=device))
model.to(device)
model.eval()

print("CloudNet loaded successfully.")
# save to model_logs.txt
with open('model_logs.txt', 'w') as f:
    f.write(f"Model: CloudNet\n")
    f.write(f"Input Size: (1, 4, 192, 192)\n")
    f.write(f"Total Operations: {num_ops.item() / 1e9} GOps\n")
    f.write(f"Total Parameters: {num_params.item() / 1e6} M\n")


# --- Preprocessing transforms ---
transform = A.Compose([
    A.Resize(192, 192, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST),
    ToTensorV2()
])

def main():
    argeparse = argparse.ArgumentParser()
    argeparse.add_argument('--image_dir', type=str, default='/kaggle/input/cloud-masking-dataset/content/train/data')
    image_dir = argeparse.parse_args().image_dir

    print("image_dir: ", image_dir)

    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'segmentation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for f in sorted(os.listdir(image_dir)):
            image_file = os.path.join(image_dir, f)
            f_id = f.split('.')[0]
            image = tifffile.imread(image_file)

            # Preprocess the image
            image = normalize_img(image)
            image = transform(image=image)['image']
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                output = torch.sigmoid(output).cpu().numpy()
                output = (output > 0.5).astype(np.uint8)
                output = output.squeeze(0).squeeze(0)
                output = cv2.resize(output, (256, 256), interpolation=cv2.INTER_NEAREST)
                rle_mask = rle_encode(output)
                writer.writerow({'id': f_id, 'segmentation': rle_mask})


    print("done")


if __name__ == "__main__":
    main()