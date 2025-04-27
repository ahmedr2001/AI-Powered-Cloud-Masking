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

import numpy as np

import sys
sys.path.insert(1, '..')
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)        # (B, 64, H, W)
        x2 = self.down2(F.max_pool2d(x1, 2))  # (B, 128, H/2, W/2)
        x3 = self.down3(F.max_pool2d(x2, 2))  # (B, 256, H/4, W/4)
        x4 = self.down4(F.max_pool2d(x3, 2))  # (B, 512, H/8, W/8)
        x5 = self.bottleneck(F.max_pool2d(x4, 2))  # (B, 1024, H/16, W/16)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        return self.final(x)
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=4, out_channels=1)
model.load_state_dict(torch.load("unet.pth"))
num_ops, num_params = profile(model, (1, 4, 512, 512))

model = UNet(in_channels=4, out_channels=1)
model.load_state_dict(torch.load("unet.pth", map_location=device))
model.to(device)
model.eval()

print("CloudNet loaded successfully.")

# save to model_logs.txt
with open('model_logs.txt', 'w') as f:
    f.write(f"Model: UNet\n")
    f.write(f"Input Size: (1, 4, 512, 512)\n")
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