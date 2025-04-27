import pandas as pd
import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt
import csv

import numpy as np
import cv2
import argparse

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

def normalize_rgb(img):
    img = img.astype('float32')
    normalize_band = lambda band: (band - band.min() + 1e-5) / (band.max() - band.min() + 1e-5)
    for i in range(3):
        img[..., i] = normalize_band(img[..., i])
    return img[..., :3]

def main():
    # argparse
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
            image = tiff.imread(image_file)
            ir = image[..., 3]
            if ir.dtype != 'uint8':
                ir = ir.astype('float32')
                ir = (ir - ir.min()) / (ir.max() - ir.min() + 1e-6)
            image = normalize_rgb(image)
            mean_image = np.mean(image, axis=2)
            if mean_image.dtype != 'uint8':
                mean_image = mean_image.astype('float32')
                mean_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min() + 1e-6)
            diff = ir - mean_image
            diff_mask = np.zeros(diff.shape, np.dtype('float32'))
            idx = np.all([ir > 0.4, abs(diff) < 0.3], axis=0)
            diff_mask[idx] = 1.0
            rle_mask = rle_encode(diff_mask)
            writer.writerow({'id': f_id, 'segmentation': rle_mask})

        print("done")

if __name__ == "__main__":
    main()