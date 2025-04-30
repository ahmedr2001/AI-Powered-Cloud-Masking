import numpy as np
import os
import tifffile as tiff
import csv
import argparse

import sys
sys.path.insert(1, '..')
from rle_encoder_decoder import rle_encode

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