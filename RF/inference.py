import os
import cv2
import tifffile
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import joblib
import csv
import numpy as np
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


class CloudDataset(Dataset):
    def __init__(self, image_files, transforms=None):
        self.image_files = image_files
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        image = tifffile.imread(img_path) 

        if image.shape[0] == 4 and image.shape[-1] != 4:
            image = np.transpose(image, (1,2,0))  # (H, W, C) 

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image

def get_transforms():
    return A.Compose([
        A.Resize(128, 128, interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST), 
        ToTensorV2()
    ])

def calculate_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def local_variance(image, size=3):
    def var_func(x):
        return np.var(x)
    return generic_filter(image, var_func, size=(size, size))

def extract_features(image):
    C, H, W = image.shape
    features = []

    for c in range(C):
        features.append(image[c])

    nir = image[3] 
    red = image[0] 
    ndvi = calculate_ndvi(nir, red)
    features.append(ndvi)

    for c in range(C):
        var_map = local_variance(image[c], size=3)
        features.append(var_map)

    feature_map = np.stack(features, axis=-1) 

    return feature_map

def prepare_data_rf(dataloader):
    X = []
    for images in dataloader:
        images = images.numpy() 
        for i in range(images.shape[0]):
            feature_map = extract_features(images[i])
            X.append(feature_map.reshape(-1, feature_map.shape[-1]))
    X = np.concatenate(X, axis=0)
    return X

def visualize_predictions(y_pred, h, w):
    start = 0
    end = (h*w)
    pred_mask = y_pred[start:end].reshape(h, w)

    plt.figure(figsize=(10,5))

    plt.subplot(1,1,1)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')

    plt.show()

def main():
    # argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--image_dir', type=str, default='/kaggle/input/cloud-masking-dataset/content/train/data', help='Directory of images')

    image_dir = parser.parse_args().image_dir

    # Error: incompatible with the original pickle file
    model = joblib.load('rf.pkl')
    with open('submission.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'segmentation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for f in sorted(os.listdir(image_dir)):
            image_file = os.path.join(image_dir, f)
            f_id = f.split('.')[0]
            dataset = CloudDataset(image_files=[image_file], transforms=get_transforms())
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
            X = prepare_data_rf(dataloader)
            y_pred = model.predict(X)
            y_pred = y_pred.reshape(-1, 128, 128)
            y_pred = np.array([cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) for mask in y_pred])
            y_pred = y_pred.reshape(-1, 256, 256)
    
            visualize_predictions(y_pred, 256, 256)
    
            for j in range(y_pred.shape[0]):
                rle_mask = rle_encode(y_pred[j])
                writer.writerow({'id': f_id, 'segmentation': rle_mask})
    
        print("done")

if __name__ == '__main__':
    main()