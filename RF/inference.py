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
import sys

sys.path.insert(1, '..')
from rle_encoder_decoder import rle_encode, rle_decode

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
    # TODO: Fix error (maybe by downgrading joblib on kaggle and trying again)
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