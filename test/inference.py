import os
import torch
import numpy as np
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import argparse
import sys
import numpy as np
import pandas as pd
from cloudnet import CloudNet

sys.path.insert(1, '../')
from profile_script import profile
from rle_encoder_decoder import rle_encode

def normalize_img(img):
    img = img.astype('float32')
    normalize_band = lambda band: (band - band.min()) / (band.max() - band.min() + 1e-6)
    for i in range(img.shape[2]):
        img[..., i] = normalize_band(img[..., i])
    return img

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

    submission_file = pd.read_csv("sample_submission.csv", dtype={'id': str})
    rle_masks = []
    for image_id in submission_file['id']:
        image_filename = image_id + '.tif'
        image_filepath = os.path.join(image_dir, image_filename)
        image = tifffile.imread(image_filepath)
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
            rle_masks.append(rle_mask)

    submission_file['segmentation'] = rle_masks
    submission_file.to_csv('submission.csv', index=False)
    print("done")

if __name__ == "__main__":
    main()