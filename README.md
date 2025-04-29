# AI-Powered-Cloud-Masking
Cloud masking using AI

# Install dependencies
```
pip install -r requirements.txt
```

# Inference
To run the model:
```
cd test
python inference.py --image_dir ../input_dir
```

## Other models
### Thresholding
```
cd thresholding
python inference.py --image_dir ../input_dir
```
### Random Forest
Currently has compatibility issues, the model won't load. Tested on kaggle only
```
cd RF
python inference.py --image_dir ../input_dir
```
### UNet
```
cd UNet
python inference.py --image_dir ../input_dir
```
### CloudNet
```
cd CloudNet
python inference.py --image_dir ../input_dir
```

# Results
Can be found in `submission.csv`

# Logs
Can be found in `model_logs.txt`

# Training
Kaggle notebooks used to train models included as well: `train.py`

# Profiling
When running `inference.py` of each model, the model is profiled and the result is saved to `model_logs.txt`. The functions used for profiling can be found in `profile_script.py`
