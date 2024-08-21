#script.sh to be run in cronjob
. /var/snap/amazon-ssm-agent/23/venv/bin/activate

BUCKET_NAME="datav"
PREFIX="fasdfsdfasdf/sdfhg"
DOWNLOAD_DIR="/var/snap/amazon-ssm-agent/23/files"




# Check if any file exists in the directory
if [ "$(ls -A $DOWNLOAD_DIR)" ]; then
  echo "Files found in $DOWNLOAD_DIR. They will be deleted after running the script."
  FILE_EXISTS=true
else
  echo "No files found in $DOWNLOAD_DIR."
  FILE_EXISTS=false
fi




#  Copy files from s3
/usr/local/bin/aws s3api list-objects-v2 \
    --bucket $BUCKET_NAME --prefix $PREFIX \
    --query 'Contents | sort_by(@, &LastModified)[-200:].{Key:Key}' \
    --output text > files_to_download.txt

#cat files_to_download.txt
tail -n 1 files_to_download.txt | while read -r line; do
    /usr/local/bin/aws s3 cp "s3://$BUCKET_NAME/$line"  $DOWNLOAD_DIR
done






# Run the Python script
# python3 /var/snap/amazon-ssm-agent/7993/script.py --input_img_size 600 --batch_size 1 --model_path "./trained_models/*" --local_dir  $DOWNLOAD_DIR --bucket_name $BUCKET_NAME --input_s3_prefix $PREFIX
# python3 script.py --model_path ./trained_models/* --image_path "$DOWNLOAD_DIR"
python3 /var/snap/amazon-ssm-agent/3452345/infer.py --input_img_size 600 --model_path /var/snap/amazon-ssm-agent/trained_models/* --image_path  $DOWNLOAD_DIR


#append to csv
cat /var/snap/amazon-ssm-agent/14234132/inference_predictions.csv >> /var/snap/amazon-ssm-agent/168273643/testpredictions/predictions.csv


# copy predictions to s3
/usr/local/bin/aws s3 cp /var/snap/amazon-ssm-agent/1451435/testpredictions/predictions.csv s3://rise-collab/


# Check the exit status of your main script
if [ $? -eq 0 ]; then
  echo "Main script executed successfully."
  # If files were found before running the main script, delete them
  if [ "$FILE_EXISTS" = true ]; then
    echo "Deleting files in $DOWNLOAD_DIR."
    #rm -f $DOWNLOAD_DIR/*
    echo "Files deleted."
  fi
else
  echo "Main script encountered an error."
  exit 1
fi

# Optionally, you can include additional cleanup or logging here

echo "Script execution completed."




#infer.py
import argparse
import pickle
from statistics import median

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sortedcontainers import SortedList

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import timm

from Datasets import Dataset
from EfficientNet_model import EfficientNet
from resamplers import split_dict_uniformly
import os


# Argparsing
parser = argparse.ArgumentParser(description='Train a model.')
parser.add_argument('--input_img_size', type=int, default=800,  help='The input image size.')
parser.add_argument('--batch_size',     type=int, default=1,    help='The batch size of input images.')
parser.add_argument('--model_path',     type=str, default=None, help='The path to a saved model to use.')
parser.add_argument('--image_path',     type=str, default=None, help='The path to an image or folder of images')
args = parser.parse_args()
input_img_size = args.input_img_size
batch_size = args.batch_size
model_path = args.model_path
image_path = args.image_path

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EfficientNet(device, model_path, param_freeze_ratio=0.66)

# Check if image_path is a directory or a single image
if os.path.isdir(image_path):
    # Get all image files in the directory
    image_files = []
    for file in os.listdir(image_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_files.append(os.path.join(image_path, file))
else:
    # Use the single image
    image_files = [image_path]

dataset_params = {
    'model_config': timm.data.resolve_model_data_config(model),
    'input_img_size': input_img_size,
    # 'ROI': ((1046,569),(1920,1440))  # RayRobertsLake
    'ROI': ((1067,417),(2393,1743))  # Pinewood
}
inference_dataset = Dataset(image_files, **dataset_params, scaler=model.scaler, training=False)
dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model.eval()
predictions = []
filepaths = []
with torch.no_grad():
    for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.float().to(device)
        
        # Perform inference
        outputs = model(images).clone().detach().cpu()
        outputs = inference_dataset.reverse_scale(outputs)
        
        # Store predictions and filepaths
        predictions.extend(outputs)
        filepaths.extend(inference_dataset.image_paths[i*batch_size:(i+1)*batch_size])

# Save the predictions and filepaths as a DataFrame
df = pd.DataFrame({'Filepath': filepaths, 'Prediction': predictions})

# Save the DataFrame as a CSV file
df.to_csv('inference_predictions.csv', index=False)


#Datasets.py

import numpy as np

from sklearn.discriminant_analysis import StandardScaler
import torch
import timm
from torchvision.io import read_image
from torchvision.transforms import Compose, ToPILImage, ColorJitter, RandomPerspective, Resize, ToTensor, Normalize


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mappings, model_config, input_img_size, ROI=None, scaler=None, training=True):
        self.mappings = mappings
        self.inference = not isinstance(mappings, dict)
        self.ROI = ROI
        self.training = training
        if self.inference:
            self.image_paths = mappings
            self.training = False
        else:
            self.image_paths = list(mappings.keys())
        
        model_config['input_size'] = (3, input_img_size, input_img_size)
        
        transforms_list = [Resize((input_img_size,input_img_size))]
        if training:
            transforms_list.extend([ColorJitter(brightness=(0.9, 1.2), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0),
                                    RandomPerspective(distortion_scale=0.1)])        
        transforms_list.extend([ToTensor(), 
                                Normalize(mean=model_config['mean'], std=model_config['std'])])
        # transforms_list.extend([ToTensor()])  # For test printing
        self.model_transforms = Compose(transforms_list)
        
        self.to_pil = ToPILImage()
        
        if not self.inference:
            targets = [mappings[path] for path in self.image_paths]
            if not scaler:
                self.scaler = StandardScaler()
                self.targets_scaled = self.scaler.fit_transform(np.array(targets).reshape(-1, 1)).flatten()
            else:
                self.scaler=scaler
                self.targets_scaled = self.scaler.transform(np.array(targets).reshape(-1, 1)).flatten()
        else:            
            self.scaler=scaler

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        if self.inference:
            target_scaled = 0.0
        else:
            target_scaled = self.targets_scaled[idx]
        
        image = read_image(image_path)
        image = self.to_pil(image)        
        if self.ROI:
            image = image.crop((self.ROI[0][0], self.ROI[0][1], self.ROI[1][0], self.ROI[1][1]))            
        image = self.model_transforms(image)        
        return image, target_scaled
    
    def reverse_scale(self, iterable):
        if torch.is_tensor(iterable):
            iterable_np = iterable.numpy()
        else:
            iterable_np = np.array(iterable)
        
        return self.scaler.inverse_transform(iterable_np.reshape(-1, 1)).flatten()



#EfficientNet.py

import os

import torch
import torch.nn as nn
import timm


class EfficientNet(nn.Module):
    class RegressionLayers(nn.Module):
        def __init__(self, in_features):
            super(EfficientNet.RegressionLayers, self).__init__()
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )

        def forward(self, x, pre_logits=None):
            return self.layers(x)
    
    def __init__(self, device, resume_from=None, param_freeze_ratio=0.5):
        super(EfficientNet, self).__init__()
        
        self.loss_history = []
        self.scaler = None
        
        self.model = timm.create_model('tf_efficientnet_l2.ns_jft_in1k', pretrained=True)
        self.model.reset_classifier(0)

        # Freeze a percentage of the model for training
        total_params = sum(p.numel() for p in self.model.parameters())
        params_to_freeze = int(total_params * param_freeze_ratio)        
        frozen_params_count = 0
        for param in self.model.parameters():
            if frozen_params_count >= params_to_freeze:
                break
            param.requires_grad = False
            frozen_params_count += param.numel()

        # Replace the classification head with dense net layers, shape calculated on the fly via dummy input
        n_features = self.model.forward_features(torch.randn(1, 3, 224, 224)) 
        self.model.classifier = EfficientNet.RegressionLayers(n_features.shape[1])

        self.model = self.model.to(device)
        
        # If a saved model was provided, load it
        if resume_from and os.path.isfile(resume_from):
            loaded_model = torch.load(resume_from)
            self.load_state_dict(loaded_model.state_dict())
            self.scaler = loaded_model.scaler
            print(f"Loaded model from {resume_from}")
    
    def forward(self, x, pre_logits=None):
            return self.model.forward(x)









## Error getting when runnning the shell script in crobtab

tail -f /var/snap/amazon-ssm-agent/23/cron.log
  File "/var/snap/amazon-ssm-agent/23/infer.py", line 70, in <module>
    outputs = inference_dataset.reverse_scale(outputs)
  File "/var/snap/amazon-ssm-agent/23/Datasets.py", line 70, in reverse_scale
    return self.scaler.inverse_transform(iterable_np.reshape(-1, 1)).flatten()
AttributeError: 'NoneType' object has no attribute 'inverse_transform'
upload: ../var/snap/amazon-ssm-agent/23/testpredictions/predictions.csv to s3://data/predictions.csv
Main script executed successfully.
Deleting files in /var/snap/amazon-ssm-agent/23/files.
Files deleted.
Script execution completed.













