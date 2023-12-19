# -*- coding: utf-8 -*-
"""mask_generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FcEQVg4kqO3MO39m_R9_k-35yv42s0r2
"""

import zipfile
from PIL import Image, ImageDraw
import numpy as np
import os
import tempfile

def extract_npz_files(zip_file_path, npz_key, image_size, data_key):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            npz_files = [file for file in zip_ref.namelist() if file.endswith('.npz')]
            print(f'Number of .npz files in the zip: {len(npz_files)}')
            masks = []
            for npz_file in npz_files:
                file_path = os.path.join(temp_dir, npz_file)
                masks += create_masks_from_npz(file_path, image_size, data_key)
        return masks

def create_masks_from_npz(file_path, image_size, data_key):
    data = np.load(file_path, allow_pickle=True)
    masks = []
    boxes_data = data[data_key]

    for frame_boxes in boxes_data:
        mask = Image.new('1', image_size, 0)
        draw = ImageDraw.Draw(mask)

        for box in frame_boxes:
            if np.any(box[:4]) and len(box[:4]) == 4:
                box_coords = [int(coord) for coord in box[:4]]
                draw.rectangle(box_coords, fill=1)

        masks.append(mask)

    return masks

image_size = (1280, 720)

# For YOLO generated boxes in a zip file
yolo_zip_file_path = 'yolonegative.zip'
yolo_data_key = 'det.npy'
#yolo_masks = extract_npz_files(yolo_zip_file_path, yolo_data_key, image_size, yolo_data_key)

# For MobileNet generated boxes in a zip file
mobilenet_zip_file_path = 'mobilepositive.zip'
mobilenet_data_key = 'arr_0.npy'
mobilenet_masks = extract_npz_files(mobilenet_zip_file_path, mobilenet_data_key, image_size, mobilenet_data_key)

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the CustomVGG16 model
class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        original_vgg16 = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(original_vgg16.children())[:-1])  # Retain most VGG16 layers

        # Additional layers to get to 20 channels and 64x64 spatial dimensions
        self.custom_layers = nn.Sequential(
            nn.Conv2d(512, 20, kernel_size=1),  # Reduce channels to 20
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)  # Resize to 64x64
        )

    def forward(self, x):
        x = self.features(x)
        x = self.custom_layers(x)
        return x

# Initialize the custom VGG16 model
custom_vgg16 = CustomVGG16()
custom_vgg16.eval()

# Define a function to preprocess the images
def preprocess_image(pil_img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_img).unsqueeze(0)  # Add a batch dimension

# Function to extract features using CustomVGG16
def extract_features(model, image_list, frames_per_video=50):
    video_features = []

    # Process the list in chunks of `frames_per_video`
    for i in range(0, len(image_list), frames_per_video):
        frame_features = []
        for img in image_list[i:i + frames_per_video]:
            # Preprocess the image and extract features
            input_tensor = preprocess_image(img)
            with torch.no_grad():
                result = model(input_tensor)
                frame_features.append(result.cpu().numpy().squeeze(0))  # Remove batch dimension

        # Stack the features along a new dimension
        video_features.append(np.stack(frame_features, axis=0))

    return video_features

yolo_video_features = extract_features(custom_vgg16, yolo_masks)

# Check the shape of the first video's feature tensor
print(yolo_video_features[0].shape if yolo_video_features else "Empty list")

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
import numpy as np
import zipfile
import os
import tempfile
import shutil

drive.mount('/content/drive')

def save_video_features_to_npz(video_features, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for idx, features in enumerate(video_features):
        npz_file_path = os.path.join(directory, f'video_{idx}.npz')
        np.savez(npz_file_path, features=features)

def compress_npz_files_to_zip(directory, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(directory):
            if file.endswith('.npz'):
                file_path = os.path.join(directory, file)
                zipf.write(file_path, arcname=file)

temp_dir = tempfile.mkdtemp()
save_video_features_to_npz(yolo_video_features, temp_dir)
google_drive_zip_path = '/content/drive/My Drive/mobile_video_features_positive.zip'
compress_npz_files_to_zip(temp_dir, google_drive_zip_path)
print(f"Compressed video features saved to Google Drive: {google_drive_zip_path}")
shutil.rmtree(temp_dir)