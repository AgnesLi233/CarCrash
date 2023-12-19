# -*- coding: utf-8 -*-
"""YOLOv8boundingbox.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16BY9JdZ9BVyLvrf13Y_xo7ml2rMUOBXi
"""

!pip install ultralytics

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import itertools
from ultralytics import YOLO
from google.colab import drive
drive.mount('/content/drive')

# Choose proper pretrained model from yolo v8: small size model for detection task
# Change model kind to nano for faster prediction and medium for more accurate prediction
# Change model type from detection to segmentaftion for more accurate background reduction
model = YOLO("yolov8m.pt")

os.chdir('/content/drive/MyDrive/CarCrashYOLO/images/test')

# Get the list of JPG files in the directory
jpg_files = os.listdir('.')
jpg_files = [file for file in jpg_files if file.lower().endswith('.jpg')]

def process_and_save_bounding_boxes_npz(output_directory, frames_per_video=50):

    # Group images by video ID
    grouped_images = {}
    for file in jpg_files:
        video_id = file.split('_')[0]
        grouped_images.setdefault(video_id, []).append(file)

    print(len(grouped_images))
    grouped_images = dict(itertools.islice(grouped_images.items(), 250, 899))

    # Process each group of images
    for video_id, image_files in tqdm(grouped_images.items()):

        # Initialize an array to hold bounding boxes for all frames in the video
        all_bboxes = np.zeros((frames_per_video, 19, 6), dtype=np.float32)

        for i, image_file in enumerate(image_files[:frames_per_video]):
            # Run inference on the image
            results = model(image_file, iou=0.96)

            # Get bounding box data
            bounding_boxes = results[0].boxes.data

            # Convert bounding box data to the required format
            for j, bbox in enumerate(bounding_boxes[:19]):
                all_bboxes[i, j] = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]]

        # Save bounding box data and video ID to an npz file
        output_file = os.path.join(output_directory, f"{video_id}.npz")
        np.savez(output_file, det=all_bboxes, id=video_id)

# generate for test data
process_and_save_bounding_boxes_npz("/content/drive/MyDrive/CarCrashYOLO/boundingbox/test", frames_per_video=50)