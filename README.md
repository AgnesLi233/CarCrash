# Car Crash Accident Prediction


## Content

1. [Project Description](#Project-Description)
2. [Dataset](#Dataset)
3. [Project Framework](#Project-Framework)
4. [Code Structure](#Code-Structure)
5. [Results and Observations](#Results-and-Observations)
6. [References](#References)

## Project Description 
![YOLOv8](https://github.com/AgnesLi233/CarCrash/assets/104687808/2587c947-b4f9-484a-9846-fb72ccc2e907)

The increasing frequency and severity of traffic accidents have raised significant concerns for public safety and prompted the need for advanced accident detection systems. In response to this pressing issue, this project presents a novel approach for traffic accident detection, aimed at improving road safety and reducing associated human and economic costs. Traditional LiDAR systems are undoubtedly effective in detecting objects; however, they fall short in object identification capabilities. This limitation hinders the decision-making process of vehicles, occasionally leading to impractical or unsafe responses. In the paper "Traffic Accident Detection Using Background Subtraction and CNN Encoder–Transformer Decoder in Video Frames," a method for detecting traffic accidents in videos is proposed, involving background subtraction, CNN encoder (YOLOv5), and a Transformer decoder. To enhance this method, the following improvements were made: upgrading the CNN Encoder to YOLOv8, or MobileNet-SSD v3, Consider employing ConvLSTM for crash prediction. This research contributes to the ongoing efforts to enhance road safety and emergency response, offering a promising solution for timely accident detection and response to mitigate impacts of accidents on roadways.


## Dataset

The Car Crush Dataset (CCD) in the paper “Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning” will be used for the project. CCD stands as a suitable choice for pursuing our interest in improving existing methods for real-time traffic accident detection within vehicle settings. The annotations, encompassing diverse attributes, allow for contextual information integration, potentially enhancing the model's capacity to recognize and predict accidents across various conditions. Most importantly, we deliberately use the same dataset as the original paper but with improvements on models and preprocessing methods in order to make the resulting metrics comparable. 

CCD contains real traffic accident videos captured by dashcamin mp4 format collected from 2 sources: 
* 1,500 trimmed video collected on Youtube, each containing 50 frames
* 3,000 normal videos randomly sampled from the BDD100K dataset

Annotations of 1,500 accident videos are saved in txt files. Each line contains the following attributes:
```
- vidname: Video name, i.e., 000018
- binlabels(targets): Binary label of frames in video, where 1 indicates accident frame
- startframe：For YouTube normal video, this is the zero-padded starting frame of each video
- youtubeID: Numeric YouTube video identifier
- timing: Timing of the day, i.e., day or night
- weather: Weather conditions of the time, i.e., Normal, Snowy, and Rainy.
- egoinvolve: Boolean identifier to indicate wheather the ego-vehicle is involved in the accident
```

The author also includes feature files in npz format containing the following parts:
```
- data: VGG-16 features of all frames in the video
- det: Detected bounding boxes of all frames, where det of one frame is denoted by (x1, y1, x2, y2, prob, cls)
- labels: One-hot video labels to indicate whether the video contains an accident
- ID: Video name and unique identification
```

Following is the file structure of CCD folder provided by Bao et al.:
```
CarCrash
├── codes                    # useful codes for analyzing the dataset
├── vgg16_features
│   ├── positive             # feature files of possitive (accident) videos
│   │   ├── 000001.npz
│   │   ├── ...
│   │   └── 001500.npz
│   ├── negative             # feature files of negative (normal) videos
│   │   ├── 000001.npz
│   │   ├── ...
│   │   └── 003000.npz
│   ├── train.txt            # list file of training split 
│   └── test.txt             # list file of testing split 
├── videos
│   ├── Normal               # normal driving videos
│   │   ├── 000001.mp4
│   │   ├── ...
│   │   └── 003000.mp4
│   ├── Crash-1500           # crash accident videos
│   │   ├── 000001.mp4
│   │   ├── ...
│   │   └── 001500.mp4
│   └── Crash-1500.txt       # annotation file for crash accident
└── README.md
```

All files in the CCD dataset was saved in a Google Drive folder provided by Bao et al. in the original [Cogito2012/CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset#overview) GitHub repository. Download CCD from [Google Drive](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F).



## Project Framework

### Updated Bounding Box Masks Extractor
* MobileNet-SSD v3
  * Model detail: The project has transitioned from YOLOv5 to MobileNet-SSD v3 for object detection due to its suitability for mobile deployment. MobileNet-SSD v3, a single-shot detector, is optimized for mobile devices, offering real-time processing capabilities for dashcam video feeds directly on smartphones. This marks an improvement over YOLOv5’s resource-heavy design. 
  * Architecture: The architecture of MobileNet-SSD v3 features depth-wise separable convolutions, re- ducing parameter count. It utilizes bottleneck layers and squeeze-and-excitation blocks for feature enhancement, and a convolutional SSD head for multi-scale object detection. ReLU6 activations aid in hardware optimization, and the model employs Smooth L1 and cross-entropy loss functions for bounding box regression and classification, respectively. 
  * Re-training: the MobileNet is pre-trained on the COCO dataset. To improve the model’s performance on traffic conditions, we re-train the model on Open Images dataset with ‘car’ and ‘truck’ classes to.
* YOLOv8
  * Model detail: The project has also upgraded object detection model from YOLOv5 to YOLOv8 for better performance. The Ultralytics YOLOv8 is used in this project. YOLOv8 is a state-of-the-art model that builds upon previous versions of YOLO and introduces new features and improvements to further boost performance and flexibility. YOLOv8 contains model for the following tasks: detection and tracking, segmentation, classification and pose. The library contains 5 different sizes of pretrained models: nano, small, medium, large, huge. Greater model size generally requires longer training/predicting time.
  * Preprocess: YOLOv8 accepts images. Videos are preprocessed using the approach described in the original GitHub repository. Each video was split equally into 50 individual frames. 
  * Implementation: Medium size YOLOv8 model for segmentation was used to generate bounding boxes.
### Mask Generator & Updated Decoder
* Mask Generator
  * The Mask generator takes the coordinates from YOLOv8 and MobileNet-SSD v3 as input, and generates masks the is white and black. The white parts is the region of the bounded by the boxes, while the black part is the removed background.
* Updated Decoder
  * The updated decoder takes 1280*720p masks and extracts frame-based features to feed into ConvLSTM2D.
### Sliding Window
* The create_shifting_window function segments videos into overlapping windows by spliting the feature frames, creating samples that provide temporal context for the model. 

### ConvLSTM2D
* Our methodology for predicting car accidents from video sequences involves preprocessing data using a sliding window technique and employing a ConvLSTM-based deep learning model, designed to handle the spatial-temporal complexities of video data. The ConvLSTM2D layers are key, merging spatial and temporal data analysis, a capability beyond that of traditional LSTMs, by integrating convolutional structures within LSTM units. This approach allows for capturing intricate patterns in both individual frames and across sequences, essential for anticipating accidents.


## Code Structure
```
CarCrash
├── data
     └─────── openimagedownloader.py (Downloading data from Open Images Dataset)
├── MobileNet-SSDv3
     ├─────── train_mobilenet.py (Re-training the MobileNet v3 using Open Images)
     └─────── object_detection_mobilenetv3.py (Output the detected bounding boxes and save as npc files)
├── YOLOv8
     ├─────── framesplit.py (Split each video for each frame)
     ├─────── yolov8boundingbox.py (Generate bounding box using YOLOv8)
     └─────── yolov8example.py (Plot bounding box example)
└── MobileNet-SSDv3
     ├─────── frozen_inference_graph.pb
     ├─────── labels.txt
     ├─────── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
     ├─────── train_mobilenet.py
     └─────── object_detection_mobilenetv3.py

```
For framesplit.py, frame images is saved in this [Google Drive](https://drive.google.com/drive/folders/1eDTVUQTuhTwwRhVpTbu9ZPKVHOXOhpXL?usp=share_link).

## Results and Observations

### Model Performance
* MobileNet-SSD v3: ![download](https://github.com/AgnesLi233/CarCrash/assets/104687808/7d246724-de67-4677-8755-952517721220)

* YOLOv8:
* 
<div align="center">
    <img width="726" alt="yolov8cmp" src="https://github.com/AgnesLi233/CarCrash/assets/104687808/3ed05970-cb6b-416a-8663-66862e742902">
</div>

As we can observed in this figure, small model size shows relatively high accuracy for bounding box prediction compared to bounding boxes given by the dataset. But sometimes larger models might cause overfitting. For example, a backpack at the corner could be recognized as a traffic light for large model but not small model.

* ConvLSTM2D, Sliding Window:![image](https://github.com/AgnesLi233/CarCrash/assets/104687808/423e7e81-4005-459e-8082-ec7821ecf324)

### Bad Cases
![download (1)](https://github.com/AgnesLi233/CarCrash/assets/104687808/241daba3-3af3-4d01-a09b-04524e8db899)


### Insights
*For MobileNet v3, we observed a poor performance under rainy and dark environments. For future work, increase instances of such conditions in the training process to improve its capability of successful b-box extraction.
*For YOLOv8, we observed a relatively stable and accurate performance but slow training time. For future work, improvement could be made by reducing model size to reduce training time while finetune with more data to maintain high accuracy.
*Train the ConvLSTM2D with extracted features on YOLOV8 and MobileNet v3
*With better hardware, train on bigger datasets, with more dynamic sliding window.

## References
Datasets
* CarCrashDataset(CCD): https://github.com/Cogito2012/CarCrashDataset#overview

Models
* MobileNet-SSD v3: https://docs.openvino.ai/2023.2/omz_models_model_mobilenet_ssd.html
* YOLOv8: https://docs.ultralytics.com
* ConvLSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D

Paper
* Zhang, Y.; Sung, Y. Traffic Accident Detection Using Background Subtraction and CNN Encoder–Transformer Decoder in Video Frames. Mathematics 2023, 11, 2884. https://doi.org/10.3390/math11132884
* Haleh Damirchi, Michael Greenspan, Ali Etemad, "Context-Aware Pedestrian Trajectory Prediction with Multimodal Transformer", 2023 IEEE International Conference on Image Processing (ICIP), pp.2535-2539, 2023.  https://ieeexplore.ieee.org/abstract/document/8967556
