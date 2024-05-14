# Real-Time Helmet Detection using YOLO

## Overview

This project aims to train a YOLO (You Only Look Once) model on a dataset to detect helmets in frames. The dataset consists of extracted frames from a video and corresponding bounding box annotations for each object in each frame, provided in a CSV file. The data has been formatted into YOLO format, with one annotation file for each image.

## Dataset Preparation

1. **Video to Frames Extraction**: The video was processed to extract individual frames, which form our dataset.

2. **Labels Generation**: Using the provided CSV file with bounding box coordinates for each object in each frame, we generated labels for each frame indicating the presence of a helmet and its location within the frame.

3. **YOLO Format Conversion**: The data was then converted into YOLO format, where each image is accompanied by an annotation file specifying the object class (helmet) and its bounding box coordinates.

## Training Process

1. **YOLO Model Selection**: We chose a YOLO model suitable for real-time object detection, balancing accuracy and inference speed.

2. **Data Augmentation**: To enhance model generalization, we applied data augmentation techniques such as random scaling, rotation, and flipping.

3. **Model Training**: The YOLO model was trained using the prepared dataset and labels to learn the features necessary for helmet detection.

## Evaluation

1. **Validation Set**: A portion of the dataset was reserved for validation to evaluate the model's performance during training.

2. **Metrics**: We measured the model's performance using metrics such as mean average precision (mAP) and intersection over union (IoU).

## Deployment

1. **Real-Time Inference**: Once trained, the YOLO model can be deployed for real-time helmet detection, capable of processing frames at high speed.

2. **Integration**: The model can be integrated into applications or systems requiring helmet detection, enhancing safety measures.

## Future Improvements

1. **Fine-Tuning**: Continual training with additional annotated data can improve the model's accuracy and robustness.

2. **Optimization**: Optimizing the model architecture and hyperparameters can further enhance real-time performance without compromising accuracy.

3. **Multi-Object Detection**: Extending the model to detect multiple objects simultaneously, such as helmets and safety gear, can provide comprehensive safety solutions.

## Conclusion

This project demonstrates the process of training a YOLO model for real-time helmet detection, from data preparation to model deployment. By leveraging YOLO's efficiency and accuracy, we aim to enhance safety measures in various applications.
