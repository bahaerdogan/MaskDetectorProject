# README

## Project Overview
This project focuses on implementing and comparing different deep learning architectures to address the problem of mask detection in images. The primary models used include YOLOv11 (You Only Look Once version 11), a Convolutional Neural Network (CNN), and a Vision Transformer (ViT). The goal is to achieve high accuracy, mitigate overfitting, and balance precision-recall metrics across different mask-related categories ("with mask," "without mask," "mask worn incorrectly").

---
##To run and try models:
Yolo model training colab link:https://colab.research.google.com/drive/1S--p8ydTdBMglIIuktQWmf5y4zLS8TQA?usp=sharing

CNN model training colab link:https://colab.research.google.com/drive/16e6cV7zKVKnd6P-xtqUOnSOwriCUfImj?usp=sharing

VIT model training colab link:https://colab.research.google.com/drive/17Y0CYewPHpkggleafXGXju1AhgsLsWEU?usp=sharing

## Models and Architectures

### YOLOv11
#### Initial Training
- **Dataset**: 784 training images, 59 validation images (7% validation split)
- **Architecture**: YOLOv11s
- **Epochs**: 80

##### Issues:
- **Overfitting**: Significant divergence between training and validation loss curves
- **Poor Performance**: Model failed to correctly detect mask categories

#### Updated Training
- **Enhanced Dataset**: 2,027 training images, 196 validation images
- **Architecture**: YOLOv11m
- **Epochs**: 80

##### Improvements:
- **Confusion Matrix**: Reduced misclassifications, better category separation
- **F1 Score**: Consistent improvements across confidence thresholds
- **Validation Accuracy**: Significant increase

### CNN
#### Configuration
- **Architecture**: Standard CNN with convolutional, pooling, and fully connected layers
- **Loss Function**: Cross-entropy loss for classification, mean squared error for bounding boxes
- **Optimizer**: Adam
- **Epochs**: 80
- **Batch Size**: 32

##### Performance
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~96%
- **mAP**: 68.3%

##### Observations
- **Generalization**: Minimal overfitting, good generalization
- **Challenges**: Lower precision for small or overlapping objects

### Vision Transformer (ViT)
#### Configuration
- **Architecture**: Vision Transformer (ViT)
- **Epochs**: 80
- **Optimizer**: AdamW
- **Learning Rate Schedule**: Cosine annealing with warm restarts
- **Batch Size**: 32
- **Loss Function**: Cross-entropy loss

##### Performance
- **Confusion Matrix**: Improved precision and recall across mask-related categories
- **Precision/Recall/F1**:
  - With Mask: Precision 91.2%, Recall 89.8%, F1 Score 90.5%
  - Without Mask: Precision 88.6%, Recall 87.4%, F1 Score 88.0%
  - Mask Worn Incorrectly: Precision 84.7%, Recall 86.1%, F1 Score 85.4%

##### Observations
- **Generalization**: Effective global feature learning
- **Challenges**: Slightly lower performance on overlapping objects

---

## Data Sources
1. [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
2. [Labeled Mask Dataset (Pascal VOC format)](https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-pascal-voc-format/data)

---




