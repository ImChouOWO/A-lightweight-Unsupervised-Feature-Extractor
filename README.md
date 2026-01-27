# A-lightweight-Unsupervised-Feature-Extractor
![image](https://github.com/ImChouOWO/A-lightweight-Unsupervised-Feature-Extractor/blob/main/demo/demo.gif)

- [中文 Readme.md](https://github.com/ImChouOWO/A-lightweight-Unsupervised-Feature-Extractor/blob/main/info/README.md)

## Model Architecture
![image](https://github.com/ImChouOWO/A-lightweight-Unsupervised-Feature-Extractor/blob/main/info/workflow.jpg)

## Features

This model is designed for **two-stage object tracking** tasks.  
It uses **YOLO** as the upper-level feature extractor and integrates  
**Kalman Filtering** with the **Hungarian Algorithm** to perform object association.

Under ideal conditions—where tracked objects remain within the camera view—the system can maintain  
**stable tracking for over 2 minutes without ID switches**.

---

## Introduction

This model is trained using **ROI-based feature extraction** from the deep layers of a YOLO backbone network.  
Specifically, the final **SPP-CSPC** layer of the backbone is used as the **only feature source**.

Detected bounding boxes are used to crop corresponding regions from the feature map.  
Each cropped region is then resized to a fixed spatial resolution of **(10, 10)** for both training and inference.

As an **Encoder**, the model receives a tensor of shape: [1, 512, Hf, Wf]


Using **ROI Align**, region-specific features are cropped and resampled into fixed-size ROI features.  
According to the number of detected bounding boxes, the tensor is reshaped into: [N, 512, 10, 10]


where **N** is the number of detected objects in the image.

### Tensor Flow Overview

| Stage | Tensor Shape | Description |
|------|-------------|-------------|
| Backbone Output | `[1, 512, Hf, Wf]` | YOLOv7 SPP-CSPC feature map |
| Detection Boxes | `[N, 4]` | Bounding boxes for each object |
| ROI Align Input | `[1, 512, Hf, Wf] + [N, 5]` | Includes batch index |
| ROI Align Output | `[N, 512, 10, 10]` | Fixed-size ROI features |

> **Note**  
> ROI Align is applied per image, so the batch index is always `0` at this stage.  
> After collecting all ROI features, they are organized into batches by the DataLoader during training.

---

## Evaluation Protocol

To evaluate the model’s ability to recognize the **same objects across different frames**,  
we construct a paired-frame validation dataset with corresponding annotation files.

Each pair must contain **at least one object that appears in both frames**  
to ensure valid metric computation.

### Evaluation Metrics

| Metric | Definition | Range / Direction | Notes |
|--------|------------|------------------|-------|
| **Top-1 Accuracy** | Whether the correct match is ranked first:<br>$$\text{Top1}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\mathbf{1}\big[\arg\max_j P_{ij}=gt_i\big]$$ | $$[0,1],\ \text{higher is better}$$ | Most intuitive indicator of recognition performance |
| **Mean Rank** | Average rank of the correct match:<br>$$\text{MeanRank}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\text{rank}_i$$ | $$[1,N],\ \text{lower is better}$$ | Indicates how early the correct match appears |
| **MRR** | Mean Reciprocal Rank:<br>$$\text{MRR}=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\frac{1}{\text{rank}_i}$$ | $$[0,1],\ \text{higher is better}$$ | Emphasizes top-ranked correctness |
| **Recall@K** | Whether the correct match appears in top-\(K\):<br>$$\text{Recall@}K=\frac{1}{Q_v}\sum_{i=1}^{Q_v}\mathbf{1}[\text{rank}_i\le K]$$ | $$[0,1],\ \text{higher is better}$$ | Measures retrieval coverage within Top-K |

### Experimental Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Top-1 Accuracy | 0.659444 | ~66% of queries correctly matched at rank-1 |
| Mean Rank | 1.720556 | Correct matches typically appear in top 1–2 |
| MRR | 0.796589 | Most correct matches are ranked very high |
| Recall@5 | 0.953704 | 95.4% of correct matches in top-5 |
| Recall@10 | 1.000000 | All correct matches in top-10 |
| Num. Queries | 6.466667 | ~6.5 query objects per sample |
| Num. Valid | 4.266667 | ~4.3 valid matches per sample |

> **Note**  
> `Recall@10` requires a larger validation dataset for stronger statistical reliability.

---

## Training Environment

The model was trained using **two NVIDIA RTX 5000 Ada Generation GPUs**  
on **Ubuntu 24.04.3**.  
Total training time was approximately **8–10 hours**.

| Item | Specification |
|------|---------------|
| Python | 3.12.3 |
| CUDA Toolkit | 12.8 (V12.8.93) |
| GPU | NVIDIA RTX 5000 Ada Generation ×2 |
| Epochs | 500 |
| Batch Size | 256 |
| Multi-GPU | DDP (torchrun) |

> **Note**  
> Inference is performed on a **single GPU**,  
> with an average VRAM usage of **9%–12%**.

---

## Usage

Clone the Repository

```bash
git clone https://github.com/ImChouOWO/A-lightweight-Unsupervised-Feature-Extractor.git
```
Create a Virtual Environment (Optional)
```
python3.12 -m venv venv312_torch
source venv312_torch/bin/activate
```
or with Conda:
```
conda create -n torch312 python=3.12 -y
conda activate torch312
```
Install Dependencies
```
cd A-lightweight-Unsupervised-Feature-Extractor
pip install -r requirements.txt
```
Run Validation
```
python3 val.py
```
Run Training (Multi-GPU)
```
torchrun --nproc_per_node=2 main_train.py
```
Run Tracking
```
python3 tracking.py 
```
> **Note**
> 
> for linux or linux like system


```
python3 tracking_win.py
```
> **Note**
> 
>for windows system

---
> **Note**
>
>This project is released under the MIT License.
>Any use, modification, or redistribution must retain the original author attribution.

