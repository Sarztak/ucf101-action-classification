# Video Classification with Deep Learning

## Overview

This project explores **deep learning approaches for video classification**, focusing on how different architectural choices capture **spatial and temporal information** in human action recognition. Using a curated subset of the **UCF101 dataset**, we compare sequential and joint spatiotemporal models under realistic computational constraints.

In addition to classification, the project extends learned visual representations to **creative video style transfer**, demonstrating how action understanding models connect to generative applications.

## Motivation

Video data dominates modern digital content, but unlike images, it carries **temporal structure** that is essential for understanding actions and events.

Key challenges addressed:

* Modeling motion and appearance jointly
* Balancing temporal modeling power with computational efficiency
* Understanding trade-offs between sequential and convolutional video models

### Core Applications

* Surveillance and activity monitoring
* Media tagging and highlight detection
* Business and user behavior analysis
* Real-time systems (autonomous driving, AR)

### Why Style Transfer

Style transfer demonstrates how learned visual features can be reused for **creative generation**, including:

* Film and animation pipelines
* Artistic video transformation
* Restoration of legacy media

## Dataset

### UCF101

UCF101 is a benchmark dataset for action recognition containing 101 human action classes across diverse scenarios.

**Dataset characteristics**

* Total actions: 101
* Total clips: 13,320
* Mean clip length: 7.21 seconds
* Frame rate: 25 FPS
* Resolution: 320 × 240

### Subset Used

To remain computationally feasible, this project uses a **9-class subset** selected to preserve:

* Diversity of motion intensity
* Indoor and outdoor scenes
* Lighting and background variation

The reduced scope allows meaningful comparison of model architectures without sacrificing interpretability.

## Exploratory Data Analysis (EDA)

### Temporal Characteristics

* Most clips fall between 2–10 seconds
* Right-skewed duration distribution supports fixed-length frame sampling

### RGB Distribution

Color histograms show strong activity-dependent patterns:

* High saturation and variation in activities like rafting
* Dominant blue/green channels in sky diving

### Optical Flow

Motion magnitude varies significantly across classes:

* High-motion actions (rafting, surfing)
* Low-motion actions (apply makeup, haircut)

This validates the need for temporal modeling beyond static frames.

## Modeling Approaches

### Sequential vs. Joint Spatiotemporal Models

| Aspect            | 2D CNN + LSTM      | (2+1)D Convolution              | CSN (Channel-Separated)            |
| ----------------- | ------------------ | ------------------------------- | ---------------------------------- |
| Temporal modeling | Long-term via LSTM | Short-term via temporal filters | Short-term with channel separation |
| Computation       | Slower, sequential | Faster, parallel                | Efficient, lightweight             |
| Flexibility       | Variable length    | Fixed clip size                 | Fixed clip size                    |
| Interpretability  | Modular            | Less interpretable              | Moderate                           |

Multiple approaches were implemented to understand how design choices impact learning behavior.

## Model Architectures

### 2D CNN + LSTM

* Frame-wise spatial feature extraction using:

  * Custom VGG-style CNN
  * ResNet-style backbone
* Temporal modeling via LSTM
* Global Average Pooling + dense classification head

This architecture separates spatial and temporal learning explicitly.

### (2 + 1)D Convolution

* Factorizes 3D convolution into:

  * 2D spatial convolution (H × W)
  * 1D temporal convolution (T)
* Reduces parameters and FLOPs
* Improves training stability while retaining motion modeling

### Channel-Separated Network (CSN)

* Depthwise 2D spatial convolution per frame
* Channel-wise separation for efficiency
* Fully convolutional spatiotemporal modeling
* Designed for scalability but sensitive to data volume

## Results and Trade-Offs

| Model                  | Test Accuracy | Parameters | Model Size | Training Time |
| ---------------------- | ------------- | ---------- | ---------- | ------------- |
| 2D CNN + LSTM (VGG)    | 90%           | 5.5M       | 21 MB      | 0.5 hrs       |
| 2D CNN + LSTM (ResNet) | 77%           | 0.77M      | 2.95 MB    | 2 hrs         |
| (2+1)D Conv            | 79%           | 0.44M      | 1.69 MB    | 0.5 hrs       |
| CSN                    | 18%           | 0.5M       | 0.15 MB    | 3 hrs         |

Key observations:

* Sequential models capture longer temporal dependencies better
* Lightweight convolutional models trade accuracy for efficiency
* CSN underperformed due to limited data and training complexity

## Style Transfer Extension

### Neural Style Transfer (VGG-based)

* Optimization-based method
* Uses pretrained VGG features
* Combines:

  * Content loss
  * Style loss (Gram matrices)
  * Total variation loss

### CycleGAN

* Unpaired image-to-image translation
* Two generators + two discriminators
* Uses:

  * Adversarial loss
  * Cycle-consistency loss
  * Identity loss

### Comparison

| Feature        | Neural Style Transfer        | CycleGAN                      |
| -------------- | ---------------------------- | ----------------------------- |
| Input          | Single content + style image | Unpaired datasets             |
| Training       | Minimal                      | Heavy                         |
| Speed          | Fast per image               | Fast inference after training |
| Generalization | Arbitrary styles             | Domain-specific               |
| Control        | Direct                       | Indirect                      |

## Key Takeaways

* Temporal modeling choice strongly impacts performance
* Explicit sequence modeling outperforms shallow temporal filters on small datasets
* Model efficiency and accuracy exist in direct tension
* Learned visual representations transfer naturally to generative tasks

## Future Work

* Train on full UCF101 for better generalization
* Evaluate fine-grained and ambiguous actions
* Optimize models for real-time and mobile inference
* Extend to:

  * Video segmentation
  * Lip reading
  * Multimodal action recognition

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pretrained CNN backbones
* CycleGAN (pretrained models)

---