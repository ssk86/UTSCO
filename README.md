# TSCO: Unsupervised Teacher-Student Collaborative Optimization for Cross-Domain Insulator Defect Detection

This repository contains the official implementation of our paper:

**Unsupervised Cross-Domain Teacher-Student Collaborative Optimization for Transmission Line Insulator Defect Detection**

---

## Overview

Insulator defect detection is a critical task in intelligent inspection of power transmission lines. However, the scarcity of labeled real-world data and the significant domain gap between synthetic and real images often limit the performance of deep learning based detectors.

To address these challenges, we propose a **Teacher-Student Collaborative Optimization (TSCO)** framework for unsupervised domain adaptation in insulator defect detection. The proposed framework performs collaborative optimization from three aspects: robust representation learning, pseudo-label refinement, and instance-level domain alignment.

---

## Key Contributions

- **Cross-View Invariance Enhancement (CVIE)**  
  Enhances feature robustness by enforcing semantic and spatial consistency across different augmented views.

- **Low Confidence Driven Self-Refinement (LCSR)**  
  Improves pseudo-label quality by exploiting low-confidence instances instead of simply discarding them.

- **Curriculum-guided Hard Instance Alignment (CHIA)**  
  Performs progressive instance-level domain alignment using curriculum-guided filtering and hard-sample weighting.

- **Collaborative Teacher-Student Optimization**  
  Enables mutual enhancement between the teacher and student models, leading to improved domain generalization.

---

## Framework

TSCO is built upon a teacher-student architecture for unsupervised domain adaptation:

- The **student model** learns from:
  - labeled source-domain images
  - pseudo-labeled target-domain images

- The **teacher model**:
  - is updated by exponential moving average (EMA)
  - is further refined using low-confidence driven self-optimization

- Domain adaptation is achieved through:
  - cross-view consistency learning
  - instance-level adversarial alignment
  - curriculum-guided hard-sample selection

---

## Experimental Results

### Self-Built Dataset

| Method | Damage AP50 | Drop AP50 |
|--------|-------------|-----------|
| Baseline | 68.2 | 83.8 |
| TSCO (Ours) | **75.3** | **88.9** |

### Public Datasets

| Method | Damage AP50 | Drop AP50 |
|--------|-------------|-----------|
| AT | 58.6 | 85.4 |
| DA2OD | 62.5 | 89.4 |
| TSCO (Ours) | **64.1** | **90.5** |

---

## Additional Analysis

To provide a comprehensive evaluation, we further analyze the proposed method from multiple perspectives:

- feature distribution alignment (MMD)
- IoU distribution statistics
- pseudo-label quality visualization
- qualitative detection results

These analyses demonstrate that TSCO improves:

- domain alignment
- pseudo-label reliability
- detection robustness
- cross-domain generalization capability

---

# Installation

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.5 and torchvision compatible with the installed PyTorch version
- Detectron2 == 0.3

## Our Tested Environment

- Ubuntu 20.04
- Python 3.8.3
- PyTorch 1.7.0
- Torchvision 0.12.0
- CUDA 11.0
- NVIDIA GeForce RTX 3090
- Batch size: 4 (2 source images + 2 target images)

## Install Python Environment

We recommend creating a virtual environment before installing the dependencies.

```bash
python3 -m venv /path/to/new/virtual/environment/
source /path/to/new/virtual/environment/bin/activate
