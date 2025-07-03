# Towards Adaptive Deep Learning: Model Elasticity via Prune-and-Grow CNN Architectures

This repository contains the implementation and experimental framework for my Master's thesis at the University of Amsterdam and Vrije Universiteit Amsterdam. The project focuses on making convolutional neural networks (CNNs) adaptive to varying computational constraints by combining structured pruning with dynamic rebuilding to support **model elasticity**.

📄 [arXiv paper](https://arxiv.org/abs/2505.11569)


---
## 🧠 Motivation

Deploying deep CNNs on edge devices is challenging due to their static structure and computational demands. This work explores methods to dynamically scale a network's capacity—pruning to shrink and rebuilding to grow—based on resource availability, enabling efficient inference across environments like smartphones, GPUs, and embedded systems.

---
## 🚀 Overview

Deep CNNs are typically static and computationally expensive. This project proposes a **prune-and-grow pipeline** that enables CNNs like **ResNet**, **VGG**, and **AlexNet** to scale dynamically:

- **Pruning:** Reduces model size and complexity.
- **Rebuilding:** Restores pruned parameters to recover capacity.
- **Iterative Nesting:** Embeds multiple sub-networks within one architecture.

---

## 🧩 Key Components

| Module                     | Description |
|---------------------------|-------------|
| `depGraph_pruning.py`     | Prunes using dependency graphs (structured pruning). |
| `high_level_pruner.py`    | Uses importance metrics (magnitude, Taylor, Hessian). |
| `iterative_pruning.py`    | Implements prune–rebuild nesting strategy. |
| `softPruning.py`          | Lightweight soft pruning logic. |
| `scripts/`                | Training and evaluation scripts. |
| `utils/`                  | Metadata storage, evaluation, helpers. |
| `vision/`                 | Model architecture definitions (ResNet, VGG, AlexNet). |

---

## 📊 Experiments

- **Architectures:** VGG-16, ResNet-20, ResNet-56, AlexNet  
- **Datasets:** CIFAR-10, Imagenette  
- **Evaluation:** Accuracy, parameters, model size (MB), adaptability across pruning levels  
- **Tracking:** All metrics tracked using **Weights & Biases**

---

## ✨ Key Contributions
| # | Contribution | One-liner |
|---|--------------|-----------|
| 1 | **DepGraph Pruning** | Structured, dependency-safe channel pruning |
| 2 | **Importance Metrics** | Magnitude, Taylor & Hessian criteria implemented |
| 3 | **Adaptive Rebuilding** | Regrows pruned filters using stored metadata |
| 4 | **Iterative Prune-and-Grow** | Embeds multiple capacities inside one checkpoint |
| 5 | **Cross-Arch Benchmarks** | VGG-16, ResNet-20/56, AlexNet on CIFAR-10 & Imagenette |

---

## 📊 Results Summary

- Achieved significant parameter and memory reduction with minimal accuracy loss.
- Adaptive models recover accuracy through rebuilding and fine-tuning.
- Single architecture supports nested sub-models for real-time resource scaling.

## 🧪 Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.13
- Torch-Pruning
- PyTorch Lightning
- Weights & Biases (optional for logging)


---

## ⚙️ Quick Start
```bash
git clone https://github.com/poojamangal15/Adaptive-Neural-Networks.git
cd Adaptive-Neural-Networks
pip install -r requirements.txt

