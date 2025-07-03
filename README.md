# Towards Adaptive Deep Learning: Model Elasticity via Prune-and-Grow CNN Architectures

This repository contains the implementation and experimental framework for my Master's thesis at the University of Amsterdam and Vrije Universiteit Amsterdam. The project focuses on making convolutional neural networks (CNNs) adaptive to varying computational constraints by combining structured pruning with dynamic rebuilding to support **model elasticity**.

## 🧠 Motivation

Deploying deep CNNs on edge devices is challenging due to their static structure and computational demands. This work explores methods to dynamically scale a network's capacity—pruning to shrink and rebuilding to grow—based on resource availability, enabling efficient inference across environments like smartphones, GPUs, and embedded systems.

## 🛠️ Core Contributions

- **Dependency Graph-Based Pruning:** Safe structured pruning preserving layer compatibility.
- **Importance Metric Selection:** Magnitude, Taylor, and Hessian-based filter pruning.
- **Rebuilding Strategy:** Regrows pruned layers using stored metadata.
- **Iterative Prune-and-Grow:** Embeds multiple model capacities into a single architecture.
- **Multi-Architecture Evaluation:** VGG-16, ResNet-20, ResNet-56, and AlexNet on CIFAR-10.

## 📁 Repository Structure

