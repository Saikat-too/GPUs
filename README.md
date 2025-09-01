# GPUs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11%2B-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains implementations and notes related to **GPU programming**, focusing on both **CUDA** (for general-purpose GPU computing) and **PyTorch** (for deep learning on GPUs).

---

## 📖 Table of Contents
- [PMPP (Programming Massively Parallel Processors)](#pmpp-parallel-memory-programming-for-gpus)
  - [Chapter 1: Introduction](#chapter-1-introduction)
  - [Chapter 2: Heterogeneous Data Parallel Computing](#chapter-2-heterogeneous-data-parallel-computing)
  - [Chapter 3: Multidimensional Grids and Data](#chapter-3-multidimensional-grids-and-data)
- [PyTorch Implementations](#pytorch-implementations)
  - [Basic Operations](#basic-operations)
  - [Neural Network Components](#neural-network-components)
  - [Architectures](#architectures)
- [Requirements](#requirements)
- [Setup](#setup)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## 📘 PMPP (Programming Massively Parallel Processors)

### ✔ Chapter 1: Introduction
- [x] Overview of CPU vs. GPU design philosophies (latency vs. throughput)
- [x] Explanation of GPU specialization for parallel computations

### ✔ Chapter 2: Heterogeneous Data Parallel Computing
- [x] Concepts of data parallelism
- [x] CUDA C program structure (Host-Device interaction)
- [x] Kernel execution and Grid/Block/Thread hierarchy
- [x] CUDA memory management (`cudaMalloc`, `cudaFree`, `cudaMemcpy`)
- [x] Example: Vector addition in CUDA
- [x] CUDA thread basics and global thread ID calculation

### ✔ Chapter 3: Multidimensional Grids and Data
- [x] Using `dim3` for 2D grids and blocks
- [x] Linearizing 2D arrays (row-major layout)
- [x] Example: Matrix multiplication in CUDA
- [x] Calculating global thread indices in 2D

---

## ⚡ PyTorch Implementations

### ✔ Basic Operations
- [x] `basic_operations.py` — Fundamental tensor operations (addition, multiplication, square root, mean)
- [x] `broadcasting_examples.py` — PyTorch broadcasting rules & common pitfalls
- [x] `gradient_computation.py` — Autograd, computational graphs, gradient retention
- [x] `inplace_operation.py` — In-place vs. out-of-place operations & their effect on gradients
- [x] `memory_layout.py` — Tensor memory layouts (contiguous vs. non-contiguous), `.view()` vs. `.reshape()`

### ✔ Neural Network Components
- [x] `activation_functions.py` — Implementations of ReLU, Sigmoid, Tanh, Leaky ReLU (from scratch & PyTorch versions)
- [x] `customdataset.py` — Custom Dataset class for regression/classification with data augmentation
- [x] `customlosses.py` — Common loss functions (MSE, Cross-Entropy, BCE) implemented from scratch + PyTorch comparison
- [x] `initialize_weights.py` — Weight initialization strategies (Xavier, He, Custom)
- [x] `linear_layer.py` — Linear layer implementation with `nn.Parameter`, initialization & forward pass
- [x] `mlp.py` — Configurable Multi-Layer Perceptron (MLP) with dropout, batch norm, activations
- [x] `train.py` — Flexible training loop with validation, metrics, early stopping, LR scheduling

### ✔ Architectures
- [x] `attention/attention.py` — Basic attention mechanism implementation
- [x] `attention/README.md` — Explanation of Query, Key, Value, scoring, scaling, softmax, weighted aggregation
- [x] `cnn/cnn.py` — CNN for image classification with multiple conv layers, pooling, BN, GAP
- [x] `cnn/README.md` — Introduction to CNNs, architecture breakdown, receptive field, stride, padding, applications
- [x] `ResNet/resnet.py` — Residual Block implementation with optional downsampling
- [x] `ResNet/README.md` — Deep Residual Learning, vanishing gradient problem & ResNet shortcut solution

---

## 📦 Requirements
See [`pytorch/requirements.txt`](pytorch/requirements.txt) for required Python packages.

---

## ⚙️ Setup
Use the provided script to set up a virtual environment and install dependencies:

```bash
bash pytorch/setup_venv.sh
source pytorch/venv/bin/activate
```

## 📜 License

This project is licensed under the **MIT License**.

## 🤝 Contributing

Contributions are welcome!
If you’d like to improve this project, please feel free to:
- Open an **issue** to report bugs or suggest features.
- Submit a **pull request** with improvements or new implementations.

Your contributions will help make this repository more useful for everyone. 🚀

---

## 📬 Contact

For any questions, feedback, or collaboration opportunities:
- Open an **issue** in this repository.
- Alternatively, reach out via GitHub Discussions (if enabled).

I’d love to hear from you! 💡
