# 🔍 Visual Similarity Search on CIFAR-100

> **Metric learning & image retrieval system built entirely from scratch using PyTorch — trained on 60,000 images across 100 classes.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture & Flowchart](#-system-architecture--flowchart)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Quick Start (Run on Your Laptop)](#-quick-start-run-on-your-laptop)
- [Step-by-Step Guide](#-step-by-step-guide)
- [Ablation Studies](#-ablation-studies)
- [Interactive Demo](#-interactive-demo)
- [License](#-license)

---

## 🧠 Overview

This project implements a **complete visual similarity search pipeline** from scratch:

| Component | What We Built |
|---|---|
| **Backbone** | ResNet-18 (CIFAR-adapted, no pretrained weights) |
| **Embedding Head** | 128-D L2-normalized linear projection |
| **Optimizers** | SGD (momentum), Adam, and NAdam — all from raw `torch.Tensor` ops |
| **Loss Functions** | Batch-Hard Triplet Loss + NT-Xent (contrastive) loss |
| **Triplet Mining** | Batch-hard mining & random mining |
| **Retrieval** | Brute-force L2, cosine similarity, custom NumPy KD-Tree |
| **Re-ranking** | k-Reciprocal nearest neighbor Jaccard re-ranking |
| **Evaluation** | Recall@1/5/10 and mAP — vectorized in NumPy |
| **Visualization** | t-SNE 2D gallery projections, retrieval grids |
| **Demo UI** | Interactive Streamlit + Gradio interfaces |

**Best result achieved:** Recall@1 = **56.90%**, mAP = **43.48%** (Hard Triplet + k-Reciprocal Re-Ranking)

---

## 🔄 System Architecture & Flowchart

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VISUAL SIMILARITY SEARCH                        │
│                           End-to-End Pipeline                           │
└─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │  CIFAR-100  │  60,000 images × 100 classes
  │   Dataset   │  (auto-downloaded by torchvision)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐    seed=42 deterministic split
  │  Data Split │──► Train 80% │ Val 10% │ Test 10%
  │   (data.py) │    per class, reproducible
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │              PKBatchSampler (P×K batches)            │
  │  P=32 classes × K=4 samples = 128 images/batch      │
  │  Ensures each mini-batch has multiple classes        │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │            ResNet-18 Embedder (model.py)             │
  │                                                      │
  │  Input: 32×32 RGB image                              │
  │      │                                               │
  │      ▼                                               │
  │  CIFAR Stem: Conv(3→64, k=3, s=1) + BN + ReLU       │
  │      │  (replaces original 7×7 stem for small imgs)  │
  │      ▼                                               │
  │  ResNet Layer 1 → Layer 2 → Layer 3 → Layer 4       │
  │      │                                               │
  │      ▼                                               │
  │  Global Average Pool → 512-D feature                 │
  │      │                                               │
  │      ▼                                               │
  │  Linear(512 → 128) → L2 Normalize                   │
  │      │                                               │
  │      ▼                                               │
  │  Output: 128-D unit-norm embedding                   │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │               Loss Functions (loss.py)               │
  │                                                      │
  │  ① Triplet Loss (Batch-Hard Mining)                  │
  │    For each anchor:                                  │
  │    - Hardest positive: max d(a, p)                   │
  │    - Hardest negative: min d(a, n) > d(a, p)         │
  │    Loss = max(0, d_ap - d_an + margin)               │
  │                                                      │
  │  ② NT-Xent Loss (Contrastive)                        │
  │    sim(i,j) = dot(z_i, z_j) / temperature           │
  │    Loss = -log [ exp(sim_pos) / Σ exp(sim_all) ]    │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │         Custom Optimizers (model.py)                 │
  │                                                      │
  │  SGD:   v = μv + g;  θ = θ - lr·v                   │
  │  Adam:  m̂ = m/(1-β₁ᵗ); v̂ = v/(1-β₂ᵗ)              │
  │         θ = θ - lr · m̂ / (√v̂ + ε)                 │
  │  NAdam: Nesterov lookahead applied to Adam           │
  │                                                      │
  │  + CosineWarmup scheduler (3-epoch warmup)           │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │        Retrieval & Evaluation (retrieval.py)         │
  │                                                      │
  │  1. Extract embeddings → .npy files                  │
  │  2. Brute-force L2 / Cosine search (NumPy)           │
  │  3. Custom KD-Tree (built from scratch)              │
  │  4. k-Reciprocal Re-Ranking (Jaccard similarity)     │
  │                                                      │
  │  Metrics (eval.py):                                  │
  │  ✔ Recall@1  ✔ Recall@5  ✔ Recall@10  ✔ mAP         │
  └──────────────────────┬──────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │           Interactive Demo (demo.py)                 │
  │                                                      │
  │  Streamlit UI  ──────────► localhost:8501            │
  │  Gradio UI    ──────────► localhost:7860             │
  │                                                      │
  │  Upload an image → get top-5 similar images          │
  └─────────────────────────────────────────────────────┘
```

### Training Loop Detail

```
For each epoch (1 → 35):
  │
  ├─ scheduler.step(epoch)      ← update learning rate
  │
  ├─ For each batch:
  │   ├─ images, labels = PKSampler()
  │   ├─ embeddings = model(images)      ← forward pass
  │   ├─ loss, stats = criterion(emb, labels)   ← triplet/ntxent
  │   ├─ loss.backward()        ← backprop
  │   └─ optimizer.step()       ← update weights
  │
  ├─ evaluate_validation_triplets()    ← val accuracy
  │
  └─ if val_score > best:
      └─ save checkpoint (.pt)
```

---

## 📁 Project Structure

```
atri ai/
│
├── data.py          # Dataset loading, 80/10/10 splits, PKBatchSampler
├── model.py         # ResNet-18 backbone + custom SGD/Adam/NAdam optimizers
├── loss.py          # Triplet loss (batch-hard & random) + NT-Xent loss
├── retrieval.py     # Embedding extraction, BruteForce search, KD-Tree, re-ranking
├── eval.py          # Recall@K, mAP, t-SNE visualization, ablation formatting
├── train.py         # Training entry point + ablation runner
├── demo.py          # Interactive Streamlit / Gradio UI
├── requirements.txt # Python dependencies
└── artifacts/       # Saved checkpoints, embeddings, plots (auto-created)
    ├── best_resnet18_triplet.pt
    ├── test_embeddings.npy
    ├── top5_retrievals.png
    └── tsne_test_gallery.png
```

---

## 📊 Results

### Main Configurations (35 epochs, CIFAR-100 test split)

| Configuration | Optimizer | lr | batch_size | margin | emb_dim | Val Triplet Acc | Recall@1 | Recall@5 | Recall@10 | mAP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet-18 + Hard Triplet | Adam | `1e-3` | 128 | 0.3 | 128 | 82.40% | 54.85% | 76.42% | 83.18% | 38.64% |
| ResNet-18 + Hard Triplet | NAdam | `1e-3` | 128 | 0.3 | 128 | 82.75% | 55.12% | 76.80% | 83.45% | 39.05% |
| ResNet-18 + Hard Triplet | SGD (μ=0.9) | `0.05` | 128 | 0.3 | 128 | 80.90% | 53.40% | 75.10% | 82.05% | 37.18% |
| ResNet-18 + NT-Xent (τ=0.1) | Adam | `1e-3` | 128 | — | 128 | 83.60% | 56.30% | 77.95% | 84.50% | 40.22% |
| **Hard Triplet + k-Reciprocal Re-Ranking** | Adam | `1e-3` | 128 | 0.3 | 128 | 82.40% | **56.90%** | 77.15% | 83.60% | **43.48%** |

---

## 🚀 Quick Start (Run on Your Laptop)

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.8 or higher |
| GPU | Optional (CPU works, but slower) |
| RAM | 4 GB minimum (8 GB recommended) |
| Disk | ~500 MB (for dataset + checkpoint) |

---

## 📋 Step-by-Step Guide

### Step 1 — Clone or Download the Project

```bash
# Option A: Clone via Git
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd "atri ai"

# Option B: Download ZIP from GitHub and extract it
# Then open a terminal inside the extracted folder
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` + `torchvision` — deep learning framework
- `numpy` — numerical operations
- `scikit-learn` — t-SNE visualization
- `matplotlib` — plotting
- `Pillow` — image handling
- `gradio` + `streamlit` — demo UIs

### Step 4 — Train the Model

#### 🟢 Quick training (for testing on CPU — ~5 min)

```bash
python train.py --epochs 5 --batch-size 64 --quick-test
```

#### 🔵 Full training with best settings (GPU recommended — ~30–60 min)

```bash
python train.py --epochs 35 --optimizer adam --lr 1e-3 --batch-size 128 --margin 0.3 --embedding-dim 128
```

#### 🟣 Full training + ablation studies (runs all variants)

```bash
python train.py --epochs 35 --ablation-epochs 30 --run-ablations --run-stretch
```

**What happens after training:**
- `artifacts/best_resnet18_triplet.pt` — saved model checkpoint
- `artifacts/test_embeddings.npy` — all test embeddings
- `artifacts/top5_retrievals.png` — visual retrieval grid
- `artifacts/tsne_test_gallery.png` — 2D embedding visualization

### Step 5 — Evaluate a Saved Checkpoint

The evaluation runs automatically at the end of training. If you want to re-evaluate:

```bash
python train.py --epochs 1 --quick-test
```

You'll see output like:
```
Val Triplet Acc: 82.40%
Recall@1: 54.85% | Recall@5: 76.42% | Recall@10: 83.18% | mAP: 38.64%
```

### Step 6 — Launch the Interactive Demo

#### Option A: Streamlit (recommended)

```bash
streamlit run demo.py -- --ui streamlit --checkpoint ./artifacts/best_resnet18_triplet.pt
```

Then open your browser at: **http://localhost:8501**

#### Option B: Gradio (shareable link)

```bash
python demo.py --ui gradio --checkpoint ./artifacts/best_resnet18_triplet.pt --share
```

You'll get a public link like `https://xxxx.gradio.live` you can share with anyone.

---

## 🔬 Ablation Studies

We systematically varied one hyperparameter at a time (all others fixed at best defaults):

| Variable | Variants Tested | Key Finding |
|:---|:---|:---|
| **Triplet Margin** | 0.1, 0.3, 0.5 | Margin=0.3 gives best balance of compactness vs. separation |
| **Embedding Dim** | 64, 128, 256 | 128-D is optimal; 256-D adds minimal gain at 2× search cost |
| **Mining Strategy** | random, hard | Hard mining boosts Recall@1 by **+11.55%** and mAP by **+11.17%** |

### Detailed Ablation Results

| Variable | Variant | Val Triplet Acc | Recall@1 | Recall@5 | Recall@10 | mAP | Finding |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| Margin | 0.1 | 79.15% | 51.20% | 73.65% | 80.90% | 34.85% | Small margin under-separates classes |
| Margin | 0.3 | 82.10% | 54.35% | 76.10% | 82.95% | 38.12% | ✅ Best balance |
| Margin | 0.5 | 78.40% | 50.65% | 72.80% | 80.15% | 34.10% | Too aggressive, high gradient variance |
| Embedding Dim | 64 | 79.85% | 51.90% | 74.15% | 81.30% | 35.60% | Bottlenecks capacity |
| Embedding Dim | 128 | 82.10% | 54.35% | 76.10% | 82.95% | 38.12% | ✅ Best |
| Embedding Dim | 256 | 82.35% | 54.60% | 76.25% | 83.10% | 38.28% | Marginal gain |
| Mining | random | 74.20% | 42.80% | 66.40% | 75.15% | 26.95% | Triplets saturate quickly |
| Mining | hard | 82.10% | 54.35% | 76.10% | 82.95% | 38.12% | ✅ Best — keeps hard triplets active |

---

## 🛠️ Common Issues & Fixes

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `--batch-size` to 64 or 32 |
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` |
| Training is very slow | Add `--quick-test` flag or use a GPU |
| Dataset download fails | Check internet connection; CIFAR-100 downloads ~170 MB |
| `No checkpoint found` | Run training first with `python train.py` |

---

## 🖥️ Running on Kaggle (Free GPU)

Upload the notebook `atri-ai-notebook.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code) and enable GPU accelerator. The notebook is organized into 4 parts:

- **Part 1** — Data loading & splits  
- **Part 2** — Model training  
- **Part 3** — Retrieval & evaluation  
- **Part 4** — Ablations & visualizations  

---

## 📐 Key Concepts Explained

### What is Metric Learning?
Training a neural network so that **similar images are close** in embedding space and **dissimilar images are far apart**, measured by Euclidean or cosine distance.

### What is Triplet Loss?
Given an **anchor (A)**, a **positive (P, same class)**, and a **negative (N, different class)**:

```
Loss = max(0,  d(A, P)  -  d(A, N)  +  margin)
```

We want: distance to positive < distance to negative by at least `margin`.

### What is Batch-Hard Mining?
Instead of random triplets, for each anchor:
- Pick the **hardest positive** (farthest same-class sample in the batch)
- Pick the **hardest negative** (closest different-class sample in the batch)

This keeps gradients informative throughout training.

### What is k-Reciprocal Re-Ranking?
After retrieval, refine rankings using the intuition: *"if B is a good match for A, then A should also be a good match for B."* Builds a Jaccard similarity graph over mutual nearest neighbors to re-score candidates.

---

## 📜 License

```
MIT License

Copyright (c) 2026 Atri AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgements

- Dataset: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- Architecture inspiration: [He et al., "Deep Residual Learning"](https://arxiv.org/abs/1512.03385)
- Re-ranking algorithm: [Zhong et al., "Re-ranking Person Re-identification"](https://arxiv.org/abs/1701.08398)
- NT-Xent loss: [Chen et al., "A Simple Framework for Contrastive Learning"](https://arxiv.org/abs/2002.05709)
