# Deep Learning Architectures for Colorectal Cancer Histopathology Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.16-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.10.0-red?logo=keras)
![License](https://img.shields.io/badge/License-Academic-green)

**Master's Thesis — Comparative Analysis of CNN and Transformer Architectures**

[Overview](#1-overview) • [Results](#2-key-results) • [Installation](#3-installation) • [Project Structure](#4-project-structure) • [Notebooks](#5-notebooks) • [Modules](#6-source-modules)

</div>

---

## 1. Overview

### 1.1 Project Description

This project presents a **systematic comparative analysis** of five deep learning architectures for automated classification of colorectal cancer histopathology images. The study evaluates both traditional Convolutional Neural Networks (CNNs) and Vision Transformers (ViT), with a focus on the practical trade-offs between **classification accuracy** and **computational efficiency** for clinical deployment.

| Aspect | Description |
|--------|-------------|
| **Domain** | Medical Image Analysis — Histopathology |
| **Task** | Multi-class tissue classification (8 classes) |
| **Dataset** | Kather et al. (2016) — 5,000 H&E stained images |
| **Architectures** | CNN Baseline, VGG19, ResNet50, EfficientNetB0, ViT-B/16 |
| **Framework** | TensorFlow 2.10.1 / Keras 2.10.0 |

### 1.2 Dataset at a Glance

<div align="center">

| Property | Value |
|----------|-------|
| **Total Images** | 5,000 H&E stained histological images |
| **Image Size** | 150 × 150 pixels |
| **Classes** | 8 tissue types |
| **Format** | TIFF (RGB, 8-bit) |
| **Magnification** | 20× objective (0.495 µm/pixel) |
| **Source** | Kather et al. (2016) — *Scientific Reports* |

</div>

**Tissue Classes:**

| Class | Description | Class | Description |
|-------|-------------|-------|-------------|
| `01_TUMOR` | Cancerous epithelium | `05_DEBRIS` | Necrotic material |
| `02_STROMA` | Connective tissue | `06_MUCOSA` | Normal mucosal tissue |
| `03_COMPLEX` | Mixed structures | `07_ADIPOSE` | Fat cells |
| `04_LYMPHO` | Immune cells | `08_EMPTY` | Background/artifacts |

**Data Split:** 70% train (3,500) / 15% validation (750) / 15% test (750) — stratified by class.

### 1.3 Research Question

> **"Which deep learning architecture provides the optimal balance between classification accuracy and computational efficiency for histopathological tissue classification in colorectal cancer?"**

### 1.4 Key Contributions

1. **Comprehensive Benchmark**: First systematic comparison of 5 architectures (4 CNNs + 1 Transformer) on the Kather colorectal histopathology dataset with identical experimental conditions.

2. **Statistical Rigor**: McNemar's test with Bonferroni correction for pairwise model comparisons, bootstrap confidence intervals, and Cohen's d effect sizes.

3. **Efficiency Analysis**: Multi-dimensional evaluation including FLOPs, inference latency, memory usage, and model size — not just accuracy.

4. **Deployment Recommendations**: Evidence-based guidance for clinical server vs. edge device deployment scenarios.

5. **Reproducible Pipeline**: Modular, GPU-optimized data pipeline with standardized preprocessing for fair model comparison.

---

## 2. Key Results

### 2.1 Classification Performance

| Model | Test Accuracy | Macro F1 | Cohen's Kappa | Parameters |
|-------|---------------|----------|---------------|------------|
| **ResNet50** | **95.07%** | **0.9510** | **0.9436** | 25.70M |
| VGG19 | 94.53% | 0.9457 | 0.9375 | 20.09M |
| EfficientNetB0 | 93.33% | 0.9338 | 0.9238 | 4.71M |
| ViT-B/16 | 92.27% | 0.9229 | 0.9116 | 86.20M |
| CNN Baseline | 92.00% | 0.9211 | 0.9086 | 1.24M |

### 2.2 Computational Efficiency

| Model | FLOPs (G) | Inference (ms) | Size (MB) | Accuracy/MParam |
|-------|-----------|----------------|-----------|-----------------|
| CNN Baseline | 2.26 | 27.09 | 15.21 | 74.19% |
| **VGG19** | 39.04 | **24.62** | ~80 | 4.70% |
| ResNet50 | 7.73 | 181.66 | 206.68 | 3.70% |
| **EfficientNetB0** | **0.79** | 169.00 | **47.62** | **19.82%** |
| ViT-B/16 | 35.23 | 328.15 | 332.07 | 1.07% |

### 2.3 Statistical Significance

With Bonferroni correction (α = 0.005), only two comparisons showed **statistically significant differences**:

- ✅ **ResNet50 vs CNN Baseline** (p = 0.0008) — Transfer learning provides real improvement
- ✅ **ResNet50 vs ViT-B/16** (p = 0.0010) — CNNs outperform Transformers on small datasets

All other pairwise comparisons (VGG19 vs ResNet50, EfficientNetB0 vs VGG19, etc.) were **not statistically significant**, indicating comparable performance.

### 2.4 Deployment Recommendations

| Scenario | Recommended Model | Justification |
|----------|-------------------|---------------|
| **Clinical Server** (Max Accuracy) | ResNet50 | 95.07% accuracy, best TUMOR F1 (0.9787) |
| **Balanced Deployment** | EfficientNetB0 | 93.33% accuracy with 5× fewer params |
| **Edge/Mobile Device** | CNN Baseline | 92.00% accuracy, only 1.24M params, 15 MB |

---

## 3. Installation

### 3.1 Prerequisites

- **Python**: 3.10.16
- **CUDA**: 11.2
- **cuDNN**: 8.1
- **GPU**: NVIDIA with ≥6GB VRAM (tested on GTX 1650)

### 3.2 Environment Setup

```bash
# Clone the repository
git clone https://github.com/username/capstone.git
cd capstone

# Create virtual environment
python -m venv tensor-gpu
source tensor-gpu/bin/activate  # Linux/Mac
# or
.\tensor-gpu\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 3.3 Dependencies

```
tensorflow==2.10.1
keras==2.10.0
keras-tuner==1.3.5
tensorflow-io==0.31.0
vit-keras==0.1.2
numpy==1.23.5
pandas==2.1.4
scikit-learn==1.1.3
matplotlib==3.8.2
seaborn==0.12.2
statsmodels==0.14.0
```

---

## 4. Project Structure

```
capstone/
│
├── 📁 data/                          # Dataset (not in repo)
│   ├── train/                        # 3,500 images (70%)
│   ├── val/                          # 750 images (15%)
│   └── test/                         # 750 images (15%)
│
├── 📁 notebooks/                     # Jupyter notebooks (main experiments)
│   ├── 01_Data_Analysis_and_Preparation.ipynb
│   ├── 02b_CNN_Baseline_Improved.ipynb
│   ├── 03_Transfer_Learning_VGG.ipynb
│   ├── 04a_Transfer_Learning_ResNet50.ipynb
│   ├── 05_EfficientNet_Transfer_Learning.ipynb
│   ├── 06_VisionTransformer_ViT-B16.ipynb
│   └── 07_Model_Comparison_Analysis.ipynb
│
├── 📁 src/                           # Reusable Python modules
│   ├── shared_pipeline.py            # Unified data pipeline
│   ├── analysis_utils.py             # Metrics, FLOPs, visualization
│   ├── statistical_tests.py          # McNemar, Bootstrap, Cohen's d
│   ├── model_utils.py                # Model building utilities
│   ├── hyperparameter_tuning.py      # Keras Tuner configurations
│   └── data_utils.py                 # Data loading helpers
│
├── 📁 models/                        # Saved trained models
│   ├── baseline_improved/            # CNN Baseline (.keras)
│   ├── vgg/                          # VGG19 (.h5)
│   ├── resnet50/                     # ResNet50 (.keras)
│   ├── efficientnet/                 # EfficientNetB0 (.keras)
│   └── vit/                          # ViT-B/16 (.keras)
│
├── 📁 logs/                          # Training histories & TensorBoard
│   ├── baseline_improved/
│   ├── vgg/
│   ├── resnet50/
│   ├── efficientnet/
│   └── vit/
│
├── 📁 results/                       # Experiment outputs
│   ├── figures/                      # Per-model plots (confusion matrices, ROC, etc.)
│   ├── comparison/                   # Cross-model comparison CSVs
│   ├── hyperparameters/              # Best hyperparameters JSON
│   └── metrics/                      # Performance metrics CSVs
│
├── 📁 informe_final/                 # Thesis document
│   └── borrador.md                   # Main thesis draft
│
├── 📄 requirements.txt               # Python dependencies
└── 📄 README.md                      # This file
```

---

## 5. Notebooks

### 5.1 Notebook Descriptions

| # | Notebook | Purpose | Duration |
|---|----------|---------|----------|
| 01 | `01_Data_Analysis_and_Preparation.ipynb` | EDA, class distribution, data splitting | ~30 min |
| 02 | `02b_CNN_Baseline_Improved.ipynb` | Custom CNN trained from scratch (baseline) | ~2 hours |
| 03 | `03_Transfer_Learning_VGG.ipynb` | VGG19 with Bayesian hyperparameter optimization | ~4 hours |
| 04 | `04a_Transfer_Learning_ResNet50.ipynb` | ResNet50 with Hyperband tuning | ~5 hours |
| 05 | `05_EfficientNet_Transfer_Learning.ipynb` | EfficientNetB0 with compound scaling | ~3 hours |
| 06 | `06_VisionTransformer_ViT-B16.ipynb` | ViT-B/16 with warmup cosine decay | ~2 hours |
| 07 | `07_Model_Comparison_Analysis.ipynb` | Statistical comparison & final analysis | ~30 min |

### 5.2 Execution Order

```
01 → 02 → 03 → 04 → 05 → 06 → 07
         ↓    ↓    ↓    ↓    ↓
       (models can be trained in parallel)
                              ↓
                         07 requires all models
```

### 5.3 Key Features per Notebook

**02b - CNN Baseline:**
- 4 convolutional blocks (32→64→128→256 filters)
- SpatialDropout2D + Dropout regularization
- HSV color augmentation + geometric augmentation
- Class weights for imbalance handling

**03 - VGG19:**
- Bayesian Optimization (50 trials)
- Block5 fine-tuning (name-based freezing)
- VGG preprocessing (BGR, ImageNet mean subtraction)

**04a - ResNet50:**
- Hyperband tuning with adaptive halving
- conv5_block1_out freezing strategy
- All BatchNormalization layers frozen (prevents distribution shift)
- Mixed precision training (float16)

**05 - EfficientNetB0:**
- Block6/Block7 fine-tuning
- Compound scaling (depth × width × resolution)
- MBConv blocks with Squeeze-and-Excitation
- ⚠️ **Note:** This notebook required a separate environment with **TensorFlow 2.9.0** due to a [known bug in TF 2.10](https://github.com/keras-team/keras/issues/17268) that prevents saving the model configuration to JSON format. The training was executed in a downgraded environment and the resulting model was exported for compatibility.

**06 - ViT-B/16:**
- Frozen backbone (0.46% trainable parameters)
- AdamW optimizer with decoupled weight decay
- Warmup Cosine Decay learning rate schedule
- ImageNet normalization (mean=0.5, std=0.5)

---

## 6. Source Modules

### 6.1 `shared_pipeline.py`

Unified, GPU-optimized data pipeline for all models.

```python
from shared_pipeline import create_dataset, get_config, CLASS_NAMES

# Example: Create datasets for ResNet50
config = get_config(model_type='resnet', img_size=224, batch_size=32)
train_ds = create_dataset(train_dir, config, augment=True)
val_ds = create_dataset(val_dir, config, augment=False)
```

**Features:**
- Model-specific preprocessing (VGG, ResNet, EfficientNet, ViT)
- Standardized geometric augmentation (rotations, flips, zoom)
- TIFF image loading via TensorFlow I/O
- Automatic prefetching and caching

### 6.2 `analysis_utils.py`

Metrics calculation, FLOPs measurement, and visualization.

```python
from analysis_utils import get_flops, measure_inference_time, plot_confusion_matrix

# Calculate FLOPs
flops = get_flops(model, batch_size=1)
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")

# Measure inference time
avg_time, std_time = measure_inference_time(model, test_dataset, n_runs=100)
```

**Key Functions:**
- `get_flops()` — Calculate floating-point operations
- `measure_inference_time()` — Benchmark inference latency
- `track_peak_memory()` — Monitor GPU memory usage
- `plot_confusion_matrix()` — Generate normalized confusion matrix
- `plot_roc_curve()` — Per-class ROC curves with AUC
- `plot_learning_curves()` — Training/validation dynamics

### 6.3 `statistical_tests.py`

Rigorous statistical comparison between models.

```python
from statistical_tests import mcnemar_test, bootstrap_ci, cohens_d

# McNemar's test for paired comparison
result = mcnemar_test(y_true, y_pred_resnet, y_pred_vgg)
print(f"p-value: {result['p_value']:.4f}")

# Bootstrap confidence interval
ci_low, ci_high = bootstrap_ci(y_true, y_pred, n_bootstrap=1000)
```

**Statistical Tests Implemented:**
- **McNemar's Test** — Paired classifier comparison
- **Bootstrap Confidence Intervals** — 95% CI for metrics
- **Cohen's d** — Effect size quantification
- **Bonferroni Correction** — Multiple comparison adjustment

### 6.4 `hyperparameter_tuning.py`

Keras Tuner configurations for each architecture.

**Tuning Strategies:**
- **Bayesian Optimization** — VGG19, EfficientNetB0
- **Hyperband** — ResNet50

**Hyperparameter Ranges:**
| Parameter | Range | Sampling |
|-----------|-------|----------|
| Learning Rate | 1e-5 to 1e-3 | Log-uniform |
| Dropout Rate | 0.2 to 0.5 | Step 0.1 |
| Dense Units | {256, 512, 1024} | Categorical |
| L2 Regularization | 1e-4 to 1e-2 | Log-uniform |

---

## 7. Dataset

### 7.1 Colorectal Histology Dataset

**Source:** Kather, J. N., et al. (2016). *Scientific Reports*, 6, 27988.

| Property | Value |
|----------|-------|
| Total Images | 5,000 |
| Image Size | 150 × 150 pixels |
| Color Space | RGB (H&E staining) |
| File Format | TIFF |
| Magnification | 20× objective |

### 7.2 Tissue Classes (8)

| Class | Description | Clinical Relevance |
|-------|-------------|-------------------|
| `01_TUMOR` | Cancerous epithelium | **Primary diagnostic target** |
| `02_STROMA` | Connective tissue | Tumor microenvironment |
| `03_COMPLEX` | Mixed tissue structures | Challenging classification |
| `04_LYMPHO` | Lymphocyte infiltration | Immune response indicator |
| `05_DEBRIS` | Necrotic material | Tissue degradation |
| `06_MUCOSA` | Normal mucosal tissue | Reference tissue |
| `07_ADIPOSE` | Fat cells | Easily distinguishable |
| `08_EMPTY` | Background/artifacts | Quality control |

### 7.3 Data Split

```
Training:   3,500 images (70%) — Stratified by class
Validation:   750 images (15%) — Stratified by class
Test:         750 images (15%) — Stratified by class
```

---

## 8. Reproducibility

### 8.1 Random Seeds

All notebooks use fixed seeds for reproducibility:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

### 8.2 Hardware Used

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| CPU | Intel Core (8 cores) |
| RAM | 16 GB |
| OS | Windows 11 |

### 8.3 Training Times

| Model | Training Time | Epochs |
|-------|---------------|--------|
| CNN Baseline | ~45 min | 100 |
| VGG19 | ~90 min | 58 |
| ResNet50 | ~120 min | 60 |
| EfficientNetB0 | ~27 min | 50 |
| ViT-B/16 | ~118 min | 50 |

---

## 9. References

All referenced papers are available in the `docs/` folder organized by topic.

### Dataset

1. **Kather, J. N., Weis, C. A., Biber, F., et al.** (2016). Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*, 6, 27988. [[PDF]](docs/01.%20Kather,%20J.%20N.pdf)

### CNN Architectures

2. **Simonyan, K., & Zisserman, A.** (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*. [[PDF]](docs/01.%20CNN%20Architectures/Simonyan,%20K.,%20&%20Zisserman,%20A.%20(2015)%20VGG.pdf)

3. **Tan, M., & Le, Q. V.** (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*. [[PDF]](docs/01.%20CNN%20Architectures/Tan,%20M.,%20&%20Le,%20Q.%20V.%20(2019)%20EfficienNet.pdf)

### Vision Transformers

4. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*. [[PDF]](docs/05.%20ViT/An%20Image%20is%20Worth%2016x16%20Words.pdf)

5. **Vaswani, A., Shazeer, N., Parmar, N., et al.** (2017). Attention Is All You Need. *NeurIPS*. [[PDF]](docs/Attention%20is%20all%20you%20need.pdf)

6. **Loshchilov, I., & Hutter, F.** (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*. [[PDF]](docs/05.%20ViT/SGDR%20Stochastic%20Gradient%20Descent%20with%20Warm%20Restarts.pdf)

7. **Goyal, P., Dollár, P., Girshick, R., et al.** (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. *arXiv*. [[PDF]](docs/05.%20ViT/Accurate,%20Large%20Minibatch%20SGD.pdf)

### Transfer Learning in Medical Imaging

8. **Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S.** (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *NeurIPS*. [[PDF]](docs/02.%20Transfer%20Learning%20in%20Medical%20Imaging/Raghu,%20M.,%20Zhang,%20C.,%20Kleinberg,%20J.,%20&%20Bengio,%20S.%20(2019)%20TransferL.pdf)

9. **Yamashita, R., Nishio, M., Do, R. K. G., & Togashi, K.** (2018). Convolutional neural networks: an overview and application in radiology. *Insights into Imaging*, 9(4), 611-629. [[PDF]](docs/02.%20Transfer%20Learning%20in%20Medical%20Imaging/Yamashita,%20R.,%20Nishio,%20M.,%20Do,%20R.%20K.%20G.,%20&%20Togashi,%20K.%20(2018)%20CNN.pdf)

### Computational Efficiency

10. **Canziani, A., Paszke, A., & Culurciello, E.** (2016). An Analysis of Deep Neural Network Models for Practical Applications. *arXiv*. [[PDF]](docs/03.%20Computational%20Efficiency/Canziani,%20A.,%20Paszke,%20A.,%20&%20Culurciello,%20E.%20(2016)%20DNN.pdf)

11. **Howard, A. G., Zhu, M., Chen, B., et al.** (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv*. [[PDF]](docs/03.%20Computational%20Efficiency/Howard,%20A.%20G.,%20Zhu,%20M.,%20Chen,%20B.,%20et%20al.%20(2017)%20MobileNets.pdf)

### Histopathology Applications

12. **Komura, D., & Ishikawa, S.** (2018). Machine Learning Methods for Histopathological Image Analysis. *Computational and Structural Biotechnology Journal*, 16, 34-42. [[PDF]](docs/04.%20Histopathology-Specific%20Applications/Komura,%20D.,%20&%20Ishikawa,%20S.%20(2018).pdf)

13. **Tellez, D., Litjens, G., Bándi, P., et al.** (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*, 58, 101544. [[PDF]](docs/04.%20Histopathology-Specific%20Applications/Tellez,%20D.,%20Litjens,%20G.,%20Bándi,%20P.,%20et%20al.%20(2019).pdf)

### Statistical Methods

14. **McNemar, Q.** (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.

15. **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

16. **Efron, B., & Tibshirani, R. J.** (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.

---

## 10. License & Contact

**Project Type:** Master's Thesis / Academic Research

**Author:** Kurt Castro

**Institution:** [University Name]

**Year:** 2025

---

<div align="center">

**⭐ If this project was helpful, please consider starring the repository ⭐**

</div>
