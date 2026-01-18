# 🚀 RLDA-ResNet: CNN Acceleration using Lookup Tables

This project implements **RLDA (Residual Lookup-based Dot-product Approximation)** on **ResNet-18** to reduce runtime **Multiply–Accumulate (MAC)** operations in convolutional neural networks while maintaining competitive accuracy on the **CIFAR-10** dataset.

Instead of performing expensive multiplications during inference, the model uses **precomputed lookup tables (LUTs)** and **nearest-centroid search**, making it suitable for **hardware-efficient and low-power systems**.

---

## 📌 Project Highlights

- ✅ Baseline **ResNet-18** trained on CIFAR-10  
- ✅ LUT generation using **K-Means clustering (K = 32)**  
- ✅ Replacement of `Conv2d` with **RLDAConv** (lookup + add operations)  
- ✅ Fine-tuning RLDA-ResNet to recover accuracy  
- ✅ Evaluation using **Accuracy, Precision, Recall, F1-Score**  
- ✅ Confusion Matrix with **TP, TN, FP, FN**  

---

## 🧠 Core Idea

### Original ResNet
```

Convolution = weight × input  → millions of MAC operations

```

### RLDA-ResNet
```

input patch → nearest centroid → LUT lookup → addition

````

✔ Multiplications are **precomputed offline**  
✔ Inference uses **lookup + add**, not multiply  
✔ Trade-off: small accuracy drop for large compute savings  

---

## 📂 Project Structure

```text
.
├── models/
│   ├── rlda_conv.py         # RLDAConv: LUT-based convolution replacing Conv2d
│   ├── rlda_resnet.py       # RLDA-ResNet18 architecture (paper-based implementation)
│   └── original_resnet.py   # Baseline ResNet-18 for accuracy comparison
│
├── utils/
│   ├── luts.py              # K-Means clustering and LUT (codebook) generation
│   ├── inspect_model.py     # Utility to inspect model layers, weights, and shapes
│   └── check_gpu.py         # GPU / device environment verification
│
├── scripts/
│   ├── train_resnet.py      # Train baseline ResNet-18 on CIFAR-10
│   ├── train_rlda.py        # Train / fine-tune RLDA-ResNet model
│   └── confusion_matrix.py # Evaluation, metrics, and visualization
│
├── data/
│   └── cifar-10-batches-py/ # CIFAR-10 dataset
│
├── trained_resnet18_cifar10.pth  # Baseline trained ResNet weights
├── best_rlda_resnet18.pth        # Best RLDA-ResNet checkpoint
├── lut_layer*_conv*.pth          # Generated LUT files (K = 32 centroids)
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
````

---

## 🧪 Dataset

**CIFAR-10**

* 60,000 images (32×32 RGB)
* 10 classes
* 50,000 training images
* 10,000 testing images

---

## ⚙️ Requirements

* Python ≥ 3.9
* PyTorch
* torchvision
* numpy
* matplotlib
* seaborn
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run (Step-by-Step)

### 1️⃣ Train Baseline ResNet-18

```bash
python scripts/train_resnet.py
```

Output:

```
trained_resnet18_cifar10.pth
```

---

### 2️⃣ Generate Lookup Tables (LUTs)

```bash
python utils/luts.py
```

Output:

```
lut_layer*_conv*.pth   (K = 32 centroids per convolution layer)
```

---

### 3️⃣ Train / Fine-Tune RLDA-ResNet

```bash
python scripts/train_rlda.py
```

Output:

```
best_rlda_resnet18.pth
```

---

### 4️⃣ Evaluate Model & Plot Confusion Matrix

```bash
python scripts/confusion_matrix.py
```

Outputs:

* Confusion Matrix
* Overall Accuracy
* Precision, Recall, F1-Score
* TP, TN, FP, FN (per class)

---

## 📊 Metrics Used

From the confusion matrix, the following metrics are computed:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **True Positive (TP)**
* **True Negative (TN)**
* **False Positive (FP)**
* **False Negative (FN)**

Metrics are reported **per class and overall**.

---

## 🔬 Technical Details

* **Clustering Method:** K-Means
* **Number of Centroids (K):** 32
* **Nearest Search:** 1-nearest centroid (not KNN classifier)
* **Distance Metric:** L1 distance (`torch.cdist`)
* **RLDAConv Parameters:**

  * centroids
  * dot_centroids
  * residual_centroids
* **Training:**

  * Optimizer: SGD
  * Loss: CrossEntropyLoss
  * Fine-tuning after approximation

---

## 📉 Accuracy Trade-off

| Model              | Accuracy       | Computation             |
| ------------------ | -------------- | ----------------------- |
| Original ResNet-18 | Higher         | Full MAC operations     |
| RLDA-ResNet-18     | Slightly Lower | LUT-based (MAC-reduced) |

The accuracy drop is **expected** due to approximation and is a known trade-off in hardware-aware deep learning.

---

## 🎓 Academic Relevance

This project demonstrates:

* CNN acceleration techniques
* Approximate computing
* Lookup-table-based inference
* Accuracy vs efficiency trade-offs

Suitable for:

* Major Project
* Research-oriented coursework
* Hardware-aware machine learning exploration

---

## 🔮 Future Improvements

* Increase K (e.g., 64 centroids)
* Apply RLDA only to deeper layers
* Quantized LUTs
* FPGA / Edge deployment
* Benchmark MAC reduction and latency

---

## 👨‍💻 Author

**Sai Chetan Barathala**
Electronics & Communication Engineering
Major Project – Deep Learning & CNN Acceleration



