# 🔍 Deep Learning Based Reverse Image Search System

> Using Feature Embeddings and FAISS Vector Similarity Search

<img width="388" height="492" alt="image" src="https://github.com/user-attachments/assets/812ecdd4-3de3-4492-9b64-1d7d50cc94c6" />

<img width="483" height="353" alt="image" src="https://github.com/user-attachments/assets/b3dba886-fd12-4e47-86d4-a5f7bb49b428" />

<img width="483" height="353" alt="image" src="https://github.com/user-attachments/assets/2f6082dd-d620-431c-bd4f-afffa2a10943" />


---

## 📌 Overview

A content-based reverse image search system that retrieves visually similar images from the **Caltech-101** dataset using **ResNet50 deep feature embeddings** and **FAISS vector similarity search** — no text labels required.

Upload any image → get the top-K most visually similar images back in under **1 second**.

---



---

## 🎯 Key Results

| Metric | Score |
|---|---|
| Accuracy | **87.1%** |
| Precision@10 | **0.87** |
| Mean Average Precision (MAP) | **0.83** |
| Mean Reciprocal Rank (MRR) | **0.97** |
| Top-5 Accuracy | **95.4%** |
| AUC-ROC | **0.9812** |
| End-to-End Query Time | **~0.8 sec** |

---

## 🏗️ System Architecture

```
User Uploads Image
       ↓
Image Preprocessing  (Resize → RGB → ImageNet Normalize)
       ↓
ResNet50 Feature Extraction  (pretrained on ImageNet, no top layer)
       ↓
2048-D Feature Embedding
       ↓
FAISS IndexFlatL2  (exact L2 nearest-neighbour search)
       ↓
Top-K Similarity Search
       ↓
Results Displayed in Gradio UI
```

---

## 📂 Dataset — Caltech-101

| Property | Details |
|---|---|
| Total Images | ~9,144 |
| Categories | 101 object classes + background |
| Avg. Images/Class | ~58 |
| Resolution | ~300 × 200 px |
| Format | JPEG |
| Split | 80% gallery / 20% query |

Sample categories: Airplanes, Butterfly, Camera, Elephant, Laptop, Motorbike, Stopwatch

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Feature Extractor | ResNet50 (Keras, pretrained ImageNet) |
| Vector Index | FAISS 1.7.4 — `IndexFlatL2` |
| Deep Learning Framework | TensorFlow 2.13 / Keras |
| Web Interface | Gradio 4.x |
| Image Processing | Pillow 10.x, OpenCV 4.8 |
| Language | Python 3.10 |

---

## ⚡ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/reverse-image-search.git
cd reverse-image-search
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Caltech-101 dataset

```bash
# Place the dataset under: data/caltech-101/
# Download from: http://www.vision.caltech.edu
```

### 4. Build the feature index (one-time, ~25 min on CPU)

```bash
python feature_extraction.py
```

This generates three persistent files:

```
features.npy    # embeddings matrix  [N × 2048]
paths.pkl       # image path list
index.faiss     # serialized FAISS index
```

### 5. Launch the Gradio app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`

---

## 📁 Project Structure

```
reverse-image-search/
│
├── data/
│   └── caltech-101/            # Dataset images
│
├── index/
│   ├── features.npy            # Precomputed embeddings (~72 MB)
│   ├── paths.pkl               # Image path index (~1 MB)
│   └── index.faiss             # FAISS index (~74 MB)
│
├── notebooks/
│   ├── feature_extraction.ipynb
│   ├── index_search.ipynb
│   ├── visualizations.ipynb
│   ├── performance_metrics.ipynb
│   └── real_comparison.ipynb
│
├── screenshots/                # UI screenshots for README
│
├── feature_extraction.py       # Offline indexing pipeline
├── index_search.py             # FAISS search module
├── app.py                      # Gradio web interface
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### Feature Extraction

ResNet50 is loaded without the classification head. Global Average Pooling converts the final convolutional feature map into a compact **2048-D vector** per image:

```python
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
embedding = model.predict(preprocessed_image)  # shape: (1, 2048)
```

### FAISS Indexing

All gallery embeddings are indexed for fast nearest-neighbour lookup:

```python
index = faiss.IndexFlatL2(2048)
index.add(features_matrix)          # shape: [N, 2048]
faiss.write_index(index, 'index.faiss')
```

### Similarity Search

At query time, the L2 distance between the query vector and all indexed vectors is minimized:

```
d(q, gᵢ) = √Σ (qⱼ − gᵢⱼ)²    for j = 1..2048
```

FAISS returns the K smallest distances in **< 1 ms** for 9,144 vectors.

---

## 📊 Model Comparison

| Method | Feature | P@5 | P@10 | MAP | Query Time |
|---|---|---|---|---|---|
| Color Histogram | Hand-crafted | 0.41 | 0.38 | 0.36 | < 1 ms |
| HOG Descriptor | Hand-crafted | 0.55 | 0.51 | 0.49 | < 1 ms |
| VGG16 + FAISS | Deep CNN | 0.88 | 0.84 | 0.80 | 1.1 s |
| **ResNet50 + FAISS (Ours)** | **Deep CNN** | **0.91** | **0.87** | **0.83** | **0.8 s** |

---

## 📈 Per-Category Performance

| Category | P@10 | Recall@10 | MAP |
|---|---|---|---|
| Airplanes | 0.97 | 0.14 | 0.95 |
| Stopwatch | 0.96 | 0.31 | 0.94 |
| Revolver | 0.95 | 0.28 | 0.93 |
| Butterfly | 0.93 | 0.22 | 0.91 |
| Camera | 0.91 | 0.26 | 0.89 |
| Elephant | 0.89 | 0.17 | 0.87 |
| Background Google | 0.71 | 0.09 | 0.68 |
| **Overall Average** | **0.87** | **0.11** | **0.83** |

---

## ⏱️ Runtime Analysis

| Operation | Time (CPU) |
|---|---|
| Feature extraction (per image) | ~180 ms |
| Full gallery indexing (~9,144 imgs) | ~25 min (one-time) |
| FAISS index build | < 5 sec |
| FAISS search (K=10) | < 1 ms |
| End-to-end query | **~0.8 sec** |

---

## 🔮 Future Work

- **Fine-tuned embeddings** — Train with triplet loss / contrastive loss for retrieval-optimized embeddings
- **IVF-PQ / HNSW indexing** — Scale to million-image datasets with approximate search
- **CLIP integration** — Enable cross-modal text-to-image and image-to-text search
- **GPU FAISS** — Reduce query time from 0.8 s to < 0.1 s
- **Mobile deployment** — Export to TensorFlow Lite for on-device search

---

## 📚 References

1. He et al. (2016) — Deep Residual Learning for Image Recognition. *CVPR*
2. Johnson et al. (2019) — Billion-scale Similarity Search with GPUs. *IEEE Transactions on Big Data*
3. Babenko et al. (2014) — Neural Codes for Image Retrieval. *ECCV*
4. Fei-Fei et al. (2004) — Caltech-101 Dataset. *CVPR Workshop*
5. Jegou et al. (2011) — Product Quantization for Nearest Neighbor Search. *IEEE TPAMI*

---

## 📄 License

This project is for academic and non-commercial research use only, in accordance with the Caltech-101 dataset license.

---

*Year: 2025 · Dataset: Caltech-101 · Model: ResNet50 · Index: FAISS*
