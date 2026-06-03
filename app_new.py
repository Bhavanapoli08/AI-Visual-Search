import numpy as np
import pickle
import faiss
from PIL import Image
import gradio as gr

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# =========================
# LOAD MODEL
# =========================
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# =========================
# LOAD DATA
# =========================
features = np.load("features.npy")

with open("paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

index = faiss.read_index("index.faiss")


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(img):
    img = img.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feat = model.predict(img)[0]
    feat = feat / np.linalg.norm(feat)

    return feat.reshape(1, -1)


# =========================
# SEARCH FUNCTION
# =========================
def search_image(query_img):
    if query_img is None:
        return [None] * 5

    query_feat = extract_features(query_img)

    D, I = index.search(query_feat, 5)

    results = [image_paths[i] for i in I[0]]

    return results


# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 AI Reverse Image Search")

    query = gr.Image(type="pil", label="Upload Image")

    btn = gr.Button("Search")

    outputs = [gr.Image() for _ in range(5)]

    btn.click(fn=search_image, inputs=query, outputs=outputs)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    demo.launch(share=True)