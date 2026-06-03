import os
import numpy as np
import pickle
from PIL import Image
import faiss

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# =========================
# SETTINGS
# =========================
DATASET_PATH = "caltech-101/101_ObjectCategories"


# =========================
# LOAD MODEL
# =========================
print("Loading ResNet50...")
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


# =========================
# EXTRACT FEATURES
# =========================
features = []
paths = []

print("Processing images...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            img_path = os.path.join(root, file)

            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                img = np.array(img)

                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)

                feat = model.predict(img, verbose=0)[0]

                # Normalize vector
                feat = feat / np.linalg.norm(feat)

                features.append(feat)
                paths.append(img_path)

            except Exception as e:
                print("Error processing:", img_path, e)


features = np.array(features).astype("float32")

print("\nDone!")
print("Feature shape:", features.shape)


# =========================
# SAFETY CHECK
# =========================
if len(features) == 0:
    print("❌ No images found. Check dataset path.")
    exit()


# =========================
# SAVE FILES
# =========================
print("Saving files...")

np.save("features.npy", features)

with open("paths.pkl", "wb") as f:
    pickle.dump(paths, f)


# =========================
# CREATE FAISS INDEX
# =========================
print("Creating FAISS index...")

dimension = features.shape[1]
index = faiss.IndexFlatIP(dimension)   # cosine similarity
index.add(features)

faiss.write_index(index, "index.faiss")


# =========================
# DONE
# =========================
print("\n✅ ALL FILES CREATED SUCCESSFULLY:")
print("features.npy")
print("paths.pkl")
print("index.faiss")