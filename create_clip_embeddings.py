"""
create_clip_embeddings.py
Generates CLIP embeddings (vectors) for your geometric shapes dataset,
and reduces them to 2D for visualization or clustering.

Outputs:
    - clip_embeddings.pt : Full 512D embeddings (PyTorch tensors)
    - clip_embeddings_2d.csv : 2D embeddings + cosine similarity

Requires:
    pip install open_clip_torch torch torchvision pillow tqdm pandas scikit-learn
"""

import torch
import open_clip
from PIL import Image
import csv
from tqdm import tqdm
import os
import pandas as pd
from sklearn.decomposition import PCA
from numpy import dot
from numpy.linalg import norm

# =======================
# CONFIGURATION
# =======================
csv_path = os.path.join("static", "data", "captions.csv")
save_pt_path = os.path.join("static", "data", "clip_embeddings.pt")
save_csv_path = os.path.join("static", "data", "clip_embeddings_2d.csv")

model_name = "ViT-B-32"
pretrained = "openai"

# =======================
# LOAD MODEL
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Loading CLIP model ({model_name}, pretrained={pretrained}) on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device).eval()

# =======================
# ENCODE IMAGES + TEXT
# =======================
image_features = []
text_features = []
image_paths = []
texts = []
similarities = []

print("🎨 Encoding images and text...")

with open(csv_path, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader, desc="Processing"):
        image_path = row["image_path"]
        text = row["text"]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Tokenize text
        text_input = tokenizer([text]).to(device)

        with torch.no_grad():
            image_emb = model.encode_image(image_input)
            text_emb = model.encode_text(text_input)

        # Normalize embeddings
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (scalar)
        sim = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()

        image_features.append(image_emb.cpu())
        text_features.append(text_emb.cpu())
        image_paths.append(image_path)
        texts.append(text)
        similarities.append(sim)

# Stack all embeddings
image_features = torch.cat(image_features)
text_features = torch.cat(text_features)

# =======================
# SAVE FULL 512D .PT FILE
# =======================
torch.save({
    "image_embeddings": image_features,
    "text_embeddings": text_features,
    "image_paths": image_paths,
    "texts": texts,
    "similarities": similarities
}, save_pt_path)

print(f"✅ Saved 512D tensor embeddings: {save_pt_path}")

# =======================
# REDUCE TO 2D SPACE (PCA)
# =======================
print("📉 Reducing embeddings to 2D space using PCA...")
pca = PCA(n_components=2)
image_2d = pca.fit_transform(image_features.numpy())
text_2d = pca.fit_transform(text_features.numpy())

# Compute cosine similarities in 2D
similarities_2d = [
    dot(image_2d[i], text_2d[i]) / (norm(image_2d[i]) * norm(text_2d[i]))
    for i in range(len(image_2d))
]

# =======================
# SAVE 2D EMBEDDINGS AS CSV
# =======================
print("🧾 Saving 2D embeddings to CSV...")

rows = []
for i in range(len(image_paths)):
    rows.append({
        "image_path": image_paths[i],
        "text": texts[i],
        "similarity_512d": similarities[i],
        "similarity_2d": similarities_2d[i],
        "img_x": image_2d[i, 0],
        "img_y": image_2d[i, 1],
        "text_x": text_2d[i, 0],
        "text_y": text_2d[i, 1]
    })

df = pd.DataFrame(rows)
df.to_csv(save_csv_path, index=False)

# =======================
# SUMMARY
# =======================
print("✅ Embedding generation complete!")
print(f"   - 🧠 512D tensor file: {save_pt_path}")
print(f"   - 📄 2D CSV file: {save_csv_path}")
print(f"📊 Total samples encoded: {len(image_paths)}")
print(f"💡 Avg similarity (512D): {sum(similarities)/len(similarities):.4f}")
print(f"💡 Avg similarity (2D): {sum(similarities_2d)/len(similarities_2d):.4f}")
