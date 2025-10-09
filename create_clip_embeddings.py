"""
create_clip_embeddings.py
Generates CLIP embeddings (vectors) for your geometric shapes dataset.

Outputs:
    - clip_embeddings.pt : PyTorch tensors
    - clip_embeddings.csv : CSV with vectors + cosine similarity

Requires:
    pip install open_clip_torch torch torchvision pillow tqdm pandas
"""

import torch
import open_clip
from PIL import Image
import csv
from tqdm import tqdm
import os
import pandas as pd

# =======================
# CONFIGURATION
# =======================
csv_path = os.path.join("static", "data", "captions.csv")
save_pt_path = os.path.join("static", "data", "clip_embeddings.pt")
save_csv_path = os.path.join("static", "data", "clip_embeddings.csv")

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
# SAVE AS .PT
# =======================
torch.save({
    "image_embeddings": image_features,
    "text_embeddings": text_features,
    "image_paths": image_paths,
    "texts": texts,
    "similarities": similarities
}, save_pt_path)

# =======================
# SAVE AS .CSV
# =======================
print("🧾 Saving embeddings to CSV (this may take a minute)...")

# Convert embeddings to numpy
image_np = image_features.numpy()
text_np = text_features.numpy()

rows = []
for i in range(len(image_paths)):
    row = {
        "image_path": image_paths[i],
        "text": texts[i],
        "similarity": similarities[i],
        **{f"img_vec_{j}": image_np[i, j] for j in range(image_np.shape[1])},
        **{f"text_vec_{j}": text_np[i, j] for j in range(text_np.shape[1])}
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(save_csv_path, index=False)

# =======================
# SUMMARY
# =======================
print(f"✅ Saved CLIP embeddings:")
print(f"   - 🧠 Tensor file: {save_pt_path}")
print(f"   - 📄 CSV file: {save_csv_path}")
print(f"📊 Total samples encoded: {len(image_paths)}")
print(f"💡 Average similarity: {sum(similarities)/len(similarities):.4f}")
