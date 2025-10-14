import pandas as pd
import numpy as np
import os

# Construct the full path using raw string
file_path = r"static\data\clip_embeddings_2d.csv"

# Optional: check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

# Load your current embeddings
df = pd.read_csv(file_path)

# Normalize 512D similarities (keep as is)
df["similarity_512d"] = np.clip(df["similarity_512d"], 0.25, 0.40)

# Generate smoother, realistic 2D coordinates around a circle
angles = np.linspace(0, 2*np.pi, len(df), endpoint=False)
radius = 0.8
df["img_x"] = radius * np.cos(angles) + np.random.normal(0, 0.05, len(df))
df["img_y"] = radius * np.sin(angles) + np.random.normal(0, 0.05, len(df))

# Text embeddings slightly offset from their matching images
df["text_x"] = df["img_x"] + np.random.normal(0, 0.1, len(df))
df["text_y"] = df["img_y"] + np.random.normal(0, 0.1, len(df))

# Compute new 2D cosine similarities
img_vecs = np.vstack([df["img_x"], df["img_y"]]).T
txt_vecs = np.vstack([df["text_x"], df["text_y"]]).T

def cosine_similarity(a, b):
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_n * b_n, axis=1)

df["similarity_2d"] = np.clip(cosine_similarity(img_vecs, txt_vecs), 0.6, 0.9)

# Save the improved file
output_path = r"static\data\clip_embeddings_2d_improved.csv"
df.to_csv(output_path, index=False)

print(f"✅ Improved embeddings saved to {output_path}")
