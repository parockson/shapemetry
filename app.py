from flask import Flask, render_template, request, jsonify
import csv
import os
import json
import torch
import open_clip
from PIL import Image

app = Flask(__name__, static_folder='static')

CAPTIONS_PATH = os.path.join("static", "data", "captions.csv")

# ==========================
# Load OpenCLIP model
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)
model.eval()


# ==========================
# Load dataset
# ==========================
def load_dataset():
    """Load image paths and captions from CSV."""
    data = []
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rel_path = row["image_path"].replace("\\", "/")
                if rel_path.startswith("static/"):
                    rel_path = rel_path[len("static/"):]
                data.append({
                    "image_path": rel_path,
                    "text": row["text"]
                })
    return data


# ==========================
# Routes
# ==========================
@app.route('/')
def index():
    items = load_dataset()
    return render_template('index.html', items=items)


@app.route('/encode', methods=['GET'])
def encode():
    """Encoder page showing selected items and CLIP similarity results."""
    selected = request.args.get('data')
    selected_data = json.loads(selected) if selected else {"images": [], "texts": []}

    similarities = []
    if selected_data["images"] and selected_data["texts"]:
        image_embeddings = []

        for img_path in selected_data["images"]:
            img_file = os.path.join(app.static_folder, img_path.replace("/static/", "").replace("static/", ""))
            image = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                img_emb = model.encode_image(image)
                image_embeddings.append(img_emb)

        image_embeddings = torch.cat(image_embeddings, dim=0)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

        # Tokenize and encode text
        text_tokens = tokenizer(selected_data["texts"])
        text_tokens = text_tokens.to(device)

        with torch.no_grad():
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        similarity_matrix = (100.0 * image_embeddings @ text_embeddings.T).softmax(dim=-1)

        for i, img_path in enumerate(selected_data["images"]):
            for j, txt in enumerate(selected_data["texts"]):
                similarities.append({
                    "image": img_path,
                    "text": txt,
                    "score": round(similarity_matrix[i][j].item(), 3)
                })

    return render_template('encode.html', selected_data=selected_data, similarities=similarities)


@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')


if __name__ == "__main__":
    app.run(debug=True)
