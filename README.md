# CLIP Geometric Shapes — Zero-shot Image-Text Retrieval Demo

## Overview
This project demonstrates a CLIP-style embedding workflow for simple geometric shapes.
It includes:
- dataset generation (images + text pairs)
- encoding & similarity computation using Hugging Face's CLIPModel
- a lightweight Flask web app to select images/text and view similarity results
- a training script scaffold for contrastive fine-tuning (optional)
- visualization (embedding space via PCA)

**Important:** This repository does NOT include the pretrained CLIP model weights.
The first run will download them (internet required).

## Requirements
Create a Python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Quick start
1. Generate the dataset:
   ```bash
   python generate_dataset.py --out_dir data/images --num_per_class 10
   ```
2. (Optional) Train/fine-tune (this will download CLIP and may take time):
   ```bash
   python train.py --data_dir data/images --epochs 3 --batch_size 8
   ```
3. Run the web app:
   ```bash
   export FLASK_APP=app.py
   flask run
   ```
   Then open http://127.0.0.1:5000

## Structure
- `generate_dataset.py` — synthetic image generator (PIL)
- `encode.py` — utilities to encode images and texts using CLIP (transformers)
- `train.py` — simple fine-tune script scaffold (contrastive loss)
- `app.py` — Flask app with pages: select, encoder, embedding space, similarity
- `static/` — holds generated images (after running generator)
- `templates/` — html templates for the Flask app

## Notes
- The training script is intentionally minimal to keep it runnable. For large-scale training, use proper dataloaders, distributed training, and checkpointing.
- If you want a pure browser app without Flask, ask and I'll adapt.

Enjoy — run the generator then the app and tell me what you'd like changed!

# shapemetry