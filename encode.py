# Utilities to encode images and texts with CLIP (transformers)
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = model.to(device)
    return model, processor

def encode_image(model, processor, image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
    # L2 normalize
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds.cpu()

def encode_texts(model, processor, texts):
    inputs = processor(text=texts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    return text_embeds.cpu()
