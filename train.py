# Minimal contrastive fine-tuning scaffold for CLIP.
# WARNING: This will download CLIP weights and is meant as an educational starting point.
import os, argparse, random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from encode import load_model
from transformers import CLIPProcessor, CLIPModel, AdamW

class ShapeDataset(Dataset):
    def __init__(self, root_dir, processor, image_size=224):
        self.samples = []
        self.processor = processor
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir): continue
            for f in os.listdir(label_dir):
                if f.lower().endswith('.png'):
                    self.samples.append((os.path.join(label_dir,f), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        return image, label, path

def collate_fn(batch):
    images, labels, paths = zip(*batch)
    return images, list(labels), list(paths)

def main(data_dir, epochs=3, batch_size=8, lr=1e-5):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    ds = ShapeDataset(data_dir, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Build simple label-to-text map
    labels = sorted({lbl for _,lbl in ds.samples})
    texts = [f"a photo of a {lbl}" for lbl in labels]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, lbls, paths in dl:
            inputs = processor(images=images, text=[f'a photo of a {l}' for l in lbls], return_tensors='pt', padding=True).to(device)
            outputs = model(**inputs)
            # outputs.logits_per_image: image-text similarity scaled by logit scale
            # Use standard CLIP cross entropy: labels are diagonal
            logits_per_image = outputs.logits_per_image
            targets = torch.arange(len(logits_per_image), device=device)
            loss = torch.nn.CrossEntropyLoss()(logits_per_image, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} -- loss {total_loss/len(dl):.4f}')
    # Save a checkpoint
    out_path = 'checkpoints/clip_finetuned.pth'
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print('Saved checkpoint to', out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/images')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size)
