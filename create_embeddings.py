"""
generate_shapes_dataset.py
Creates a synthetic dataset of colored geometric shapes with variation
for CLIP embedding experiments.

Outputs:
    - static/data/images/<shape>/<shape>_<color>_<index>.png
    - static/data/captions.csv  (columns: image_path, text)
"""

import os
import csv
import math
import random
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter

# =======================
# CONFIGURATION
# =======================
output_dir = os.path.join("static", "data", "images")
csv_path = os.path.join("static", "data", "captions.csv")

shapes = ["circle", "square", "triangle", "pentagon"]
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
}
num_variations = 10  # more variations for better coverage
image_size = 256

# Random background options (light neutral colors)
backgrounds = [
    (240, 240, 240), (250, 245, 255), (245, 250, 240), (255, 250, 245)
]

# =======================
# GENERATE IMAGES
# =======================
os.makedirs(output_dir, exist_ok=True)
rows = []

for shape in shapes:
    shape_dir = os.path.join(output_dir, shape)
    os.makedirs(shape_dir, exist_ok=True)

    for color_name, rgb in colors.items():
        for i in range(num_variations):
            bg_color = random.choice(backgrounds)
            img = Image.new("RGB", (image_size, image_size), bg_color)
            draw = ImageDraw.Draw(img)

            # Random scale and position
            scale = random.uniform(0.6, 0.9)
            margin = int((1 - scale) * image_size / 2)
            bbox = [margin, margin, image_size - margin, image_size - margin]

            # Draw shape
            if shape == "circle":
                draw.ellipse(bbox, fill=rgb)
            elif shape == "square":
                draw.rectangle(bbox, fill=rgb)
            elif shape == "triangle":
                draw.polygon(
                    [(image_size / 2, margin),
                     (margin, image_size - margin),
                     (image_size - margin, image_size - margin)],
                    fill=rgb
                )
            elif shape == "pentagon":
                r = (image_size - 2 * margin) / 2
                cx, cy = image_size / 2, image_size / 2
                points = [
                    (cx + r * math.cos(2 * math.pi * k / 5 - math.pi / 2),
                     cy + r * math.sin(2 * math.pi * k / 5 - math.pi / 2))
                    for k in range(5)
                ]
                draw.polygon(points, fill=rgb)

            # Apply random rotation
            angle = random.randint(0, 359)
            img = img.rotate(angle, expand=False, fillcolor=bg_color)

            # Add mild noise, brightness variation, blur, or contrast jitter
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.9, 1.1))

            if random.random() < 0.3:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))

            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            if random.random() < 0.3:
                # Add a bit of pixel noise
                noise = Image.effect_noise((image_size, image_size), random.uniform(2, 8))
                img = Image.blend(img, noise.convert("RGB"), alpha=0.05)

            # Save image
            filename = f"{shape}_{color_name}_{i}.png"
            image_path = os.path.join(shape_dir, filename)
            img.save(image_path)

            # Caption
            text = f"{color_name} {shape}"
            rows.append({"image_path": image_path, "text": text})

# =======================
# SAVE CAPTIONS CSV
# =======================
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image_path", "text"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Generated {len(rows)} images across {len(colors)} colors × {len(shapes)} shapes.")
print(f"📄 Captions saved to: {csv_path}")
