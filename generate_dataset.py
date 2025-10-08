"""
generate_clip_dataset.py
Generates one unique image per (shape, color) pair for CLIP-style datasets.
Usage:
    python generate_clip_dataset.py
"""

import os
import csv
from PIL import Image, ImageDraw

# =======================
# CONFIGURATION
# =======================
shapes = ["circle", "square", "triangle", "pentagon"]
colors = ["red", "green", "blue", "yellow", "orange", "purple"]
img_size = 128  # image width/height in pixels

# Output paths
base_dir = os.path.join("static", "data", "images")
os.makedirs(base_dir, exist_ok=True)

csv_path = os.path.join("static", "data", "captions.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# =======================
# GENERATE DATA
# =======================
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "text"])

    total = 0
    for shape in shapes:
        shape_dir = os.path.join(base_dir, shape)
        os.makedirs(shape_dir, exist_ok=True)

        for color in colors:
            img = Image.new("RGBA", (img_size, img_size), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)

            # Draw shape
            if shape == "circle":
                draw.ellipse((20, 20, img_size - 20, img_size - 20), fill=color)
            elif shape == "square":
                draw.rectangle((20, 20, img_size - 20, img_size - 20), fill=color)
            elif shape == "triangle":
                draw.polygon(
                    [(img_size / 2, 20), (img_size - 20, img_size - 20), (20, img_size - 20)],
                    fill=color
                )
            elif shape == "pentagon":
                draw.polygon(
                    [
                        (img_size / 2, 10),
                        (img_size - 18, 50),
                        (img_size - 38, img_size - 18),
                        (38, img_size - 18),
                        (18, 50)
                    ],
                    fill=color
                )

            # Save image
            filename = f"{shape}_{color}.png"
            filepath = os.path.join(shape_dir, filename)
            img.save(filepath)

            # Write caption
            caption = f"{color} {shape}"
            writer.writerow([filepath.replace("\\", "/"), caption])
            total += 1

print(f"✅ Generated {total} unique images (1 per shape–color pair)")
print(f"🖼️  Images in: {base_dir}")
print(f"📝 Captions: {csv_path}")
