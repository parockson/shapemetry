import streamlit as st
import pandas as pd
import os
from PIL import Image

# ==========================
# Configuration
# ==========================
st.set_page_config(page_title="CLIP Encoder Viewer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "data")
EMBEDDINGS_PATH = os.path.join(STATIC_DIR, "clip_embeddings_2d.csv")

# ==========================
# Load Dataset
# ==========================
@st.cache_data
def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        df = pd.read_csv(EMBEDDINGS_PATH)
        df["image_path"] = df["image_path"].apply(lambda x: x.replace("\\", "/"))
        return df
    else:
        st.error(f"Embeddings file not found: {EMBEDDINGS_PATH}")
        return pd.DataFrame(columns=[
            "image_path", "text", "similarity_512d", "similarity_2d",
            "img_x", "img_y", "text_x", "text_y"
        ])

data = load_embeddings()

# ==========================
# Streamlit UI
# ==========================
st.title("🧠 CLIP Encoder Viewer")
st.write("View precomputed 2D CLIP embeddings for selected images and texts.")

if data.empty:
    st.stop()

# ==========================
# Image Grid Selection
# ==========================
st.subheader("🖼️ Select Images")
selected_images = st.session_state.get("selected_images", set())

image_files = data["image_path"].unique().tolist()
num_cols = 6
rows = [image_files[i:i + num_cols] for i in range(0, len(image_files), num_cols)]

for row in rows:
    cols = st.columns(num_cols)
    for col, img_path in zip(cols, row):
        img_full_path = os.path.join(BASE_DIR, img_path)
        img_full_path = os.path.normpath(img_full_path)

        if os.path.exists(img_full_path):
            col.image(img_full_path, use_container_width=True)
            img_name = os.path.basename(img_path)

            if col.checkbox(f"{img_name}", key=img_path):
                selected_images.add(img_path)
            elif img_path in selected_images:
                selected_images.remove(img_path)
        else:
            col.warning("Missing image")

st.session_state["selected_images"] = selected_images

# ==========================
# Text Selection
# ==========================
st.subheader("✍️ Select Text Captions")
text_options = data["text"].unique().tolist()
selected_texts = st.multiselect("Select text captions:", text_options)

# ==========================
# Display Encoders
# ==========================
if selected_images or selected_texts:
    st.subheader("🔢 Encoders")

    if selected_images:
        img_df = data[data["image_path"].isin(selected_images)][
            ["image_path", "img_x", "img_y"]
        ].rename(columns={"image_path": "Image", "img_x": "x", "img_y": "y"})
        st.markdown("#### 🖼️ Selected Image Encoders")
        st.dataframe(img_df, hide_index=True, use_container_width=True)

    if selected_texts:
        text_df = data[data["text"].isin(selected_texts)][
            ["text", "text_x", "text_y"]
        ].rename(columns={"text": "Text", "text_x": "x", "text_y": "y"})
        st.markdown("#### ✍️ Selected Text Encoders")
        st.dataframe(text_df, hide_index=True, use_container_width=True)

else:
    st.info("Select some images and/or texts to view their encoders.")
