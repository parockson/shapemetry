import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---- Page setup ----
st.set_page_config(page_title="CLIP Encoder Selector", layout="wide")
st.title("🧠 CLIP Encoder Selection Interface")

# ---- Load Data ----
csv_path = "static/data/clip_embeddings_2d.csv"
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# ---- Group by images and texts ----
image_paths = df['image_path'].unique()
texts = df['text'].unique()

# ---- Session state ----
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "selected_texts" not in st.session_state:
    st.session_state.selected_texts = []

# ---- Image Selection ----
st.subheader("🖼️ Select up to 4 Images")
cols = st.columns(6)
for idx, image_path in enumerate(image_paths):
    col = cols[idx % 6]
    with col:
        if os.path.exists(image_path):
            st.image(image_path, width='stretch')
        else:
            st.warning(f"Missing: {image_path.split('/')[-1]}")

        label = os.path.basename(image_path)
        if label in st.session_state.selected_images:
            if st.button("✅ Deselect", key=f"img_{idx}"):
                st.session_state.selected_images.remove(label)
        else:
            if len(st.session_state.selected_images) < 4:
                if st.button("Select", key=f"img_{idx}"):
                    st.session_state.selected_images.append(label)
            else:
                st.button("Limit reached", key=f"img_disabled_{idx}", disabled=True)

# ---- Text Selection ----
st.subheader("💬 Select up to 4 Texts")
text_cols = st.columns(4)
for i, text_item in enumerate(texts):
    col = text_cols[i % 4]
    with col:
        if text_item in st.session_state.selected_texts:
            if st.button("✅ Deselect", key=f"text_{i}"):
                st.session_state.selected_texts.remove(text_item)
        else:
            if len(st.session_state.selected_texts) < 4:
                if st.button(f"Select: {text_item}", key=f"text_btn_{i}"):
                    st.session_state.selected_texts.append(text_item)
            else:
                st.button("Limit reached", key=f"text_disabled_{i}", disabled=True)

# ---- Display Selected Encoders ----
st.markdown("---")
st.subheader("📊 Selected Encoders")

# --- Image Encoders Table ---
if st.session_state.selected_images:
    selected_img_data = df[df["image_path"].str.contains('|'.join(st.session_state.selected_images))]
    st.dataframe(selected_img_data[["image_path", "img_x", "img_y"]], width='stretch')
else:
    st.info("No images selected yet.")

# --- Text Encoders Table ---
if st.session_state.selected_texts:
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]
    st.dataframe(selected_text_data[["text", "text_x", "text_y"]], width='stretch')
else:
    st.info("No texts selected yet.")

# ---- Scatterplots Side-by-Side ----
if st.session_state.selected_images or st.session_state.selected_texts:
    st.markdown("---")
    st.subheader("🔎 2D Encoder Spaces")

    col1, col2 = st.columns(2)

    # --- Image Encoder Plot ---
    with col1:
        if st.session_state.selected_images:
            fig_img, ax_img = plt.subplots()
            ax_img.scatter(selected_img_data["img_x"], selected_img_data["img_y"], s=80)
            for _, row in selected_img_data.iterrows():
                ax_img.text(row["img_x"], row["img_y"], os.path.basename(row["image_path"]), fontsize=8, ha='right')
            ax_img.set_title("🖼️ Image Encoder Space (2D)")
            ax_img.set_xlabel("img_x")
            ax_img.set_ylabel("img_y")
            st.pyplot(fig_img, use_container_width=True)
        else:
            st.info("No images selected for plotting.")

    # --- Text Encoder Plot ---
    with col2:
        if st.session_state.selected_texts:
            fig_txt, ax_txt = plt.subplots()
            ax_txt.scatter(selected_text_data["text_x"], selected_text_data["text_y"], color='orange', s=80)
            for _, row in selected_text_data.iterrows():
                ax_txt.text(row["text_x"], row["text_y"], row["text"], fontsize=8, ha='right')
            ax_txt.set_title("💬 Text Encoder Space (2D)")
            ax_txt.set_xlabel("text_x")
            ax_txt.set_ylabel("text_y")
            st.pyplot(fig_txt, use_container_width=True)
        else:
            st.info("No texts selected for plotting.")

# ---- Common Embedding Space ----
if st.session_state.selected_images and st.session_state.selected_texts:
    st.markdown("---")
    st.subheader("🌐 Common Embedding Space")

    # Prepare coordinates
    img_coords = selected_img_data[["img_x", "img_y"]].to_numpy()
    text_coords = selected_text_data[["text_x", "text_y"]].to_numpy()
    
    # Compute Euclidean distance matrix (lower = more similar)
    distances = np.zeros((len(selected_img_data), len(selected_text_data)))
    for i, img_vec in enumerate(img_coords):
        for j, text_vec in enumerate(text_coords):
            distances[i, j] = np.linalg.norm(img_vec - text_vec)
    
    # Build similarity table
    sim_table = pd.DataFrame(distances, 
                             index=[os.path.basename(p) for p in st.session_state.selected_images],
                             columns=st.session_state.selected_texts)
    st.markdown("**Euclidean Distance (Lower = More Similar)**")
    st.dataframe(sim_table)

    # Combined Scatterplot
    fig, ax = plt.subplots()
    ax.scatter(img_coords[:,0], img_coords[:,1], s=100, c='blue', label='Images')
    ax.scatter(text_coords[:,0], text_coords[:,1], s=100, c='orange', label='Texts')
    
    # Annotate points
    for idx, row in selected_img_data.iterrows():
        ax.text(row["img_x"], row["img_y"], os.path.basename(row["image_path"]), fontsize=8, ha='right')
    for idx, row in selected_text_data.iterrows():
        ax.text(row["text_x"], row["text_y"], row["text"], fontsize=8, ha='right')
    
    ax.set_title("Combined 2D Embedding Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
