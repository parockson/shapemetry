import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

# ---- Page setup ----
st.set_page_config(page_title="CLIP Encoder Stepwise Interface", layout="wide")
st.title("🧠 CLIP Encoder Stepwise Interface")

# ---- Load Data ----
csv_path = "static/data/clip_embeddings_2d_improved.csv"
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)
image_paths = df['image_path'].unique()
texts = df['text'].unique()

# ---- Session state ----
if "step" not in st.session_state:
    st.session_state.step = 1
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "selected_texts" not in st.session_state:
    st.session_state.selected_texts = []

st.sidebar.title("Steps")
st.sidebar.text(f"Current Step: {st.session_state.step}")

# ========================================
# STEP 1 — SELECT IMAGES AND TEXTS TOGETHER
# ========================================
if st.session_state.step == 1:
    st.subheader("Step 1: Select up to 4 Images and 4 Texts")

    col_img, col_txt = st.columns([2, 2])

    # ---- Image Selection (Left) ----
    with col_img:
        st.markdown("### 🖼️ Select Images")
        cols = st.columns(4)
        for idx, image_path in enumerate(image_paths):
            col = cols[idx % 4]
            with col:
                if os.path.exists(image_path):
                    st.image(image_path, width=100)
                else:
                    st.warning(f"Missing: {os.path.basename(image_path)}")
                label = os.path.basename(image_path)
                if label in st.session_state.selected_images:
                    if st.button("✅ Deselect", key=f"img_{idx}"):
                        st.session_state.selected_images.remove(label)
                else:
                    if len(st.session_state.selected_images) < 4:
                        if st.button("Select", key=f"img_sel_{idx}"):
                            st.session_state.selected_images.append(label)
                    else:
                        st.button("Limit reached", key=f"img_disabled_{idx}", disabled=True)

    # ---- Text Selection (Right) ----
    with col_txt:
        st.markdown("### 📝 Select Texts")
        txt_cols = st.columns(2)
        for i, text_item in enumerate(texts):
            col = txt_cols[i % 2]
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

    # ---- Continue Button ----
    if st.session_state.selected_images and st.session_state.selected_texts:
        st.markdown("---")
        if st.button("Next: Compute Similarity ➡️"):
            st.session_state.step = 2
            st.rerun()

# ========================================
# STEP 2 — SIMILARITY COMPUTATION
# ========================================
elif st.session_state.step == 2:
    st.subheader("Step 2: Compute Similarities")
    selected_img_data = df[df["image_path"].str.contains('|'.join(st.session_state.selected_images))]
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]

    img_coords = selected_img_data[["img_x", "img_y"]].to_numpy()
    text_coords = selected_text_data[["text_x", "text_y"]].to_numpy()

    # Euclidean distances & softmax probabilities
    euclidean_distances = np.linalg.norm(img_coords[:, np.newaxis, :] - text_coords[np.newaxis, :, :], axis=2)
    euclidean_probs = softmax(-euclidean_distances, axis=1)  # Lower distance → higher probability

    # Cosine similarity & softmax probabilities
    cosine_sim_mat = cosine_similarity(img_coords, text_coords)
    cosine_probs = softmax(cosine_sim_mat, axis=1)

    # Highlight most similar
    min_euc_idx = np.unravel_index(np.argmin(euclidean_distances), euclidean_distances.shape)
    max_cos_idx = np.unravel_index(np.argmax(cosine_sim_mat), cosine_sim_mat.shape)

    st.markdown("**Euclidean Distance (Lower = More Similar)**")
    euclidean_table = pd.DataFrame(
        euclidean_distances,
        index=[os.path.basename(p) for p in st.session_state.selected_images],
        columns=st.session_state.selected_texts
    )
    st.dataframe(euclidean_table)

    st.markdown("**Cosine Similarity (Higher = More Similar)**")
    cosine_table = pd.DataFrame(
        cosine_sim_mat,
        index=[os.path.basename(p) for p in st.session_state.selected_images],
        columns=st.session_state.selected_texts
    )
    st.dataframe(cosine_table)

    if st.button("Next: Domain–CoDomain Diagram ➡️"):
        st.session_state.step = 3
        st.rerun()

# ========================================
# STEP 3 — DOMAIN / CO-DOMAIN DIAGRAM
# ========================================
elif st.session_state.step == 3:
    st.subheader("Step 3: Domain–CoDomain Interactive Diagram")
    selected_img_data = df[df["image_path"].str.contains('|'.join(st.session_state.selected_images))]
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]

    img_coords = selected_img_data[["img_x", "img_y"]].to_numpy()
    text_coords = selected_text_data[["text_x", "text_y"]].to_numpy()

    # Compute similarities
    euclidean_distances = np.linalg.norm(img_coords[:, np.newaxis, :] - text_coords[np.newaxis, :, :], axis=2)
    euclidean_probs = softmax(-euclidean_distances, axis=1)

    combined_df = pd.DataFrame({
        "x": np.concatenate([img_coords[:, 0], text_coords[:, 0]]),
        "y": np.concatenate([img_coords[:, 1], text_coords[:, 1]]),
        "label": [os.path.basename(p) for p in st.session_state.selected_images] + st.session_state.selected_texts,
        "type": ["Image"] * len(img_coords) + ["Text"] * len(text_coords)
    })

    fig_domain = px.scatter(
        combined_df, x="x", y="y", color="type",
        color_discrete_map={"Image": "blue", "Text": "orange"},
        width=900, height=500
    )
    fig_domain.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=True)
    fig_domain.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_domain.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    # Connect images to best-matched text (Euclidean)
    for i, img in enumerate(st.session_state.selected_images):
        txt_idx = np.argmax(euclidean_probs[i])
        fig_domain.add_trace(go.Scatter(
            x=[img_coords[i, 0], text_coords[txt_idx, 0]],
            y=[img_coords[i, 1], text_coords[txt_idx, 1]],
            mode='lines',
            line=dict(color='green', width=2 + 5 * euclidean_probs[i, txt_idx]),
            hovertemplate=f'Similarity Prob: {euclidean_probs[i, txt_idx]:.2f}<extra></extra>',
            showlegend=False
        ))

    st.plotly_chart(fig_domain)
