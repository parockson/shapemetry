import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

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
            st.image(image_path, width=100)
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

if st.session_state.selected_images:
    selected_img_data = df[df["image_path"].str.contains('|'.join(st.session_state.selected_images))]
    st.dataframe(selected_img_data[["image_path", "img_x", "img_y"]], width='stretch')
else:
    st.info("No images selected yet.")

if st.session_state.selected_texts:
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]
    st.dataframe(selected_text_data[["text", "text_x", "text_y"]], width='stretch')
else:
    st.info("No texts selected yet.")

# ---- Scatterplots Side-by-Side ----
if st.session_state.selected_images or st.session_state.selected_texts:
    st.markdown("---")
    st.subheader("🔎 2D Encoder Spaces")

    if st.session_state.selected_images:
        fig_img = px.scatter(
            selected_img_data,
            x="img_x", y="img_y",
            text=selected_img_data["image_path"].apply(lambda p: os.path.basename(p)),
            title="🖼️ Image Encoder Space (2D)"
        )
        st.plotly_chart(fig_img, use_container_width=True)
    else:
        st.info("No images selected for plotting.")

    if st.session_state.selected_texts:
        fig_txt = px.scatter(
            selected_text_data,
            x="text_x", y="text_y",
            text="text",
            color_discrete_sequence=["orange"],
            title="💬 Text Encoder Space (2D)"
        )
        st.plotly_chart(fig_txt, use_container_width=True)
    else:
        st.info("No texts selected for plotting.")

# ---- Similarity Formulas ----
st.markdown("---")
st.subheader("📐 Similarity Formulas")
st.markdown(r"""
**Euclidean Distance**

$$
d(v_i, v_t) = \sqrt{ \sum_{k=1}^{n} (v_{i,k} - v_{t,k})^2 }
$$

**Cosine Similarity**

$$
\text{cosine\_sim}(v_i, v_t) = \frac{v_i \cdot v_t}{\|v_i\| \|v_t\|}
$$
""")

# ---- Common Embedding Space with Zero-Mean & Unit Vectors ----
if st.session_state.selected_images and st.session_state.selected_texts:
    st.markdown("---")
    st.subheader("🌐 Common Embedding Space (Zero-Mean & Unit Vectors)")

    img_coords = selected_img_data[["img_x","img_y"]].to_numpy()
    text_coords = selected_text_data[["text_x","text_y"]].to_numpy()

    # Zero-mean
    img_coords_centered = img_coords - np.mean(img_coords, axis=0)
    text_coords_centered = text_coords - np.mean(text_coords, axis=0)

    # Unit-normalize
    img_coords_unit = img_coords_centered / np.linalg.norm(img_coords_centered, axis=1, keepdims=True)
    text_coords_unit = text_coords_centered / np.linalg.norm(text_coords_centered, axis=1, keepdims=True)

    # Euclidean distance
    euclidean_distances = np.linalg.norm(img_coords_unit[:, np.newaxis, :] - text_coords_unit[np.newaxis, :, :], axis=2)
    euclidean_table = pd.DataFrame(
        euclidean_distances,
        index=[os.path.basename(p) for p in st.session_state.selected_images],
        columns=st.session_state.selected_texts
    )

    # Cosine similarity
    cosine_sim = np.dot(img_coords_unit, text_coords_unit.T)
    cosine_table = pd.DataFrame(
        cosine_sim,
        index=[os.path.basename(p) for p in st.session_state.selected_images],
        columns=st.session_state.selected_texts
    )

    # Highlight most similar pairs
    min_euc_idx = np.unravel_index(np.argmin(euclidean_distances), euclidean_distances.shape)
    max_cos_idx = np.unravel_index(np.argmax(cosine_sim), cosine_sim.shape)

    def highlight_min_max(df, highlight_idx):
        styled = pd.DataFrame('', index=df.index, columns=df.columns)
        r, c = highlight_idx
        styled.iloc[r, c] = 'background-color: lightgreen'
        return styled

    st.markdown("**Euclidean Distance (Lower = More Similar)**")
    st.dataframe(euclidean_table.style.apply(lambda df: highlight_min_max(df, min_euc_idx), axis=None))

    st.markdown("**Cosine Similarity (Higher = More Similar)**")
    st.dataframe(cosine_table.style.apply(lambda df: highlight_min_max(df, max_cos_idx), axis=None))

    # ---- Best Match / Zero-Shot ----
    st.markdown("---")
    st.subheader("🏆 Best Match & Zero-Shot Prediction")

    best_matches = []
    for i, img in enumerate(st.session_state.selected_images):
        eu_idx = np.argmin(euclidean_distances[i])
        cos_idx = np.argmax(cosine_sim[i])
        best_matches.append({
            "Image": img,
            "Best Match (Euclidean)": st.session_state.selected_texts[eu_idx],
            "Best Match (Cosine)": st.session_state.selected_texts[cos_idx]
        })

    st.dataframe(pd.DataFrame(best_matches), width='stretch')

    st.markdown("""
**Explanation:**  
- Each image is compared against all candidate text descriptions.  
- The text with the **highest similarity** is chosen as the **best match**.  
- This predicted label represents a **zero-shot classification**—the model generalizes based on learned image-text relationships.
""")

    # ---- Combined Interactive Scatter ----
    combined_df = pd.DataFrame({
        "x": np.concatenate([img_coords[:,0], text_coords[:,0]]),
        "y": np.concatenate([img_coords[:,1], text_coords[:,1]]),
        "label": [os.path.basename(p) for p in st.session_state.selected_images] + st.session_state.selected_texts,
        "type": ["Image"]*len(img_coords) + ["Text"]*len(text_coords)
    })

    fig_combined = px.scatter(
        combined_df, x="x", y="y", color="type", text="label",
        title="Combined 2D Embedding Space (Interactive)",
        color_discrete_map={"Image":"blue", "Text":"orange"}
    )

    # Line for most similar pair (Euclidean)
    img_idx, txt_idx = min_euc_idx
    fig_combined.add_trace(
        go.Scatter(
            x=[img_coords[img_idx,0], text_coords[txt_idx,0]],
            y=[img_coords[img_idx,1], text_coords[txt_idx,1]],
            mode='lines',
            line=dict(color='green', width=2),
            name='Most Similar Pair (Euclidean)'
        )
    )

    st.plotly_chart(fig_combined, use_container_width=True)
