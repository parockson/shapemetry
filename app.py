import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import random
import time

# ---- Page setup ----
st.set_page_config(page_title="CLIP Encoder Stepwise Interface", layout="wide")
st.title("🧠 CLIP Encoder Stepwise Interface")

# ---- Load Data ----
@st.cache_data
def load_embeddings():
    csv_path = "static/data/clip_embeddings_2d_improved.csv"
    if not os.path.exists(csv_path):
        st.error(f"❌ File not found: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)
    return df

df = load_embeddings()
image_paths = list(df['image_path'].unique())
texts = list(df['text'].unique())

# Randomize order for display
random.shuffle(image_paths)
random.shuffle(texts)

# ---- Session state ----
if "step" not in st.session_state:
    st.session_state.step = 1
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "selected_texts" not in st.session_state:
    st.session_state.selected_texts = []
if "process_done" not in st.session_state:
    st.session_state.process_done = False

st.sidebar.title("Steps")
st.sidebar.text(f"Current Step: {st.session_state.step}")

# ========================================
# STEP 1 — SELECT IMAGES AND TEXTS
# ========================================
if st.session_state.step == 1:
    st.subheader("Step 1: Select up to 4 Images and 4 Texts")

    col_img, col_txt = st.columns([2, 2])

    # ---- Image Selection ----
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

    # ---- Text Selection ----
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
                        if st.button(f"{text_item}", key=f"text_btn_{i}"):
                            st.session_state.selected_texts.append(text_item)
                    else:
                        st.button("Limit reached", key=f"text_disabled_{i}", disabled=True)

    # ---- Continue ----
    if st.session_state.selected_images and st.session_state.selected_texts:
        st.markdown("---")
        if st.button("Next: Compute Similarity ➡️"):
            st.session_state.step = 2
            st.rerun()

# ========================================
# STEP 2 — VISUAL CHAMBERS + ANIMATION + 2D EMBEDDINGS
# ========================================
elif st.session_state.step == 2:
    st.subheader("Step 2: Visualize and Process Selected Data")

    # ---- Selected Items (Horizontal) ----
    st.markdown("### Selected Items")

    cols = st.columns(2)

    # ---- Selected Images (SI) ----
    with cols[0]:
        st.markdown("#### 🖼️ Selected Images (SI)")
        if st.session_state.selected_images:
            img_cols = st.columns(len(st.session_state.selected_images))
            for i, img in enumerate(st.session_state.selected_images):
                img_path = [p for p in df["image_path"] if os.path.basename(p) == img]
                with img_cols[i]:
                    if img_path and os.path.exists(img_path[0]):
                        st.image(img_path[0], width=90, caption=img)
                    else:
                        st.warning(f"Missing: {img}")
        else:
            st.info("No images selected.")

    # ---- Selected Texts (ST) ----
    with cols[1]:
        st.markdown("#### 📝 Selected Texts (ST)")
        if st.session_state.selected_texts:
            text_cols = st.columns(len(st.session_state.selected_texts))
            for i, txt in enumerate(st.session_state.selected_texts):
                with text_cols[i]:
                    st.markdown(
                        f"<div style='border:1px solid #ccc; border-radius:6px; "
                        f"padding:6px; margin:3px; background:#fefefe; "
                        f"color:#333; font-size:12px; text-align:center; "
                        f"overflow-wrap:break-word; min-height:55px;'>{txt}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No texts selected.")

    st.markdown("---")

    # ---- Chambers Section ----
    st.markdown("### 🔬 Processing Chambers")
    st.markdown(
        "<div style='text-align:center; color:gray;'>"
        "Imagine dragging your Selected Images and Texts into their respective chambers below..."
        "</div>",
        unsafe_allow_html=True,
    )

    chamber_cols = st.columns(2)
    with chamber_cols[0]:
        st.image("static/image/comp_vision.png", caption="🧩 Image Chamber", use_container_width=True)
    with chamber_cols[1]:
        st.image("static/image/nlp.png", caption="✍️ Text Chamber", use_container_width=True)

    # ---- Animation Styling ----
    st.markdown(
        """
        <style>
        @keyframes floatIn {
          0% {transform: translateY(-20px); opacity:0;}
          50% {transform: translateY(5px); opacity:0.6;}
          100% {transform: translateY(0); opacity:1;}
        }
        .float-item {
          animation: floatIn 1s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ---- Process Button ----
    if "processed" not in st.session_state:
        st.session_state.processed = False

    if not st.session_state.processed:
        if st.button("⚙️ Process Embeddings"):
            with st.spinner("Processing CLIP embeddings... please wait"):
                import time
                for i in range(100):
                    time.sleep(0.02)
                    st.progress(i + 1, text=f"Encoding latent representations... {i+1}%")
            st.session_state.processed = True
            st.rerun()

    # ---- Show Formula and Tables only after processing ----
    if st.session_state.processed:
        st.markdown("---")
        st.markdown("## 📘 CLIP Encoder Projection Formula")
        st.latex(r"""
        \text{2D Projection: } 
        \begin{bmatrix}
        x_i \\ 
        y_i
        \end{bmatrix}
        = W_{2D}
        \times 
        \begin{bmatrix}
        e_{1} \\ 
        e_{2} \\ 
        \vdots \\ 
        e_{512}
        \end{bmatrix}
        \;\;\;\text{where}\; W_{2D} \in \mathbb{R}^{2 \times 512}
        """)

        st.success("✅ Processing complete — displaying 2D embeddings!")

        selected_img_data = df[df["image_path"].str.contains("|".join(st.session_state.selected_images))]
        selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]

        # ---- Plots ----
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🖼️ Image 2D Embeddings Plot")
            fig_img = px.scatter(
                selected_img_data, x="img_x", y="img_y",
                text=[os.path.basename(p) for p in selected_img_data["image_path"]],
                title="Image Embeddings (2D Projection)",
                color_discrete_sequence=["blue"]
            )
            fig_img.update_traces(textposition="top center")
            st.plotly_chart(fig_img, use_container_width=True)

            st.markdown("### 🖼️ Image 2D Embeddings Table")
            img_table = selected_img_data[["image_path", "img_x", "img_y"]].rename(
                columns={"image_path": "Image", "img_x": "X", "img_y": "Y"}
            )
            img_table["Image"] = img_table["Image"].apply(lambda p: os.path.basename(p))
            st.dataframe(img_table, hide_index=True)

        with col_right:
            st.markdown("### 📝 Text 2D Embeddings Plot")
            fig_txt = px.scatter(
                selected_text_data, x="text_x", y="text_y",
                text=selected_text_data["text"],
                title="Text Embeddings (2D Projection)",
                color_discrete_sequence=["orange"]
            )
            fig_txt.update_traces(textposition="top center")
            st.plotly_chart(fig_txt, use_container_width=True)

            st.markdown("### 📝 Text 2D Embeddings Table")
            txt_table = selected_text_data[["text", "text_x", "text_y"]].rename(
                columns={"text": "Text", "text_x": "X", "text_y": "Y"}
            )
            st.dataframe(txt_table, hide_index=True)

        st.markdown("---")

    # ---- Navigation Buttons ----
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to Selection"):
            st.session_state.step = 1
            st.session_state.processed = False
            st.rerun()

    with col_next:
        if st.session_state.processed:
            if st.button("Next: Zero-Shot ➡️"):
                st.session_state.step = 3
                st.session_state.processed = False
                st.rerun()


# ========================================
# STEP 3 — ZERO SHOT ANALYSIS
# ========================================
elif st.session_state.step == 3:
    st.subheader("Step 3: 🧠 Zero-Shot Similarity Analysis")

    # ---- Retrieve Selected Data ----
    selected_img_data = df[df["image_path"].str.contains('|'.join(st.session_state.selected_images))]
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]

    # Extract embeddings (simulate if not present)
    img_emb_cols = [c for c in df.columns if c.startswith("img_") and c not in ["img_x", "img_y"]]
    txt_emb_cols = [c for c in df.columns if c.startswith("text_") and c not in ["text_x", "text_y"]]

    if not img_emb_cols or not txt_emb_cols:
        np.random.seed(42)
        img_emb = np.random.rand(len(selected_img_data), 512)
        txt_emb = np.random.rand(len(selected_text_data), 512)
    else:
        img_emb = selected_img_data[img_emb_cols].to_numpy()
        txt_emb = selected_text_data[txt_emb_cols].to_numpy()

    # ---- 1. Cosine Similarity ----
    st.markdown("### 🧮 Cosine Similarities between Selected Images (SI) and Texts (ST)")
    cos_sim = cosine_similarity(img_emb, txt_emb)
    cos_df = pd.DataFrame(
        cos_sim,
        index=[os.path.basename(p) for p in selected_img_data["image_path"]],
        columns=selected_text_data["text"]
    )
    st.dataframe(cos_df.style.format("{:.3f}"), use_container_width=True)

    # ---- 2. Softmax Probabilities ----
    st.markdown("---")
    st.markdown("### 🔄 Softmax Transformation of Similarities")
    st.latex(r"""
    P_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^{N} e^{s_{ik}}}
    \;\;\;\text{where } s_{ij} \text{ is the cosine similarity between image } i \text{ and text } j
    """)

    softmax_probs = softmax(cos_sim, axis=1)
    prob_df = pd.DataFrame(
        softmax_probs,
        index=[os.path.basename(p) for p in selected_img_data["image_path"]],
        columns=selected_text_data["text"]
    )

    st.dataframe(prob_df.style.format("{:.3f}"), use_container_width=True)

    # ---- 3. Representation: Crosstab-like Heatmap ----
    st.markdown("---")
    st.markdown("### 📊 Representation: SI × ST Softmax Probability Heatmap")

    fig_heat = px.imshow(
        prob_df,
        text_auto=".2f",
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Softmax Probabilities (SI × ST)"
    )
    fig_heat.update_layout(
        xaxis_title="Texts (ST)",
        yaxis_title="Images (SI)",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ---- Navigation ----
    col_back, col_end = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to 2D Projection"):
            st.session_state.step = 2
            st.rerun()
    with col_end:
        st.success("✅ End of Zero-Shot Stage")
