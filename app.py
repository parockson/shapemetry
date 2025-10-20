import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
import random
import time
import json
import streamlit_sortables as sortables  # ✅ moved to the top

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="MONICCA ENVIRONMENT", layout="wide")
st.title("🧠 MonicCA Interactive Learning Env't")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
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
image_paths = list(df["image_path"].unique())
texts = list(df["text"].unique())
random.shuffle(image_paths)
random.shuffle(texts)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "Lesson" not in st.session_state:
    st.session_state.Lesson = 0  # 0 = onboarding
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "selected_texts" not in st.session_state:
    st.session_state.selected_texts = []
if "process_done" not in st.session_state:
    st.session_state.process_done = False
if "processed" not in st.session_state:
    st.session_state.processed = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# --------------------------------------------------
# LOAD COUNTRY / REGION JSON
# --------------------------------------------------
@st.cache_data
def load_country_data():
    path = "static/data/countries_states.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.warning("⚠️ countries_states.json not found — using fallback data.")
        return [
            {"name": "Ghana", "states": [{"name": "Ashanti"}, {"name": "Greater Accra"}, {"name": "Northern"}]},
            {"name": "Nigeria", "states": [{"name": "Lagos"}, {"name": "Abuja"}, {"name": "Ogun"}]},
            {"name": "United States", "states": [{"name": "California"}, {"name": "New York"}, {"name": "Texas"}]},
        ]


countries_data = load_country_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Lessons")
lesson_names = ["Welcome", "Data Collection", "Embeddings", "Zero-Shot"]
st.sidebar.text(f"Current Lesson: {lesson_names[st.session_state.Lesson]}")

# --------------------------------------------------
# LESSON 0 — ONBOARDING / WELCOME
# --------------------------------------------------
if st.session_state.Lesson == 0:
    st.title("🎓 MonicCa Embedding Learning Environment")

    st.markdown("""
    The **MonicCa Embedding Learning Environment** introduces how AI models represent images and text 
    as embeddings in a shared vector space. Through hands-on exploration, 
    you’ll learn how similarity metrics like cosine distance enable AI to relate concepts.
                
    --- 
    ### 🧩 What You’ll Learn 
     - How AI models represent **images** and **text** as vectors in a shared embedding space 
     - How **normalization**, **cosine similarity**, and **Euclidean distance** work 
     - How AI performs **zero-shot image–text classification** without explicit task-specific training 
                
    --- 
    ### 📘 Learning Journey 
    You’ll begin by: 
    1. Selecting images and text descriptions 
    2. Generating their embedding representations 
    3. Observing how the system measures similarity and identifies matches 
                
    Each stage combines **mathematics**, **data science**, and **AI concepts**, allowing you to reflect on how multimodal systems interpret the world. 
    """)

    st.markdown("---")
    st.subheader("📝 Participant Information")

    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "Select a Country"
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = "Select a Region/State"

    with st.form("onboarding_form"):
        st.markdown("### 👤 Demographic Information")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name*", placeholder="Enter your full name")
            age = st.number_input("🎂 Age*", min_value=5, max_value=100, value=None, placeholder="Enter your age")
            gender = st.selectbox("🚻 Gender*", ["Select", "Male", "Female", "Non-binary", "Prefer not to say", "Other"])
            grade_level = st.selectbox(
                "🎓 Grade Level*", ["Select", "Junior High", "Senior High", "Undergraduate", "Postgraduate", "Professional", "Other"]
            )

        with col2:
            country_names = ["Select a Country"] + [c["name"] for c in countries_data]
            selected_country = st.selectbox(
                "🌍 Country", country_names,
                index=country_names.index(st.session_state.selected_country)
                if st.session_state.selected_country in country_names else 0,
                key="country_select"
            )
            st.session_state.selected_country = selected_country

            region_options = ["Select a Region/State"]
            if selected_country != "Select a Country":
                country_info = next((c for c in countries_data if c["name"] == selected_country), None)
                if country_info and "states" in country_info:
                    region_options += [s["name"] for s in country_info["states"]]

            selected_region = st.selectbox(
                "🏙️ Region / State", region_options,
                index=region_options.index(st.session_state.selected_region)
                if st.session_state.selected_region in region_options else 0,
                key="region_select"
            )
            st.session_state.selected_region = selected_region

        st.markdown("---")
        consent = st.checkbox(
            "✅ I consent to the collection and use of my data (text, images, or both) "
            "for research and model training within this project."
        )

        submitted = st.form_submit_button("Start Learning 🚀")
        if submitted:
            if not name.strip():
                st.warning("⚠️ Please enter your full name.")
            elif age is None or age < 5:
                st.warning("⚠️ Please enter a valid age (5–100).")
            elif gender == "Select":
                st.warning("⚠️ Please select your gender.")
            elif selected_country == "Select a Country":
                st.warning("⚠️ Please select your country.")
            elif selected_region == "Select a Region/State":
                st.warning("⚠️ Please select your region/state.")
            elif grade_level == "Select":
                st.warning("⚠️ Please select your grade level.")
            elif not consent:
                st.warning("⚠️ Consent is required to continue.")
            else:
                st.session_state.user_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "country": selected_country,
                    "region": selected_region,
                    "grade_level": grade_level,
                    "consent": consent,
                }
                st.success(f"Welcome, {name} ({age}, {gender}) from {selected_region}, {selected_country}! 🎉")
                st.session_state.Lesson = 1
                st.rerun()

# --------------------------------------------------
# LESSON 1 — SELECTION
# --------------------------------------------------
elif st.session_state.Lesson == 1:
    st.subheader("Lesson 1: Select up to 4 Images and 4 Texts")
    col_img, col_txt = st.columns([2, 2])

    with col_img:
        st.markdown("### 🖼️ Select Images")
        cols = st.columns(4)
        for idx, image_path in enumerate(image_paths):
            col = cols[idx % 4]
            with col:
                if os.path.exists(image_path):
                    st.image(image_path, width=100)
                label = os.path.basename(image_path)
                if label in st.session_state.selected_images:
                    if st.button("✅ Deselect", key=f"img_{idx}"):
                        st.session_state.selected_images.remove(label)
                elif len(st.session_state.selected_images) < 4:
                    if st.button("Select", key=f"img_sel_{idx}"):
                        st.session_state.selected_images.append(label)

    with col_txt:
        st.markdown("### 📝 Select Texts")
        txt_cols = st.columns(2)
        for i, text_item in enumerate(texts):
            col = txt_cols[i % 2]
            with col:
                if text_item in st.session_state.selected_texts:
                    if st.button("✅ Deselect", key=f"text_{i}"):
                        st.session_state.selected_texts.remove(text_item)
                elif len(st.session_state.selected_texts) < 4:
                    if st.button(f"{text_item}", key=f"text_btn_{i}"):
                        st.session_state.selected_texts.append(text_item)

    # Reflection
    if st.session_state.selected_images and st.session_state.selected_texts:
        st.markdown("---")
        st.subheader("💡 Reflection Questions (Required)")
        fields_selected = st.multiselect(
            "1️⃣ Which of these fields of study did you apply their concepts in this lesson?",
            ["Data Science", "Artificial Intelligence", "Mathematics Education"]
        )
        reflection_text = st.text_area(
            "2️⃣ Describe which concepts you used from each field.",
            placeholder="E.g., I used cosine similarity from AI and vector projection from mathematics..."
        )
        ready = bool(fields_selected) and reflection_text.strip()
        st.markdown("---")
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("⬅️ Back"):
                st.session_state.Lesson = 0
                st.rerun()
        with col_next:
            if ready and st.button("Next ➡️"):
                st.session_state.Lesson = 2
                st.rerun()
            elif not ready:
                st.warning("⚠️ Please complete the reflection questions before continuing.")

# --------------------------------------------------
# LESSON 2 — EMBEDDINGS VISUALIZATION (FULL ROBUST VERSION)
# --------------------------------------------------
elif st.session_state.Lesson == 2:
    st.subheader("Lesson 2: Visualize and Process Selected Data")

    st.markdown("""
    In this step, you’ll **move your Selected Images (SI)** and **Texts (ST)** 
    into their respective processing chambers — one for **Computer Vision (Vision)** 
    and one for **Natural Language Processing (NLP)**.
    Only items placed in the chambers will be processed. ⚙️
    """)

    # --- Initialize chambers ---
    if "vision_chamber" not in st.session_state:
        st.session_state.vision_chamber = []
    if "text_chamber" not in st.session_state:
        st.session_state.text_chamber = []
    if "initial_selected_images" not in st.session_state:
        # store basenames for consistency
        st.session_state.initial_selected_images = [os.path.basename(img) for img in st.session_state.selected_images]
    if "initial_selected_texts" not in st.session_state:
        st.session_state.initial_selected_texts = st.session_state.selected_texts.copy()

    st.markdown("---")
    st.markdown("### 🔬 Processing Chambers")

    chamber_cols = st.columns(2)

    # ===== Image Chamber =====
    with chamber_cols[0]:
        st.image("static/image/comp_vision.png", caption="🧩 Image Chamber (Vision)", use_container_width=True)
        st.markdown(
            "<div style='color:white; font-weight:bold; text-align:center; background:#000;"
            "padding:6px; border-radius:5px;'>Drop Images (SI) Here ↓</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.vision_chamber:
            img_cols = st.columns(min(len(st.session_state.vision_chamber), 5))
            for i, img in enumerate(st.session_state.vision_chamber):
                with img_cols[i % len(img_cols)]:
                    img_path = next((p for p in image_paths if os.path.basename(p) == img), None)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=60)
                    if st.button("➖", key=f"remove_img_{img}"):
                        st.session_state.vision_chamber.remove(img)
                        st.session_state.selected_images.append(img)
                        st.rerun()
        else:
            st.info("No images yet — add from below.")

    # ===== Text Chamber =====
    with chamber_cols[1]:
        st.image("static/image/nlp.png", caption="✍️ Text Chamber (NLP)", use_container_width=True)
        st.markdown(
            "<div style='color:white; font-weight:bold; text-align:center; background:#000;"
            "padding:6px; border-radius:5px;'>Drop Texts (ST) Here ↓</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.text_chamber:
            txt_cols = st.columns(min(len(st.session_state.text_chamber), 5))
            for i, txt in enumerate(st.session_state.text_chamber):
                with txt_cols[i % len(txt_cols)]:
                    st.markdown(
                        f"<div style='border:1px solid #000; border-radius:6px; padding:4px; margin:3px;"
                        f"background:#f0f0ff; font-size:11px; text-align:center; color:black;'>{txt}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("➖", key=f"remove_txt_{txt}"):
                        st.session_state.text_chamber.remove(txt)
                        st.session_state.selected_texts.append(txt)
                        st.rerun()
        else:
            st.info("No texts yet — add from below.")

    st.markdown("---")
    st.markdown("### 🎯 Remaining Items")

    remaining_cols = st.columns(2)

    # --- Remaining Images (SI) ---
    with remaining_cols[0]:
        st.markdown("#### 🖼️ Remaining Images (SI)")
        if st.session_state.selected_images:
            img_cols = st.columns(min(len(st.session_state.selected_images), 6))
            for i, img in enumerate(st.session_state.selected_images):
                with img_cols[i % len(img_cols)]:
                    img_path = next((p for p in image_paths if os.path.basename(p) == img), None)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=60)
                    if st.button("➕", key=f"add_img_{img}"):
                        st.session_state.selected_images.remove(img)
                        st.session_state.vision_chamber.append(img)
                        st.rerun()

    # --- Remaining Texts (ST) ---
    with remaining_cols[1]:
        st.markdown("#### 📝 Remaining Texts (ST)")
        if st.session_state.selected_texts:
            txt_cols = st.columns(min(len(st.session_state.selected_texts), 5))
            for i, txt in enumerate(st.session_state.selected_texts):
                with txt_cols[i % len(txt_cols)]:
                    st.markdown(
                        f"<div style='border:1px solid #aaa; border-radius:6px; padding:4px; margin:3px;"
                        f"background:#f9f9ff; font-size:11px; text-align:center; color:black;'>{txt}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("➕", key=f"add_txt_{txt}"):
                        st.session_state.selected_texts.remove(txt)
                        st.session_state.text_chamber.append(txt)
                        st.rerun()

    st.markdown("---")

    # --- PROCESS BUTTON ---
    if not st.session_state.processed:
        if st.button("⚙️ Process Embeddings"):
            if not st.session_state.vision_chamber or not st.session_state.text_chamber:
                st.warning("⚠️ Please move at least one image and one text into the chambers before processing.")
            else:
                with st.spinner("Processing embeddings... please wait"):
                    for i in range(100):
                        time.sleep(0.02)
                        st.progress(i + 1)
                st.session_state.processed = True
                st.rerun()

    # --- REFLECTION QUESTIONS ---
    if st.session_state.processed:
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
        """)

        selected_img_data = df[df["image_path"].apply(lambda x: os.path.basename(x) in st.session_state.vision_chamber)]
        selected_text_data = df[df["text"].isin(st.session_state.text_chamber)]

        col_left, col_right = st.columns(2)

        with col_left:
            fig_img = px.scatter(
                selected_img_data, x="img_x", y="img_y",
                text=[os.path.basename(p) for p in selected_img_data["image_path"]],
                title="Image Embeddings (2D Projection)", color_discrete_sequence=["blue"]
            )
            st.plotly_chart(fig_img, use_container_width=True)
            st.markdown("#### 🖼️ Image Embedding Table")
            for _, row in selected_img_data.iterrows():
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    if os.path.exists(row["image_path"]):
                        st.image(row["image_path"], width=45)
                with c2:
                    st.markdown(f"**X:** {row['img_x']:.3f}")
                with c3:
                    st.markdown(f"**Y:** {row['img_y']:.3f}")

        with col_right:
            fig_txt = px.scatter(
                selected_text_data, x="text_x", y="text_y",
                text=selected_text_data["text"],
                title="Text Embeddings (2D Projection)", color_discrete_sequence=["orange"]
            )
            st.plotly_chart(fig_txt, use_container_width=True)
            st.markdown("#### 📝 Text Embedding Table")
            st.dataframe(
                selected_text_data[["text", "text_x", "text_y"]]
                .rename(columns={"text": "Text", "text_x": "X-Coordinate", "text_y": "Y-Coordinate"}),
                hide_index=True, use_container_width=True,
            )

        # --- Navigation ---
        st.markdown("---")
        col_back, col_refresh, col_next = st.columns(3)
        with col_back:
            if st.button("⬅️ Back"):
                st.session_state.Lesson = 1
                st.session_state.processed = False
                st.rerun()

        with col_refresh:
            if st.button("🔄 Refresh Session"):
                st.session_state.selected_images = st.session_state.initial_selected_images.copy()
                st.session_state.selected_texts = st.session_state.initial_selected_texts.copy()
                st.session_state.vision_chamber = []
                st.session_state.text_chamber = []
                st.session_state.processed = False
                st.success("🔁 Lesson 2 reset. Chambers cleared, items restored.")
                st.rerun()

        with col_next:
            if st.button("Next ➡️"):
                st.session_state.Lesson = 3
                st.session_state.processed = False
                st.rerun()







# --------------------------------------------------
# LESSON 3 — ZERO-SHOT SIMILARITY (ROBUST VERSION)
# --------------------------------------------------
elif st.session_state.Lesson == 3:
    st.subheader("Lesson 3: 🧠 Zero-Shot Similarity Analysis")

    # --- Select data based on basenames to avoid mismatch ---
    selected_img_data = df[df["image_path"].apply(lambda x: os.path.basename(x) in st.session_state.selected_images)]
    selected_text_data = df[df["text"].isin(st.session_state.selected_texts)]

    if selected_img_data.empty or selected_text_data.empty:
        st.warning("⚠️ No images or texts selected for similarity analysis. Please go back and select items.")
    else:
        # --- Identify embedding columns ---
        img_emb_cols = [c for c in df.columns if c.startswith("img_") and c not in ["img_x", "img_y"]]
        txt_emb_cols = [c for c in df.columns if c.startswith("text_") and c not in ["text_x", "text_y"]]

        if not img_emb_cols or not txt_emb_cols:
            st.error("❌ No embedding columns found. Check your CSV column names!")
        else:
            # --- Extract embeddings ---
            img_emb = selected_img_data[img_emb_cols].to_numpy()
            txt_emb = selected_text_data[txt_emb_cols].to_numpy()

            # Safety check: ensure embeddings are non-empty
            if img_emb.size == 0 or txt_emb.size == 0:
                st.warning("⚠️ Embeddings are empty — cannot compute similarity.")
            else:
                # --- Cosine Similarity ---
                st.markdown("### 🧮 Cosine Similarities between Selected Images (SI) and Texts (ST)")
                cos_sim = cosine_similarity(img_emb, txt_emb)
                cos_df = pd.DataFrame(
                    cos_sim,
                    index=[os.path.basename(p) for p in selected_img_data["image_path"]],
                    columns=selected_text_data["text"]
                )
                st.dataframe(cos_df.style.format("{:.3f}"), use_container_width=True)

                # --- Softmax Probabilities ---
                st.markdown("---")
                st.markdown("### 🔄 Softmax Probabilities")
                st.latex(r"P_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^{N} e^{s_{ik}}}")
                softmax_probs = softmax(cos_sim, axis=1)
                prob_df = pd.DataFrame(
                    softmax_probs,
                    index=[os.path.basename(p) for p in selected_img_data["image_path"]],
                    columns=selected_text_data["text"]
                )
                st.dataframe(prob_df.style.format("{:.3f}"), use_container_width=True)

                # --- Heatmap ---
                st.markdown("---")
                st.markdown("### 📊 SI × ST Probability Heatmap")
                fig_heat = px.imshow(
                    prob_df, text_auto=".2f", color_continuous_scale="Viridis",
                    aspect="auto", title="Softmax Probabilities (SI × ST)"
                )
                fig_heat.update_layout(xaxis_title="Texts (ST)", yaxis_title="Images (SI)")
                st.plotly_chart(fig_heat, use_container_width=True)

                # --- Navigation ---
                col_back, col_end = st.columns([1, 1])
                with col_back:
                    if st.button("⬅️ Back to 2D Projection"):
                        st.session_state.Lesson = 2
                        st.rerun()
                with col_end:
                    st.success("✅ End of Zero-Shot Stage")
                    st.markdown("### 🎉 Congratulations on completing the MonicCa Embedding Learning Environment!")
