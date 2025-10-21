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
import time
from PIL import Image

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
# LESSON 2 — EMBEDDINGS VISUALIZATION (ENHANCED, WITH REFLECTIONS)
# --------------------------------------------------
elif st.session_state.Lesson == 2:
    st.subheader("Lesson 2: Visualize and Process Selected Data")

    st.markdown("""
    In this step, you’ll organize your **Selected Images (SI)** and **Texts (ST)** 
    into their respective processing chambers — one for **Computer Vision (Vision)** 
    and one for **Natural Language Processing (NLP)**.
    """)

    # --- Initialize chambers and session states ---
    if "vision_chamber" not in st.session_state:
        st.session_state.vision_chamber = []
    if "text_chamber" not in st.session_state:
        st.session_state.text_chamber = []
    if "initial_selected_images" not in st.session_state:
        st.session_state.initial_selected_images = [os.path.basename(img) for img in st.session_state.selected_images]
    if "initial_selected_texts" not in st.session_state:
        st.session_state.initial_selected_texts = st.session_state.selected_texts.copy()
    if "image_processed" not in st.session_state:
        st.session_state.image_processed = False
    if "text_processed" not in st.session_state:
        st.session_state.text_processed = False
    if "lesson2_vectors" not in st.session_state:
        st.session_state.lesson2_vectors = {}
    if "reflections_lesson2" not in st.session_state:
        st.session_state.reflections_lesson2 = {}

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
                    if st.button("➕", key=f"add_img_{img}_{i}_lesson2"):
                        st.session_state.selected_images.remove(img)
                        st.session_state.vision_chamber.append(img)
                        st.rerun()
        else:
            st.info("All images are in the chamber.")

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
                    if st.button("➕", key=f"add_txt_{i}_lesson2"):
                        st.session_state.selected_texts.remove(txt)
                        st.session_state.text_chamber.append(txt)
                        st.rerun()
        else:
            st.info("All texts are in the chamber.")

    st.markdown("---")
    st.markdown("### 🔬 Processing Chambers")

    chamber_cols = st.columns(2)

    # ===== Image Chamber =====
    with chamber_cols[0]:
        st.image("static/image/comp_vision.png", caption="🧩 Image Chamber (Vision)", use_container_width=True)
        st.markdown("<div style='text-align:center; font-weight:bold;'>Drop Images Here ↓</div>", unsafe_allow_html=True)

        if st.session_state.vision_chamber:
            img_cols = st.columns(min(len(st.session_state.vision_chamber), 5))
            for i, img in enumerate(st.session_state.vision_chamber):
                with img_cols[i % len(img_cols)]:
                    img_path = next((p for p in image_paths if os.path.basename(p) == img), None)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, width=60)
                    if st.button("➖", key=f"remove_img_{img}_{i}_lesson2"):
                        st.session_state.vision_chamber.remove(img)
                        st.session_state.selected_images.append(img)
                        st.rerun()
        else:
            st.info("No images yet — add from above.")

        # --- Process Image Embeddings (with camera icon) ---
        st.markdown("##### ⚙️ Process Image Embeddings")
        if st.button("🖼️ Embed image vectors", key="process_image_embeddings_lesson2"):
            if not st.session_state.vision_chamber:
                st.warning("⚠️ Add at least one image.")
            else:
                if os.path.exists("static/image/camera.jpg"):
                    st.image("static/image/camera.jpg", width=80)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
                st.session_state.image_processed = True
                st.success("✅ Image embeddings ready!")

    # ===== Text Chamber =====
    with chamber_cols[1]:
        st.image("static/image/nlp.png", caption="✍️ Text Chamber (NLP)", use_container_width=True)
        st.markdown("<div style='text-align:center; font-weight:bold;'>Drop Texts Here ↓</div>", unsafe_allow_html=True)

        if st.session_state.text_chamber:
            txt_cols = st.columns(min(len(st.session_state.text_chamber), 5))
            for i, txt in enumerate(st.session_state.text_chamber):
                with txt_cols[i % len(txt_cols)]:
                    st.markdown(
                        f"<div style='border:1px solid #000; border-radius:6px; padding:4px; margin:3px;"
                        f"background:#f0f0ff; font-size:11px; text-align:center; color:black;'>{txt}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("➖", key=f"remove_txt_{txt}_{i}_lesson2"):
                        st.session_state.text_chamber.remove(txt)
                        st.session_state.selected_texts.append(txt)
                        st.rerun()
        else:
            st.info("No texts yet — add from above.")

        # --- Process Text Embeddings (with desktop icon) ---
        st.markdown("##### ⚙️ Process Text Embeddings")
        if st.button("💻 Embed Text Vectors", key="process_text_embeddings_lesson2"):
            if not st.session_state.text_chamber:
                st.warning("⚠️ Add at least one text.")
            else:
                if os.path.exists("static/image/desktop.svg"):
                    st.image("static/image/desktop.svg", width=80)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
                st.session_state.text_processed = True
                st.success("✅ Text embeddings ready!")

    # --- Visualization ---
    if st.session_state.get("image_processed") or st.session_state.get("text_processed"):
        st.markdown("---")
        st.markdown("## 📘 Embedding Visualizations")

        col_left, col_right = st.columns(2)

        # === Image Embeddings Graph ===
        if st.session_state.get("image_processed"):
            with col_left:
                st.markdown("### 🖼️ Image Embeddings Projection (2D)")
                selected_img_data = df[df["image_path"].apply(
                    lambda x: os.path.basename(x) in st.session_state.vision_chamber
                )]

                if not selected_img_data.empty:
                    fig_img = px.scatter(
                        selected_img_data,
                        x="img_x", y="img_y",
                        text=[os.path.basename(p) for p in selected_img_data["image_path"]],
                        title="Image Embeddings (2D)",
                        color_discrete_sequence=["#007BFF"]
                    )
                    fig_img.update_traces(textfont=dict(color="black", size=12))
                    fig_img.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_img, use_container_width=True)
                else:
                    st.info("No embeddings found for selected images.")

                st.markdown("##### 🧮 Image Embedding Vectors (Editable)")
                img_vectors = pd.DataFrame([
                    {"Image": os.path.basename(p), "X": "", "Y": ""} 
                    for p in st.session_state.vision_chamber
                ])
                updated_img_vectors = st.data_editor(img_vectors, num_rows="dynamic", use_container_width=True, key="img_vectors_lesson2")
                st.session_state.lesson2_vectors["images"] = updated_img_vectors.to_dict(orient="records")

        # === Text Embeddings Graph ===
        if st.session_state.get("text_processed"):
            with col_right:
                st.markdown("### ✍️ Text Embeddings Projection (2D)")
                selected_txt_data = df[df["text"].isin(st.session_state.text_chamber)]

                if not selected_txt_data.empty:
                    fig_txt = px.scatter(
                        selected_txt_data,
                        x="text_x", y="text_y",
                        text=selected_txt_data["text"],
                        title="Text Embeddings (2D)",
                        color_discrete_sequence=["#FF8800"]
                    )
                    fig_txt.update_traces(textfont=dict(color="black", size=12))
                    fig_txt.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_txt, use_container_width=True)
                else:
                    st.info("No embeddings found for selected texts.")

                st.markdown("##### 🧮 Text Embedding Vectors (Editable)")
                txt_vectors = pd.DataFrame([
                    {"Text": txt, "X": "", "Y": ""} 
                    for txt in st.session_state.text_chamber
                ])
                updated_txt_vectors = st.data_editor(txt_vectors, num_rows="dynamic", use_container_width=True, key="txt_vectors_lesson2")
                st.session_state.lesson2_vectors["texts"] = updated_txt_vectors.to_dict(orient="records")

    # --- Reflection Questions (always visible after processing) ---
    if st.session_state.get("image_processed") or st.session_state.get("text_processed"):
        st.markdown("---")
        st.subheader("💡 Reflection Questions (Required for Lesson 2)")

        fields_selected_lesson2 = st.multiselect(
            "1️⃣ Which of these fields of study did you apply their concepts in this lesson?",
            ["Data Science", "Artificial Intelligence", "Mathematics Education"],
            key="fields_selected_lesson2"
        )
        reflection_text_lesson2 = st.text_area(
            "2️⃣ Describe which concepts you used from each field.",
            placeholder="E.g., I used cosine similarity from AI and vector projection from mathematics...",
            key="reflection_text_lesson2"
        )
        action_items_lesson2 = st.text_input(
            "3️⃣ Next steps / action items (optional)",
            placeholder="E.g., refine image preprocessing, tune embedding dims...",
            key="action_items_lesson2"
        )

        ready = bool(fields_selected_lesson2) and reflection_text_lesson2.strip()

        if ready:
            st.session_state.reflections_lesson2 = {
                "fields": fields_selected_lesson2,
                "reflection": reflection_text_lesson2,
                "actions": action_items_lesson2
            }

        col_ref_back, col_ref_next = st.columns(2)
        with col_ref_back:
            if st.button("⬅️ Back", key="back_ref_lesson2"):
                st.session_state.Lesson = 1
                st.rerun()
        with col_ref_next:
            if ready and st.button("Next ➡️", key="next_ref_lesson2"):
                st.session_state.Lesson = 3
                st.rerun()
            elif not ready:
                st.warning("⚠️ Please complete the reflection questions before continuing.")

    


# --------------------------------------------------
# LESSON 3 — DATA NORMALIZATION (UNIT CIRCLE)
# --------------------------------------------------
elif st.session_state.Lesson == 3:
    st.subheader("Lesson 3: Data Normalization (Unit Circle)")

    st.markdown("""
    In this lesson, we normalize the **embedding vectors** generated in Lesson 2.
    Normalization ensures that all vectors lie on a **unit circle**, making
    comparison and distance calculations fairer across dimensions.
    """)

    # --- Retrieve vectors from Lesson 2 ---
    vectors_data = st.session_state.get("lesson2_vectors", {})
    img_vectors = vectors_data.get("images", [])
    txt_vectors = vectors_data.get("texts", [])

    if not img_vectors and not txt_vectors:
        st.warning("⚠️ No vectors found. Please complete Lesson 2 first.")
        if st.button("⬅️ Go Back to Lesson 2"):
            st.session_state.Lesson = 2
            st.rerun()
    else:
        st.markdown("### 📊 Before Normalization")
        col1, col2 = st.columns(2)

        # --- Convert to DataFrames ---
        if img_vectors:
            df_img = pd.DataFrame(img_vectors)
            df_img = df_img[df_img["X"].astype(str).str.strip() != ""]
            df_img[["X", "Y"]] = df_img[["X", "Y"]].astype(float)

            with col1:
                st.markdown("#### 🖼️ Image Vectors")
                fig_img_before = px.scatter(
                    df_img, x="X", y="Y", text="Image", color_discrete_sequence=["#007BFF"],
                    title="Image Embeddings (Before Normalization)"
                )
                fig_img_before.update_traces(textfont=dict(color="black", size=11))
                fig_img_before.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_img_before, use_container_width=True)

        if txt_vectors:
            df_txt = pd.DataFrame(txt_vectors)
            df_txt = df_txt[df_txt["X"].astype(str).str.strip() != ""]
            df_txt[["X", "Y"]] = df_txt[["X", "Y"]].astype(float)

            with col2:
                st.markdown("#### ✍️ Text Vectors")
                fig_txt_before = px.scatter(
                    df_txt, x="X", y="Y", text="Text", color_discrete_sequence=["#FF8800"],
                    title="Text Embeddings (Before Normalization)"
                )
                fig_txt_before.update_traces(textfont=dict(color="black", size=11))
                fig_txt_before.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_txt_before, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚙️ Normalization Process")

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # --- Normalize Function ---
        def normalize_df(df):
            norm = np.sqrt(df["X"] ** 2 + df["Y"] ** 2)
            df["X_norm"] = df["X"] / norm
            df["Y_norm"] = df["Y"] / norm
            return df

        df_img_norm, df_txt_norm = None, None
        if img_vectors:
            df_img_norm = normalize_df(df_img.copy())
        if txt_vectors:
            df_txt_norm = normalize_df(df_txt.copy())

        st.success("✅ Normalization Complete! Each vector now lies on the unit circle.")

        st.markdown("### 🌀 After Normalization")
        col3, col4 = st.columns(2)

        if df_img_norm is not None:
            with col3:
                st.markdown("#### 🖼️ Normalized Image Vectors")
                fig_img_after = px.scatter(
                    df_img_norm, x="X_norm", y="Y_norm", text="Image",
                    color_discrete_sequence=["#007BFF"],
                    title="Image Embeddings (After Normalization)"
                )
                fig_img_after.update_traces(textfont=dict(color="black", size=11))
                fig_img_after.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_img_after, use_container_width=True)

        if df_txt_norm is not None:
            with col4:
                st.markdown("#### ✍️ Normalized Text Vectors")
                fig_txt_after = px.scatter(
                    df_txt_norm, x="X_norm", y="Y_norm", text="Text",
                    color_discrete_sequence=["#FF8800"],
                    title="Text Embeddings (After Normalization)"
                )
                fig_txt_after.update_traces(textfont=dict(color="black", size=11))
                fig_txt_after.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_txt_after, use_container_width=True)

        # Save normalized vectors for next lesson
        st.session_state["lesson3_vectors"] = {
            "images": df_img_norm.to_dict(orient="records") if df_img_norm is not None else [],
            "texts": df_txt_norm.to_dict(orient="records") if df_txt_norm is not None else []
        }

        # --- Reflection Section ---
        st.markdown("---")
        st.subheader("💭 Reflection Questions (Lesson 3)")

        q1 = st.multiselect(
            "1️⃣ Which mathematical operations were used in normalization?",
            ["Vector Length", "Dot Product", "Unit Circle Concept", "Division by Magnitude"],
            key="lesson3_q1"
        )
        q2 = st.text_area(
            "2️⃣ Explain why normalization helps in comparing embeddings.",
            placeholder="E.g., normalization makes all vectors comparable regardless of scale...",
            key="lesson3_q2"
        )
        q3 = st.text_input(
            "3️⃣ What next step will you take to analyze these embeddings?",
            placeholder="E.g., compute cosine similarity or clustering...",
            key="lesson3_q3"
        )

        ready3 = bool(q1) and q2.strip()

        if ready3:
            st.session_state["reflections_lesson3"] = {
                "concepts": q1,
                "reflection": q2,
                "next_steps": q3
            }

        # --- Navigation Buttons ---
        st.markdown("---")
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("🔄 Refresh", key="refresh_lesson3"):
                st.rerun()
        with nav_col2:
            if st.button("⬅️ Back", key="back_lesson3"):
                st.session_state.Lesson = 2
                st.rerun()
        with nav_col3:
            if ready3 and st.button("Next ➡️", key="next_lesson3"):
                st.session_state.Lesson = 4
                st.rerun()
            elif not ready3:
                st.warning("⚠️ Please complete the reflection questions before continuing.")

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
