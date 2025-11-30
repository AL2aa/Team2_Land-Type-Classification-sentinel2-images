import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import base64
from tensorflow.keras.applications import efficientnet
from pathlib import Path
from visualization import plot_prediction_probs

# -------------------------------
# إعداد الصفحة
# -------------------------------
st.set_page_config(page_title="Land Type Classifier", page_icon="🌍", layout="wide")


# -------------------------------
# دالة لتعيين الخلفية
# -------------------------------
def set_background(image_file: Path):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        overflow-y: scroll;
    }}
    
    ::-webkit-scrollbar {{
        width: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(245, 245, 220, 0.3);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: #556B2F;
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #6B8E23;
    }}
    
    h1, h2, h3, h4, h5, h6, p, label {{
        color: #556B2F !important;
        background-color: rgba(245, 245, 220, 0.8);
        padding: 6px 10px;
        border-radius: 8px;
    }}
    
    .prediction-result h1,
    .prediction-result h2,
    .prediction-result h3,
    .prediction-result p {{
        background-color: transparent !important;
        padding: 0 !important;
    }}
    
    .uploaded-info {{
        background-color: rgba(245, 245, 220, 0.85);
        padding: 12px;
        border-radius: 10px;
        font-size: 18px;
        color: #556B2F;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
    }}
    
    .prediction-result {{
        background-color: rgba(245, 245, 220, 0.9);
        padding: 20px 40px;
        border-radius: 15px;
        border: 3px solid #556B2F;
        text-align: center;
        margin: 20px auto;
        display: inline-block;
        max-width: fit-content;
    }}
    
    .prediction-class {{
        color: #2E7D32 !important;
        font-size: 48px !important;
        font-weight: bold !important;
        margin: 0 !important;
        background-color: transparent !important;
    }}
    
    .center-content {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    
    [data-testid="stFileUploader"] {{
        background-color: rgba(245, 245, 220, 0.9);
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #556B2F;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent
set_background(BASE_DIR / "assets" / "background.jpg")

# -------------------------------
# العنوان الرئيسي
# -------------------------------
st.markdown(
    "<h1 style='text-align: center;'>Land Type Classification</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload a satellite image for prediction.</p>",
    unsafe_allow_html=True,
)

# -------------------------------
# تحميل المودل (Inference only)
# -------------------------------
MODEL_PATH = BASE_DIR.parent / "models" / "eurosat_effb0_best.keras"


@st.cache_resource
def load_model_cached():
    return keras.saving.load_model(MODEL_PATH)


model = load_model_cached()

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# -------------------------------
# رفع الصورة والتنبؤ
# -------------------------------
col_upload, col_reset = st.columns([4, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        label="Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader"
    )

with col_reset:
    if uploaded_file is not None:
        st.write("")
        if st.button("Upload New Image", key="reset_upload"):
            st.session_state.pop("file_uploader", None)
            st.rerun()

if uploaded_file:
    # معلومات عن الملف
    file_details = f"{uploaded_file.name} — {round(uploaded_file.size / 1024, 1)} KB"
    st.markdown(
        f"<div class='uploaded-info'>{file_details}</div>", unsafe_allow_html=True
    )

    # عرض الصورة
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة للمودل
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        pred_probs = prediction[0]

    # عرض النتيجة
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='prediction-class'>{predicted_class}</p>", unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # عرض الـ bar chart
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        fig_bar = plot_prediction_probs(pred_probs, class_names)
        st.pyplot(fig_bar)
