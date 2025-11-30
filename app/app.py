import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.preprocessing import image_dataset_from_directory
from visualization import plot_prediction_probs, plot_random_image
import matplotlib.pyplot as plt
import random

# -------------------------------
# إعداد الصفحة
# -------------------------------
st.set_page_config(
    page_title="Land Type Classifier",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# دالة لتعيين الخلفية
# -------------------------------
def set_background(image_file):
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
    
    .section-box {{
        background-color: rgba(245, 245, 220, 0.85);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #556B2F;
        margin-bottom: 30px;
        margin-top: 30px;
    }}
    
    .section-title {{
        color: #556B2F !important;
        font-size: 28px !important;
        font-weight: bold !important;
        text-align: center;
        margin-bottom: 20px !important;
        background-color: transparent !important;
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

set_background("./assets/background.jpg")

# -------------------------------
# العنوان الرئيسي
# -------------------------------
st.markdown("<h1 style='text-align: center;'>Land Type Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a satellite image for prediction.</p>", unsafe_allow_html=True)

# -------------------------------
# تحميل المودل
# -------------------------------
MODEL_PATH = '../models/eurosat_effb0_best.keras'
DATA_DIR = './data/EuroSAT_RGB'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42
MAX_SAMPLES = 250  # تقليل العدد لزيادة السرعة

@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_cached()

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# -------------------------------
# رفع الصورة والتنبؤ
# -------------------------------
col_upload, col_reset = st.columns([4, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        label="Choose an image...", 
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

with col_reset:
    if uploaded_file is not None:
        st.write("")
        if st.button("Upload New Image", key="reset_upload"):
            st.session_state.pop('file_uploader', None)
            st.rerun()

if uploaded_file:
    file_details = f"{uploaded_file.name} — {round(uploaded_file.size / 1024, 1)} KB"
    st.markdown(f"<div class='uploaded-info'>{file_details}</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # التنبؤ بالمودل
    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        pred_probs = prediction[0]
    
    # عرض النتيجة
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
    st.markdown(f"<p class='prediction-class'>{predicted_class}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # عرض الـ Plot
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        fig_bar = plot_prediction_probs(pred_probs, class_names)
        st.pyplot(fig_bar)

# -------------------------------
# تحميل Dataset للـ Visualization (محسّن للسرعة)
# -------------------------------
@st.cache_data
def load_validation_data():
    with st.spinner("Loading validation data (one time only)..."):
        val_ds = image_dataset_from_directory(
            DATA_DIR,
            labels='inferred',
            label_mode='categorical',
            validation_split=VAL_SPLIT,
            subset='validation',
            seed=SEED,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
        class_names_ds = val_ds.class_names
        
        # تقليل عدد الـ samples لـ 500 فقط لزيادة السرعة
        y_true, y_pred, pred_probs = [], [], []
        sample_count = 0
        dataset_list = []
        
        for images, labels in val_ds:
            if sample_count >= MAX_SAMPLES:
                break
            
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            pred_probs.extend(preds)
            
            dataset_list.append((images, labels))
            sample_count += len(images)
        
        return dataset_list, class_names_ds, np.array(y_true), np.array(y_pred), np.array(pred_probs)

dataset_list, class_names_ds, y_true, y_pred, pred_probs_val = load_validation_data()

# -------------------------------
# Sample Predictions
# -------------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Sample Predictions</h2>", unsafe_allow_html=True)

if 'sample_key' not in st.session_state:
    st.session_state.sample_key = 0

if st.button("Show New Samples", key="refresh_samples"):
    st.session_state.sample_key += 1

@st.cache_data
def get_sample_predictions(_dataset_list, _y_true, _y_pred, _class_names, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    # التأكد من عدم تجاوز عدد العينات المتاحة
    max_available = len(_y_true)
    n_samples = min(12, max_available)
    indices = random.sample(range(max_available), n_samples)
    
    for i, idx in enumerate(indices):
        batch_idx = idx // BATCH_SIZE
        img_idx = idx % BATCH_SIZE
        
        if batch_idx < len(_dataset_list):
            img = _dataset_list[batch_idx][0].numpy()[img_idx]
            
            axes[i].imshow(img.astype('uint8'))
            axes[i].axis('off')
            
            true_class = _class_names[_y_true[idx]]
            pred_class = _class_names[_y_pred[idx]]
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}",
                              color='green' if true_class == pred_class else 'red', fontsize=10)
    
    plt.tight_layout()
    return fig

fig_sample = get_sample_predictions(dataset_list, y_true, y_pred, class_names_ds, st.session_state.sample_key)
st.pyplot(fig_sample)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Random Image Prediction
# -------------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Random Image Prediction</h2>", unsafe_allow_html=True)

if 'random_key' not in st.session_state:
    st.session_state.random_key = 0

if st.button("Show Random Image", key="refresh_random"):
    st.session_state.random_key += 1

@st.cache_data
def get_random_prediction(_dataset_list, _y_true, _y_pred, _pred_probs, _class_names, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    random_idx = random.randint(0, len(_y_true) - 1)
    batch_idx = random_idx // BATCH_SIZE
    img_idx = random_idx % BATCH_SIZE
    
    if batch_idx < len(_dataset_list):
        img = _dataset_list[batch_idx][0].numpy()[img_idx]
    else:
        # في حالة تجاوز العدد، استخدم آخر batch
        batch_idx = len(_dataset_list) - 1
        img_idx = 0
        img = _dataset_list[batch_idx][0].numpy()[img_idx]
        random_idx = batch_idx * BATCH_SIZE + img_idx
    
    true_class = _class_names[_y_true[random_idx]]
    pred_class = _class_names[_y_pred[random_idx]]
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(img.astype('uint8'))
    ax1.axis('off')
    ax1.set_title(f"True: {true_class}\nPred: {pred_class}",
                  color='green' if true_class == pred_class else 'red', fontsize=14)
    
    probs = _pred_probs[random_idx]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(_class_names, probs, color='skyblue')
    ax2.set_xticks(range(len(_class_names)))
    ax2.set_xticklabels(_class_names, rotation=45)
    ax2.set_title(f"Prediction Probabilities\n(True: {true_class}, Pred: {pred_class})", fontsize=12)
    ax2.set_ylabel("Probability")
    
    return fig1, fig2

fig_rand, fig_bar_rand = get_random_prediction(dataset_list, y_true, y_pred, pred_probs_val, class_names_ds, st.session_state.random_key)

col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig_rand)

with col2:
    st.pyplot(fig_bar_rand)

st.markdown("</div>", unsafe_allow_html=True)