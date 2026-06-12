import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Soil Type Classification",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2E7D32;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:20px;
}

.result-box {
    padding:20px;
    border-radius:15px;
    background-color:#f0f8f0;
    border:2px solid #2E7D32;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<p class="title">🌱 Soil Type Classification System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload a soil image and let AI identify the soil type.</p>',
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "models/soil_classifier.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()

except Exception as e:

    st.error(f"Could not load model: {e}")
    st.stop()

# ==========================================
# IMPORTANT
# MATCH TRAINING ORDER
# ==========================================

categories = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil"
]

IMG_SIZE = 128

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        """
        This application uses a CNN
        model trained on soil images
        to classify soil types.
        """
    )

    st.write("### Supported Formats")
    st.write("✔ JPG")
    st.write("✔ JPEG")
    st.write("✔ PNG")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Soil Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION SECTION
# ==========================================

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    image_np = np.array(image)

    # RGBA -> RGB

    if len(image_np.shape) == 3 and image_np.shape[-1] == 4:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGBA2RGB
        )

    # GRAYSCALE -> RGB

    elif len(image_np.shape) == 2:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )

    image_resized = cv2.resize(
        image_np,
        (IMG_SIZE, IMG_SIZE)
    )

    image_resized = image_resized.astype(
        np.float32
    ) / 255.0

    image_resized = np.expand_dims(
        image_resized,
        axis=0
    )

    # ======================================
    # BUTTON
    # ======================================

    if st.button("🔍 Predict Soil Type"):

        with st.spinner("Analyzing Image..."):

            prediction = model.predict(
                image_resized,
                verbose=0
            )

        class_index = np.argmax(prediction)

        confidence = (
            np.max(prediction) * 100
        )

        predicted_class = categories[
            class_index
        ]

        with col2:

            st.markdown(
                """
                <div class="result-box">
                <h2>Prediction Result</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                f"Predicted Soil Type: {predicted_class}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

        # ==================================
        # PROBABILITY TABLE
        # ==================================

        st.subheader(
            "Prediction Probabilities"
        )

        probability_data = pd.DataFrame({

            "Soil Type": categories,

            "Probability (%)":
                prediction[0] * 100
        })

        st.dataframe(
            probability_data,
            use_container_width=True
        )

        # ==================================
        # BAR CHART
        # ==================================

        st.subheader(
            "Probability Distribution"
        )

        chart_df = probability_data.set_index(
            "Soil Type"
        )

        st.bar_chart(chart_df)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    **AI-Based Soil and Land Type Classification Using Deep Learning**

    Developed using:
    - TensorFlow
    - OpenCV
    - Streamlit
    - CNN Deep Learning Model
    """
)