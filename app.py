import sys
from pathlib import Path
from io import BytesIO

import streamlit as st

from PIL import Image

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from predict_forgery import (
    load_models,
    predict_classification,
    predict_segmentation,
    create_overlay
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Image Forgery Detection System",
    page_icon="🔍",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2563eb;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: gray;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f4f8;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def get_models():

    classification_model, segmentation_model = load_models()

    return classification_model, segmentation_model


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Image Forgery & Tampering Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Image Classification and Tampering Localization</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About Project")

st.sidebar.write(
    """
    This application detects possible image manipulation
    using deep learning models.
    """
)

st.sidebar.subheader("Models Used")

st.sidebar.write("1. ResNet18 - Image Classification")

st.sidebar.write("2. U-Net - Tampering Localization")

st.sidebar.divider()

st.sidebar.info(
    "The localization output is experimental and may not "
    "accurately identify the actual manipulated region."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image to analyze",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    try:

        original_image = Image.open(uploaded_file).convert("RGB")

        st.success("Image uploaded successfully!")

        st.write(
            f"Image dimensions: {original_image.width} × "
            f"{original_image.height}"
        )

        st.subheader("Original Image")

        st.image(
            original_image,
            use_container_width=True
        )

        st.divider()

        if st.button("Analyze Image", type="primary"):

            try:

                with st.spinner("Loading AI models..."):

                    classification_model, segmentation_model = get_models()

                with st.spinner("Analyzing image..."):

                    # Classification
                    label, confidence = predict_classification(
                        classification_model,
                        original_image
                    )

                    # Segmentation
                    predicted_mask = predict_segmentation(
                        segmentation_model,
                        original_image
                    )

                    # Overlay
                    overlay = create_overlay(
                        original_image,
                        predicted_mask
                    )

                # ==================================
                # DISPLAY CLASSIFICATION RESULTS
                # ==================================

                st.divider()

                st.subheader("Classification Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Prediction",
                        label
                    )

                with col2:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                if label == "Tampered":

                    st.error("Possible image manipulation detected.")

                else:

                    st.success("The model classified this image as Authentic.")

                # ==================================
                # DISPLAY SEGMENTATION RESULTS
                # ==================================

                st.divider()

                st.subheader("Tampering Localization")

                col1, col2 = st.columns(2)

                with col1:

                    st.write("Predicted Tampering Mask")

                    st.image(
                        predicted_mask,
                        use_container_width=True
                    )

                with col2:

                    st.write("Highlighted Image")

                    st.image(
                        overlay,
                        use_container_width=True
                    )

                # ==================================
                # DOWNLOAD RESULTS
                # ==================================

                st.divider()

                st.subheader("Download Results")

                mask_buffer = BytesIO()

                predicted_mask.save(
                    mask_buffer,
                    format="PNG"
                )

                overlay_buffer = BytesIO()

                overlay.save(
                    overlay_buffer,
                    format="PNG"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.download_button(
                        label="Download Predicted Mask",
                        data=mask_buffer.getvalue(),
                        file_name="predicted_mask.png",
                        mime="image/png"
                    )

                with col2:

                    st.download_button(
                        label="Download Highlighted Image",
                        data=overlay_buffer.getvalue(),
                        file_name="tampering_overlay.png",
                        mime="image/png"
                    )

                # ==================================
                # DISCLAIMER
                # ==================================

                st.warning(
                    """
                    Note: Classification confidence is the model's
                    estimated probability, not a guarantee of correctness.
                    The U-Net localization mask is experimental and
                    should not be treated as verified forensic evidence.
                    """
                )

            except Exception as error:

                st.error("An error occurred during image analysis.")

                st.exception(error)

    except Exception as error:

        st.error("Unable to read the uploaded image.")

        st.exception(error)


else:

    st.info("Please upload an image to begin analysis.")