import streamlit as st
from PIL import Image

from preprocessing import image_preprocessing
from feature_extraction import feature_extract
from model import predict

st.title("Deepfake Detection System")

up_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if up_file is not None:
    image = Image.open(up_file)
    st.image(image, caption="Uploaded Image")

    # Step 1: Preprocessing
    processed = image_preprocessing(image)
    st.success("Preprocessing Done")

    # Step 2: Feature Extraction
    features = feature_extract(processed)
    st.success("Feature Extraction Done")

    # Step 3: Prediction
    result = predict(features)
    st.success(f"Prediction: {result}")