import streamlit as st
from model import predict
import tempfile

st.title("Deepfake Detection App")

st.write("Upload an image to check whether it is fake or real.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    if st.button("Check"):

        result, confidence = predict(temp_path)

        st.write("Confidence:", round(confidence, 3))

        if result == "Real":
            st.success("This image is real")
        elif result == "Fake":
            st.error("This image is fake")
        else:
            st.warning("Prediction is uncertain")