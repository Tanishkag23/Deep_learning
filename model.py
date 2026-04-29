import os
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

model = load_model(MODEL_PATH)


def predict(image_path):
    # use preprocessing from separate file
    image = preprocess_image(image_path)

    prediction = model.predict(image)[0][0]

    if prediction > 0.6:
        return "Real", float(prediction)
    elif prediction < 0.4:
        return "Fake", float(prediction)
    else:
        return "Uncertain", float(prediction)