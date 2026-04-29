import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    # resize image to match model input
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # normalize pixel values
    image = image / 255.0

    # convert to batch format
    image = np.expand_dims(image, axis=0)

    return image