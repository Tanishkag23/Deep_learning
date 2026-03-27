import cv2
import numpy as np

def image_preprocessing(image):
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (128, 128))

    # Normalize
    img = img / 255.0

    return img