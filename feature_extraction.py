import cv2
import numpy as np

def feature_extract(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    features = np.concatenate([
        gray.flatten(),
        edges.flatten(),
        laplacian.flatten()
    ])

    return features