import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

from preprocessing import image_preprocessing
from feature_extraction import feature_extract

data = []
labels = []

# Dataset folders
real_path = "dataset/real"
fake_path = "dataset/fake"

# Load real images
for file in os.listdir(real_path):
    img = cv2.imread(os.path.join(real_path, file))
    if img is not None:
        img = image_preprocessing(img)
        features = feature_extract(img)
        data.append(features)
        labels.append(0)

# Load fake images
for file in os.listdir(fake_path):
    img = cv2.imread(os.path.join(fake_path, file))
    if img is not None:
        img = image_preprocessing(img)
        features = feature_extract(img)
        data.append(features)
        labels.append(1)

# Train model
model = RandomForestClassifier()
model.fit(data, labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")