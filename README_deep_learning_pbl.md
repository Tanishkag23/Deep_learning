# Deepfake Image Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Status-Active-success)

------------------------------------------------------------------------

## Overview

This project is a deep learning-based web application that detects
whether an image is **Real** or **Fake (Deepfake)**.

It uses **Transfer Learning (MobileNetV2)** to achieve high accuracy and
provides a simple interface for users to upload and test images.

------------------------------------------------------------------------

## Features

-   Binary Classification (Real vs Fake)
-   Confidence Score Output
-   Streamlit Web Interface
-   Modular Code Structure
-   Optimized preprocessing pipeline

------------------------------------------------------------------------

## Project Structure

project_folder/ │ ├── app.py\
├── model.py\
├── preprocessing.py\
├── train_model.py\
├── cnn_model.h5\
│ ├── dataset/ │ └── Final Dataset/ │ ├── Fake/ │ └── Real/

------------------------------------------------------------------------

## Tech Stack

-   Python
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Streamlit

------------------------------------------------------------------------

## Working Flow

### Training

Dataset → Augmentation → MobileNetV2 → Training → Save Model

### Prediction

Image → Preprocessing → Model → Probability → Result

### App

Upload → Predict → Display Result

------------------------------------------------------------------------

## Installation

### 1. Create Virtual Environment

python -m venv venv venv`\Scripts`{=tex}`\activate`{=tex}

### 2. Install Dependencies

pip install tensorflow opencv-python streamlit numpy

------------------------------------------------------------------------

## Dataset Structure

dataset/ Final Dataset/ Fake/ Real/

------------------------------------------------------------------------

## Training

python train_model.py

Model will be saved as: cnn_model.h5

------------------------------------------------------------------------

## Run Application

streamlit run app.py

Open in browser: http://localhost:8501

------------------------------------------------------------------------

## Output Example

-   Prediction: Real / Fake
-   Confidence: 0.87

------------------------------------------------------------------------

## Expected Accuracy

-   80% -- 90% (depends on dataset quality)

------------------------------------------------------------------------

## Future Improvements

-   Video Deepfake Detection
-   Real-time Webcam Detection
-   Cloud Deployment (Render / AWS)
-   Model Optimization (95%+ accuracy)

------------------------------------------------------------------------

## Author

Developed as a deep learning project using CNN and transfer learning.
