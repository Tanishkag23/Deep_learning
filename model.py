import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(features):
    model = load_model()
    prediction = model.predict([features])

    if prediction[0] == 1:
        return "Fake"
    else:
        return "Real"