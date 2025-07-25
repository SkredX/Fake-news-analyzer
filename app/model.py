import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

def predict_news(text):
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]
    return {
        "verdict": "Real" if label == 1 else "Fake",
        "confidence": round(np.max(proba) * 100, 2)
    }
