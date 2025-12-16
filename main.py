from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import re
import numpy as np

app = FastAPI(
    title="Language Detection API",
    description="Detects the language of one or more texts using a trained ML model."
)


# Load artifacts
model = joblib.load("language_detection_pipeline.pkl")
label_encoder = joblib.load("Label_encoder.pkl")

# ---------- Schemas ----------

class SingleTextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

# ---------- Utilities ----------

def clean_text(text: str) -> str:
    text = re.sub(r'[!@#$(),"\n%^*?:~`0-9]', ' ', text)
    text = re.sub(r'[\[\]]', ' ', text)
    return text.lower()

# ---------- Routes ----------

@app.get("/")
def home():
    return {
        "message": "Welcome to the Language Detection API. Use /predict or /predict-batch."
    }

@app.post("/predict")
def predict_language(input: SingleTextInput):
    cleaned_text = clean_text(input.text)

    probs = model.predict_proba([cleaned_text])[0]
    pred_index = np.argmax(probs)

    language = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(probs[pred_index])

    return {
        "predicted_language": language,
        "confidence": round(confidence, 4)
    }

@app.post("/predict-batch")
def predict_languages(input: BatchTextInput):
    cleaned_texts = [clean_text(text) for text in input.texts]

    probs = model.predict_proba(cleaned_texts)
    pred_indices = np.argmax(probs, axis=1)
    languages = label_encoder.inverse_transform(pred_indices)

    results = []
    for text, lang, prob in zip(input.texts, languages, probs):
        confidence = float(np.max(prob))
        results.append({
            "text": text,
            "predicted_language": lang,
            "confidence": round(confidence, 4)
        })

    return {"results": results}





