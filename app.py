from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Classifier API")
model = joblib.load("model.joblib")


class TextIn(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextIn):
    proba = model.predict_proba([payload.text])[0]
    spam_prob = float(proba[1])
    label = "spam" if spam_prob >= 0.5 else "ham"
    return {
        "label": label,
        "spam_probability": round(spam_prob, 4),
    }