# Spam Classifier API

SMS spam detection using TF-IDF + classical ML, served as a REST API with FastAPI.

## Tech Stack
Python, Scikit-learn, Pandas, TF-IDF, FastAPI

## Dataset
SMS Spam Collection (Kaggle/UCI): 5,572 messages (~13% spam).
403 duplicate rows removed, leaving 5,169. Split 80/20 with stratification.

## Approach
- TF-IDF features (unigrams + bigrams, sublinear TF)
- Compared Logistic Regression and Multinomial Naive Bayes
- Handled class imbalance (`class_weight="balanced"` for LR) and tuned smoothing (`alpha`) for NB
- TF-IDF + classifier wrapped in a single sklearn `Pipeline`, saved with joblib

## Results (spam class, test set of 1,034 messages)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Naive Bayes (baseline: stopwords removed, alpha=1) | 1.00 | 0.63 | 0.78 |
| Logistic Regression (balanced, C=10) | 0.95 | 0.90 | 0.93 |
| **Naive Bayes (alpha=0.1)** | **0.99** | **0.89** | **0.94** |

Final model: Naive Bayes (alpha=0.1), accuracy 98.5%.
Accuracy alone is misleading on imbalanced data, so precision/recall/F1 on the spam class were used for selection.

Note: model selection was done on the hold-out test set; cross-validation would give a less optimistic estimate.

## Run locally

```bash
git clone https://github.com/rohan-kirdak/text-classifier.git
cd text-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## API

`POST /predict`

Request:
```json
{"text": "Congratulations! You won a free iPhone. Click here to claim now!"}
```

Response:
```json
{"label": "spam", "spam_probability": 0.97}
```

`GET /` returns a health check.

## Project structure

```
├── data/spam.csv
├── train.py        # trains and compares models, saves model.joblib
├── app.py          # FastAPI service
├── check_data.py   # quick dataset sanity checks
└── requirements.txt
```