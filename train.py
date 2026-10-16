import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df = df.drop_duplicates()
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print("Rows after dedup:", len(df))

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

models = {
    "LR (balanced, C=1)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LR (balanced, C=10)": LogisticRegression(max_iter=1000, class_weight="balanced", C=10),
    "NB (alpha=0.1)": MultinomialNB(alpha=0.1),
    "NB (alpha=0.3)": MultinomialNB(alpha=0.3),
}

best_name, best_f1, best_pipe = None, 0, None

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe

print(f"\nBest model: {best_name} (F1={best_f1:.3f})")
joblib.dump(best_pipe, "model.joblib")
print("Saved model.joblib")