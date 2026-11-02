
import csv
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "spam_model.pkl"

texts = []
labels = []

# Read the real CSV dataset without pandas
with open(DATASET_PATH, "r", encoding="latin-1", newline="") as file:
    reader = csv.reader(file)

    for row in reader:
        if len(row) < 2:
            continue

        label = row[0].strip().lower()
        message = row[1].strip()

        if label in ("ham", "spam") and message:
            texts.append(message)
            labels.append(label)

if not texts:
    raise ValueError(
        "No valid messages found. Check the CSV file and its format."
    )

print(f"Total messages loaded: {len(texts)}")
print(f"Spam messages: {labels.count('spam')}")
print(f"Normal messages: {labels.count('ham')}")

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"]
))

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

import json

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    predictions,
    labels=["ham", "spam"],
    zero_division=0
)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "not_spam_precision": float(precision[0]),
    "not_spam_recall": float(recall[0]),
    "not_spam_f1": float(f1[0]),
    "spam_precision": float(precision[1]),
    "spam_recall": float(recall[1]),
    "spam_f1": float(f1[1]),
    "test_messages": len(y_test),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ).tolist()
}

with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Evaluation metrics saved to metrics.json")


print(f"\nModel saved successfully: {MODEL_PATH}")
