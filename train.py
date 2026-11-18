"""Train SafeNet text + URL models and report metrics.
Uses data/text.csv (text,label) and data/urls.csv (url,label); generates synthetic seed data if missing.
label: 0 = legitimate, 1 = phishing/scam."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from app.features import url_features

ROOT = Path(__file__).parent
DATA, MODELS = ROOT / "data", ROOT / "models"
MODELS.mkdir(exist_ok=True)


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
    }


def main():
    if not (DATA / "text.csv").exists() or not (DATA / "urls.csv").exists():
        from app import make_sample_data  # noqa: F401  (writes CSVs on import)

    report = {}

    # ---- Text model: TF-IDF + Logistic Regression ----
    t = pd.read_csv(DATA / "text.csv").dropna().drop_duplicates()
    Xtr, Xte, ytr, yte = train_test_split(t["text"], t["label"], test_size=0.2, stratify=t["label"], random_state=7)
    text_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    text_model.fit(Xtr, ytr)
    report["text_model"] = metrics(yte, text_model.predict(Xte))
    joblib.dump(text_model, MODELS / "text_model.joblib")

    # ---- URL model: lexical features + Random Forest ----
    u = pd.read_csv(DATA / "urls.csv").dropna().drop_duplicates()
    X = np.array([url_features(x) for x in u["url"]])
    Xtr, Xte, ytr, yte = train_test_split(X, u["label"], test_size=0.2, stratify=u["label"], random_state=7)
    url_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=7, n_jobs=-1)
    url_model.fit(Xtr, ytr)
    report["url_model"] = metrics(yte, url_model.predict(Xte))
    joblib.dump(url_model, MODELS / "url_model.joblib")

    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
