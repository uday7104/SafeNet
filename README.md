# SafeNet: AI-Powered Phishing & Scam Detection (prototype)

Pipeline: Input -> Pre-process -> Text AI -> URL AI -> Fusion -> Decision -> Guidance

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                                       # trains models, prints metrics (auto-runs on first start too)
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (API docs at /docs).

## API
`POST /analyze` with `{"text": "...", "url": "..."}` (either or both) returns risk score, level, threat type,
recommendation, per-model probabilities and human-readable reasons.

## Structure
| File | Purpose |
|---|---|
| `app/features.py` | URL lexical features, red-flag explanations, text cue detection |
| `train.py` | Trains TF-IDF + Logistic Regression (text) and Random Forest (URL), reports Accuracy/Precision/Recall/F1/FPR |
| `app/fusion.py` | Fuses probabilities, thresholds (LOW/MEDIUM/HIGH), threat type, recommendation |
| `app/main.py` | FastAPI backend |
| `static/index.html` | Web UI |
| `app/make_sample_data.py` | Generates SYNTHETIC seed data |

## IMPORTANT: use real data
The bundled data is small and synthetic (template-based), so its near-perfect metrics are NOT meaningful.
For real results, replace the CSVs (label 1 = phishing/scam, 0 = legitimate) and rerun `python train.py`:
- `data/text.csv` with columns `text,label`: SMS Spam Collection, Nazario / Enron phishing emails, Kaggle phishing-email sets
- `data/urls.csv` with columns `url,label`: PhishTank / OpenPhish (phishing) + Tranco top sites (legit), or Kaggle "Phishing Site URLs"
Then compare against your own held-out data, check the false-positive rate, and tune thresholds in `app/fusion.py`.

## Roadmap (maps to your 28-week plan)
1. Real datasets + cleaning/dedup (weeks 6-8)
2. More features, XGBoost/LightGBM baselines, cross-validation (9-15)
3. DistilBERT/BERT text classifier (16-18)
4. Learned fusion (logistic regression on model outputs) + calibration (19-21)
5. LIME/SHAP explanations, history page, auth (22-23)
6. Tests, usability study, report (24-27)
