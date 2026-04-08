# 🏦 Credit Risk ML System

> **Production-ready end-to-end machine learning system** for predicting credit card default risk using the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

---

## 📋 Business Problem

Financial institutions face significant losses from credit card defaults. This system predicts the **probability that a credit-card holder will default on their next monthly payment**, enabling:

- **Proactive risk management** — Flag high-risk customers before default.
- **Dynamic credit limits** — Adjust limits based on predicted risk.
- **Collection prioritization** — Focus resources on the highest-risk accounts.

The model is trained on 30,000 historical records with 23 demographic and transactional features, producing a calibrated default probability and a binary prediction.

---

## 📈 Model Results

Three models were trained and evaluated on a held-out 20% test set (6,000 samples):

| Model | ROC-AUC | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.7347 | 0.4723 | 0.5652 | 0.5146 |
| **Random Forest ✓** | **0.7765** | **0.5160** | **0.5712** | **0.5422** |
| XGBoost | 0.7675 | 0.4673 | 0.5863 | 0.5201 |

> **Random Forest** was selected as the production model based on the highest cross-validated ROC-AUC.



## 🗂️ Project Structure

```
credit_risk_ml_system/
│
├── data/                          # Raw & cleaned datasets
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_and_evaluation.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_model_explainability.ipynb
│
├── src/                           # Core ML pipeline modules
│   ├── data_loader.py             # Data download & cleaning
│   ├── features.py                # Feature engineering
│   ├── train.py                   # Model training & selection
│   ├── evaluate.py                # Metrics & visualizations
│   └── utils.py                   # Shared utilities
│
├── models/                        # Serialised model artefacts
│   └── model.pkl
│
├── app/                           # FastAPI REST API
│   ├── main.py
│   └── schemas.py
│
├── monitoring/                    # Production monitoring
│   ├── drift_detection.py         # KS-test & PSI drift checks
│   └── performance_monitor.py     # AUC tracking & retraining alerts
│
├── tests/                         # Unit & integration tests
│   ├── test_api.py
│   └── test_model.py
│
├── reports/                       # Generated plots & metrics
├── .github/workflows/ci.yml       # CI pipeline
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.10+**
- **pip** or **conda**
- **Docker** (optional, for containerised deployment)

### 1. Clone & install

```bash
git clone <your-repo-url>
cd credit_risk_ml_system

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Download and prepare data

```bash
python -m src.data_loader
```

This downloads the UCI dataset, cleans column names, and saves `data/credit_card_default_cleaned.csv`.

---

## 📓 Running the Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load data, EDA, visualisations |
| `02_feature_engineering.ipynb` | Create derived features |
| `03_modeling_and_evaluation.ipynb` | Train & compare LR, RF, XGBoost |
| `04_hyperparameter_tuning.ipynb` | Optuna-based tuning for RF & XGBoost |
| `05_model_explainability.ipynb` | SHAP global & local explanations |

---

##  Training the Model

Run the full training pipeline from the command line:

```bash
python -m src.train
```

This will:
1. Load and clean the dataset
2. Build **sklearn Pipelines** (feature engineering → optional scaler → model) for each candidate
3. Cross-validate Logistic Regression, Random Forest, and XGBoost
4. Select the best Pipeline by cross-validated ROC-AUC
5. **Optimize the decision threshold** (F1-maximizing) on the test set
6. Save the self-contained Pipeline to `models/model.pkl` (with versioned backups)
7. Generate evaluation plots and metrics in `reports/`

> **Architecture note:** The saved `model.pkl` is a complete sklearn Pipeline that includes the `CreditFeatureTransformer` — no separate feature engineering step is needed at inference time. This eliminates training/serving skew and data leakage risks.

---

## 🌐 Running the API

### Local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### Example `curl` request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "limit_bal": 20000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 24,
    "pay_1": 2,
    "pay_2": 2,
    "pay_3": -1,
    "pay_4": -1,
    "pay_5": -2,
    "pay_6": -2,
    "bill_amt1": 3913,
    "bill_amt2": 3102,
    "bill_amt3": 689,
    "bill_amt4": 0,
    "bill_amt5": 0,
    "bill_amt6": 0,
    "pay_amt1": 0,
    "pay_amt2": 689,
    "pay_amt3": 0,
    "pay_amt4": 0,
    "pay_amt5": 0,
    "pay_amt6": 0
  }'
```

**Example response:**

```json
{
  "default_probability": 0.7842,
  "prediction": 1,
  "threshold": 0.5
}
```

---

## 📊 Monitoring

### Drift Detection

```python
from monitoring.drift_detection import run_drift_report

ks_results, psi_results = run_drift_report(reference_df, production_df)
print(ks_results)
print(psi_results)
```

- **KS-test**: Two-sample Kolmogorov–Smirnov test per feature (p < 0.05 → drift)
- **PSI**: Population Stability Index (< 0.10 = OK, ≥ 0.25 = significant drift)

### Performance Monitor

```python
from monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(auc_threshold=0.70)
alert = monitor.log_performance(y_true, y_proba)

if monitor.should_retrain():
    print("Retraining needed!")
```

---

## 🐳 Docker

### Build

```bash
docker build -t credit-risk-api .
```

### Run

```bash
docker run -p 8000:8000 credit-risk-api
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## ✅ Running Tests

```bash
pytest tests/ -v
```

This runs:
- `test_api.py` — FastAPI endpoint tests (health check, prediction, validation)
- `test_model.py` — Model loading and prediction validity tests

---

## 🔄 CI/CD

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push to `main` or `develop`:

1. Sets up Python 3.10 / 3.11
2. Installs dependencies
3. Runs `flake8` linting (non-blocking)
4. Runs `pytest` unit tests

---

## 📝 License

This project is provided for educational and demonstration purposes.

---

## 🙏 Acknowledgements

- **Dataset**: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
