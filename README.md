# 🛡️ Fraud Shield — Fraud Detection on Imbalanced Data

An interactive Streamlit app that detects fraudulent financial transactions with machine learning, treating class imbalance as the core challenge.


## ✨ Features
- **Class imbalance analysis:** percentage of fraudulent transactions, pie and log-scale bar charts
- **EDA:** transaction amounts for fraud vs legit, time-of-day pattern
- **Accuracy trap:** shows why plain accuracy is misleading on imbalanced data
- **Imbalance handling:** SMOTE oversampling, random undersampling, `class_weight='balanced'`, or a no-handling baseline
- **Stratified train/test split** so fraud appears in both sets
- **Two models:** Logistic Regression and Random Forest
- **Evaluation:** Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC and Precision-Recall curves
- **Feature importance** for both models
- **Threshold tuning:** pick the decision threshold using the cost of missed fraud vs false alarms
- **Scalability:** how the model could handle 1 million transactions per hour, with measured prediction speed

## 🛠️ Tech stack
Python · pandas · NumPy · scikit-learn · imbalanced-learn · Plotly · Streamlit

## 📂 Dataset
The app runs immediately on built-in synthetic demo data. For real results, download **Credit Card Fraud Detection** (`creditcard.csv`, 284,807 transactions, 492 fraudulent) from Kaggle, choose **Upload creditcard.csv** in the sidebar and keep `Class` as the label column.

## 🚀 Run locally
```bash
git clone https://github.com/mdamaanali-ai/FUTURE_DS_03.git
cd FUTURE_DS_03
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project structure
```
├── app.py                   # Streamlit application
├── requirements.txt
├── .streamlit/config.toml   # dark theme
└── README.md
```

## 📊 Results
_Run the app on the Kaggle dataset and add your own numbers, for example:_

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | – | – | – | – |
| Random Forest | – | – | – | – |

## 🔍 Key takeaways
- Accuracy is misleading here: a model that always says "legit" scores very high yet catches no fraud.
- Recall matters most for fraud, but Precision keeps false alarms manageable. The threshold tab shows the trade-off.
- SMOTE and scaling are fitted on the training data only, to avoid leakage.


## LIVE DEMO
<p align="center">
  <a href="https://frauddetectiononaheavilyimbalanceddatase-m5b9owskyssvts9kbo7bp.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_LAUNCH-LIVE_DEMO-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch Live Demo"/>
  </a>
</p>

## 👤 Author
**mdamaanali-ai** · [GitHub](https://github.com/mdamaanali-ai)

Built as part of the Future Interns Data Science internship (Level 2, Task 3).
