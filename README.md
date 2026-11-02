# Customer Churn Prediction

A machine learning app that predicts how likely a telecom customer is to leave, so a company can contact high-risk customers first.

**Live demo:** [https://customer-churn-prediction-90.streamlit.app/]

## Dataset
IBM Telco Customer Churn (Kaggle). 7,043 customers, 21 columns. 26.5% of customers churned.

<img width="800" height="500" alt="feature_importance" src="https://github.com/user-attachments/assets/91ea3733-d4df-4ead-a304-733e100ca9dc" />


## What I did
1. Cleaned the data (removed 11 rows with missing TotalCharges, dropped the ID column)
2. Split 80/20 and built a baseline
3. Trained 3 models: logistic regression, random forest, gradient boosting
4. Evaluated with precision, recall, F1 and ROC-AUC
5. Tuned the threshold and checked stability with 5-fold cross-validation
6. Built a Streamlit app

## Results (test set)
| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic regression | 0.648 | 0.572 | 0.608 | 0.836 |
| Random forest | 0.634 | 0.492 | 0.554 | 0.814 |
| Gradient boosting | 0.643 | 0.529 | 0.581 | 0.839 |

The baseline (always predicting "stays") scores 73.4% accuracy but catches no churners, so accuracy alone is misleading.
I chose logistic regression with a 0.4 threshold: precision 0.579, recall 0.684, F1 0.627.

Short tenure, month-to-month contracts and fiber optic internet push toward churn. Two-year contracts push toward staying.

## Limitations
- The threshold was chosen using the test set
- One dataset only
- Charge features overlap, so coefficients should be read carefully

## Run locally
    pip install -r requirements.txt
    python -m streamlit run app.py

**Author:** Mehr Hussain
