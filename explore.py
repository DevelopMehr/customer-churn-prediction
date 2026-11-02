from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt

# pandas is the library for working with data tables.
# "as pd" is a short nickname so we type less later.
import pandas as pd

# Read the CSV file into a DataFrame, which is pandas' word for a table.
# We save it in a variable called "df" so we can reuse it.
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# .shape gives (number of rows, number of columns).
# Each row is one customer. Each column is one fact about that customer.
print("Shape:", df.shape)

# .head() shows the first 5 rows so we can see what the data looks like.
print(df.head())

# .columns lists the name of every column.
# This shows us all 21, which the narrow terminal was hiding.
print(df.columns.tolist())

# .info() shows each column's name, how many values are filled in, and its data type.
# Data types matter: "int64"/"float64" are numbers, "object" is text.
print(df.info())

# --- Step 3: how balanced is the target? ---
# value_counts() counts how many times each value appears in a column.
# normalize=True turns counts into fractions (0.27 = 27%).
# We want to know what share of customers left.
print(df["Churn"].value_counts(normalize=True))

# --- Step 4: fix TotalCharges ---
# It is text because some rows hold a blank space instead of a number.
# to_numeric converts text to numbers. errors="coerce" turns anything
# it can't convert (like a blank) into NaN, which means "missing".
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# isna() marks missing values, sum() counts them per column.
# We print only the columns that have at least one missing value.
missing = df.isna().sum()
print(missing[missing > 0])

# Only a handful of rows are affected, so we drop them.
# dropna() removes rows with missing values. Dropping is fine
# here because losing so few rows won't change the results.
df = df.dropna()

# customerID is just a unique label. It says nothing about whether
# someone leaves, so it would only confuse the model. We remove it.
df = df.drop(columns=["customerID"])

# --- Step 5: turn the target into numbers ---
# Models work with numbers, not "Yes"/"No". We map No -> 0, Yes -> 1.
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Final check: new shape and the first rows.
print("Shape after cleaning:", df.shape)
print(df.head())

# X = the clues (features), y = the answer we predict
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Hold back 20% for testing. stratify keeps the same churn % in both parts.
# random_state=42 makes the split the same every run.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Split columns by type: numbers vs text categories
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Numbers get scaled to a similar range. Text becomes 0/1 columns (one-hot).
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Baseline: a dummy that always guesses the most common answer.
# Any real model must beat this.
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Baseline accuracy:", base.score(X_test, y_test))

# Pipeline = prep + model in one object, so test data gets the same prep.
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)
print("Logistic accuracy:", model.score(X_test, y_test))

# Check: rows left after cleaning (should be 5 fewer if 5 were missing)
print("Rows after cleaning:", len(df))

# Confusion matrix for logistic regression:
# [[correct stays, wrongly flagged], [missed churners, caught churners]]
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))

# Three models, same prep, same split, so the comparison is fair
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

for name, clf in models.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]   # churn probability per customer
    print(
        name,
        "| precision:", round(precision_score(y_test, pred), 3),   # of flagged, how many truly leave
        "| recall:", round(recall_score(y_test, pred), 3),         # of leavers, how many we caught
        "| F1:", round(f1_score(y_test, pred), 3),                 # balance of the two
        "| AUC:", round(roc_auc_score(y_test, proba), 3),          # ranking quality
    )

# 1. class_weight="balanced": the model pays more attention to the rare churners
bal = Pipeline([("prep", prep),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
bal.fit(X_train, y_train)
pred = bal.predict(X_test)
print("Balanced | precision:", round(precision_score(y_test, pred), 3),
      "| recall:", round(recall_score(y_test, pred), 3),
      "| F1:", round(f1_score(y_test, pred), 3))

# 2. Threshold: predict "churn" when probability >= t. Lower t catches more churners.
proba = model.predict_proba(X_test)[:, 1]
for t in [0.5, 0.4, 0.3, 0.2]:
    pred = (proba >= t).astype(int)
    print("threshold", t,
          "| precision:", round(precision_score(y_test, pred), 3),
          "| recall:", round(recall_score(y_test, pred), 3),
          "| F1:", round(f1_score(y_test, pred), 3))

# 3. Cross-validation: train/test on 5 different splits to check the score is stable
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("CV AUC mean:", round(scores.mean(), 3), "| std:", round(scores.std(), 3))

# 4. What drives churn? Logistic regression coefficients:
# positive = pushes toward churn, negative = pushes toward staying.
names = model.named_steps["prep"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
imp = pd.Series(coefs, index=names).sort_values()
print("Pushes toward staying:\n", imp.head(6))
print("Pushes toward churn:\n", imp.tail(6))

import os

THRESHOLD = 0.4

# Final model: same pipeline, trained on all the data
final = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
final.fit(X, y)

# Save everything the app needs in one file
bundle = {
    "model": final,
    "threshold": THRESHOLD,
    "columns": X.columns.tolist(),
    "cat_options": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
    "num_ranges": {c: (float(X[c].min()), float(X[c].max()), float(X[c].median()))
                   for c in num_cols},
}
joblib.dump(bundle, "churn_model.joblib")

# Save a chart for your README
os.makedirs("images", exist_ok=True)
top = pd.concat([imp.head(6), imp.tail(6)])
top.plot(kind="barh", figsize=(8, 5), title="What drives churn")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
print("Saved model and chart")