import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\AMT\Downloads\statlog+german+credit+data\german.data-numeric"

if not os.path.exists(DATASET_PATH):
    print("Dataset not found!")
    print("Expected Location:", DATASET_PATH)
    exit()

df = pd.read_csv(
    DATASET_PATH,
    sep=r"\s+",
    header=None
)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print("Dataset Shape:", df.shape)
print(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

y = y.replace({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

imputer = SimpleImputer(strategy="median")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

best_model = None
best_auc = 0

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

for name, model in models.items():

    pipeline = Pipeline([
        ("imputer", imputer),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    if auc > best_auc:
        best_auc = auc
        best_model = pipeline

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_credit_model.pkl")
joblib.dump(best_model, MODEL_PATH)

print("\n" + "=" * 60)
print("Best Model Saved Successfully!")
print("Model Location:", MODEL_PATH)
print("Best ROC AUC:", round(best_auc, 4))
print("=" * 60)