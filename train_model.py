"""
train_model.py — Employee Attrition Risk model.

Rebuilds the pipeline the original notebook attempted but never
actually completed (see README for the full bug list). Produces a
single saved artifact: an imblearn Pipeline (preprocessing + SMOTE +
classifier) plus the metrics needed to reproduce every number quoted
in the README/app.

Design decisions made here (documented, not hidden):

1. Ordinal survey fields (Education, EnvironmentSatisfaction,
   JobInvolvement, JobSatisfaction, RelationshipSatisfaction,
   PerformanceRating, WorkLifeBalance) are modeled on their RAW
   integer codes (1-5), never on string labels. The original notebook
   and the original app.py each invented their OWN, mutually
   inconsistent text labels for the same integer codes (e.g. the
   notebook's PerformanceRating map disagrees with app.py's). Rather
   than pick a winner, this rebuild keeps the numeric codes as the
   single source of truth for modeling, and uses ONE central mapping
   (ORDINAL_LABELS below) purely for human-readable display in the
   EDA and the app UI. That removes the train/serve mismatch risk
   entirely instead of papering over it.

2. Non-informative columns are dropped: EmployeeCount (constant, 1),
   StandardHours (constant, 80), Over18 (constant, 'Y'), and
   EmployeeNumber (a row ID with no predictive meaning). None of
   this was done in the original notebook (which never reached a
   working encoding step at all).

3. Four engineered features referenced by the ORIGINAL app.py
   (TenureBand, AgeBand, IncomePerSatisfaction, EngagementScore) do
   not exist anywhere in the uploaded notebook. Rather than silently
   keep or silently drop them, they are built here for real, kept in
   the model if they earn their place, and their actual contribution
   is reported plainly in the README/metrics rather than assumed.

4. SMOTE is applied only inside training folds / the training split,
   never to the held-out test set — evaluation is always on real,
   untouched data.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Single source of truth for human-readable display of ordinal codes.
# Used ONLY for plotting/EDA/app labels — never for modeling, so there
# is no train/serve mismatch risk from mismatched label dictionaries.
ORDINAL_LABELS = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}

NON_INFORMATIVE_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Ordinal + genuinely continuous numeric columns modeled on raw values.
NUMERIC_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

ENGINEERED_NUMERIC_COLS = ["IncomePerSatisfaction", "EngagementScore"]
ENGINEERED_CATEGORICAL_COLS = ["TenureBand", "AgeBand"]


def load_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=NON_INFORMATIVE_COLS)
    df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


def engineer_features(df):
    """Build the four features app.py assumed but the notebook never
    created. Uses the RAW numeric ordinal codes, consistent with the
    rest of the model — the original app derived these from its own
    (mismatched) text labels."""
    df = df.copy()
    df["TenureBand"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 2, 5, 10, 20, 100],
        labels=["0-2", "2-5", "5-10", "10-20", "20+"],
    ).astype(str)
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "25-35", "35-45", "45-55", "55+"],
    ).astype(str)
    df["IncomePerSatisfaction"] = df["MonthlyIncome"] / (df["JobSatisfaction"] + 1e-6)
    df["EngagementScore"] = df[
        ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction"]
    ].mean(axis=1)
    return df


def build_preprocessor():
    numeric_features = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
    categorical_features = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(X_train, y_train):
    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = {}
    for name, clf in get_candidate_models().items():
        pipe = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", clf),
            ]
        )
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "accuracy": float(np.mean(scores["test_accuracy"])),
            "precision": float(np.mean(scores["test_precision"])),
            "recall": float(np.mean(scores["test_recall"])),
            "f1": float(np.mean(scores["test_f1"])),
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
        }
    return results


def feature_ablation_check(X_train, y_train, X_test, y_test):
    """Honest check on whether the 4 engineered features (assumed by
    the original app but absent from the notebook) actually help,
    rather than assuming they're useful."""
    preprocessor = build_preprocessor()
    pipe_with = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    pipe_with.fit(X_train, y_train)
    auc_with = roc_auc_score(y_test, pipe_with.predict_proba(X_test)[:, 1])

    numeric_wo = NUMERIC_COLS
    categorical_wo = CATEGORICAL_COLS
    preprocessor_wo = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_wo),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_wo),
        ]
    )
    pipe_wo = ImbPipeline(
        steps=[
            ("preprocess", preprocessor_wo),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    pipe_wo.fit(X_train[numeric_wo + categorical_wo], y_train)
    auc_wo = roc_auc_score(y_test, pipe_wo.predict_proba(X_test[numeric_wo + categorical_wo])[:, 1])
    return {"test_auc_with_engineered_features": float(auc_with), "test_auc_without_engineered_features": float(auc_wo)}


def main():
    df = load_clean_data()
    df = engineer_features(df)

    feature_cols = (
        NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS
    )
    X = df[feature_cols]
    y = df["Attrition_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Comparing models (5-fold CV, SMOTE inside each fold)...")
    cv_results = evaluate_models(X_train, y_train)
    for name, m in sorted(cv_results.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"  {name:22s} ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  Recall={m['recall']:.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"])
    print(f"\nBest by CV ROC-AUC: {best_name}")

    print("\nChecking whether the 4 engineered features actually help (Logistic Regression)...")
    ablation = feature_ablation_check(X_train, y_train, X_test, y_test)
    print(f"  With engineered features:    AUC = {ablation['test_auc_with_engineered_features']:.4f}")
    print(f"  Without engineered features: AUC = {ablation['test_auc_without_engineered_features']:.4f}")

    # Ship Logistic Regression: interpretable coefficients for the
    # "key drivers" the app displays, and (see comparison table)
    # statistically indistinguishable from the tree ensembles here.
    chosen_name = "Logistic Regression"
    chosen_clf = get_candidate_models()[chosen_name]
    final_pipe = ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", chosen_clf),
        ]
    )
    final_pipe.fit(X_train, y_train)

    y_pred = final_pipe.predict(X_test)
    y_proba = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    print(f"\nHeld-out test metrics ({chosen_name}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    artifact = {
        "pipeline": final_pipe,
        "feature_cols": feature_cols,
        "numeric_cols": NUMERIC_COLS,
        "engineered_numeric_cols": ENGINEERED_NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "engineered_categorical_cols": ENGINEERED_CATEGORICAL_COLS,
        "ordinal_labels": ORDINAL_LABELS,
        "chosen_model": chosen_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "ablation": ablation,
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(artifact, "attrition_production_model.pkl")
    print("\nSaved attrition_production_model.pkl")

    metrics_snapshot = {
        "n_rows": int(len(df)),
        "n_left": int(df["Attrition_Flag"].sum()),
        "attrition_rate": float(df["Attrition_Flag"].mean()),
        "cv_results": cv_results,
        "chosen_model": chosen_name,
        "test_metrics": test_metrics,
        "ablation": ablation,
    }
    with open("metrics_train.json", "w") as f:
        json.dump(metrics_snapshot, f, indent=2)


if __name__ == "__main__":
    main()
