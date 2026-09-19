"""
generate_figures.py — regenerates every chart used in the README plus
a metrics.json snapshot of every real number quoted there. Run after
train_model.py (needs attrition_production_model.pkl for the model
comparison / coefficient plots).
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from train_model import ORDINAL_LABELS, engineer_features, load_clean_data

plt.rcParams["figure.dpi"] = 110
sns.set_theme(style="whitegrid")

ASSETS = "assets"


def labeled(df, col):
    """Human-readable copy of an ordinal column, for display only."""
    return df[col].map(ORDINAL_LABELS[col]) if col in ORDINAL_LABELS else df[col]


def main():
    df = load_clean_data()
    df = engineer_features(df)
    metrics = {}

    metrics["n_rows"] = int(len(df))
    metrics["n_left"] = int(df["Attrition_Flag"].sum())
    metrics["attrition_rate"] = float(df["Attrition_Flag"].mean())

    # 1. Overall attrition class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Attrition"].value_counts()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, f"{v}\n({v/len(df):.1%})", ha="center")
    ax.set_title(f"Attrition class balance (n={len(df)})")
    ax.set_ylabel("Employees")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/class_balance.png")
    plt.close(fig)

    # 2. Attrition rate by department
    dept_attr = df.groupby("Department")["Attrition_Flag"].mean().sort_values(ascending=False)
    metrics["attrition_by_department"] = {k: float(v) for k, v in dept_attr.items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(dept_attr.index, dept_attr.values * 100, color="#C44E52")
    ax.set_xlabel("Attrition rate (%)")
    ax.set_title("Attrition rate by department")
    for i, v in enumerate(dept_attr.values * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/attrition_by_department.png")
    plt.close(fig)

    # 3. Attrition rate by job role
    role_attr = df.groupby("JobRole")["Attrition_Flag"].mean().sort_values(ascending=False)
    metrics["attrition_by_role"] = {k: float(v) for k, v in role_attr.items()}
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C44E52" if v == role_attr.max() else "#4C72B0" for v in role_attr.values]
    ax.barh(role_attr.index[::-1], role_attr.values[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Attrition rate (%)")
    ax.set_title("Attrition rate by job role")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/attrition_by_role.png")
    plt.close(fig)

    # 4. Attrition rate by income quartile
    df["IncomeQuartile"] = pd.qcut(df["MonthlyIncome"], 4, labels=["Low", "Medium", "High", "Very High"])
    income_attr = df.groupby("IncomeQuartile")["Attrition_Flag"].mean()
    metrics["attrition_by_income_quartile"] = {str(k): float(v) for k, v in income_attr.items()}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(income_attr.index.astype(str), income_attr.values * 100, color="#55A868")
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title("Attrition rate by monthly income quartile")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/attrition_by_income.png")
    plt.close(fig)

    # 5. Attrition rate by tenure band (engineered feature)
    tenure_order = ["0-2", "2-5", "5-10", "10-20", "20+"]
    tenure_attr = df.groupby("TenureBand")["Attrition_Flag"].mean().reindex(tenure_order)
    metrics["attrition_by_tenure_band"] = {k: float(v) for k, v in tenure_attr.items()}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(tenure_attr.index, tenure_attr.values * 100, color="#8172B2")
    ax.set_ylabel("Attrition rate (%)")
    ax.set_xlabel("Years at company")
    ax.set_title("Attrition rate by tenure band")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/attrition_by_tenure.png")
    plt.close(fig)

    # 6. OverTime vs attrition
    ot_attr = df.groupby("OverTime")["Attrition_Flag"].mean()
    metrics["attrition_by_overtime"] = {k: float(v) for k, v in ot_attr.items()}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(ot_attr.index, ot_attr.values * 100, color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Attrition rate (%)")
    ax.set_title("Attrition rate by overtime status")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/attrition_by_overtime.png")
    plt.close(fig)

    # 7. Age distribution by attrition (KDE)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [("No", "#4C72B0"), ("Yes", "#C44E52")]:
        sns.kdeplot(df.loc[df["Attrition"] == label, "Age"], ax=ax, label=label, color=color, fill=True, alpha=0.3)
    ax.set_title("Age distribution by attrition")
    ax.legend(title="Attrition")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/age_kde.png")
    plt.close(fig)

    # 8. Monthly income distribution by attrition (KDE)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [("No", "#4C72B0"), ("Yes", "#C44E52")]:
        sns.kdeplot(df.loc[df["Attrition"] == label, "MonthlyIncome"], ax=ax, label=label, color=color, fill=True, alpha=0.3)
    ax.set_title("Monthly income distribution by attrition")
    ax.legend(title="Attrition")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/income_kde.png")
    plt.close(fig)

    # 9. Correlation heatmap, numeric columns, WITH annotations (original omitted annot=True)
    numeric_cols = df.select_dtypes(include="number").columns
    fig, ax = plt.subplots(figsize=(13, 11))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax, square=True)
    ax.set_title("Correlation heatmap, numeric features")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/correlation_heatmap.png")
    plt.close(fig)

    # 10. Job satisfaction heatmap: satisfaction fields vs attrition rate (kept string-labeled,
    #     unlike the original notebook which silently overwrote JobSatisfaction back to numeric)
    sat_cols = ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction"]
    order = ["Low", "Medium", "High", "Very High"]
    rows = []
    for col in sat_cols:
        lbl = labeled(df, col)
        rate = df.assign(_lbl=lbl).groupby("_lbl")["Attrition_Flag"].mean().reindex(order)
        rows.append(rate.values)
    heat = pd.DataFrame(np.array(rows) * 100, index=sat_cols, columns=order)
    metrics["attrition_by_satisfaction_field"] = heat.round(2).to_dict(orient="index")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={"label": "Attrition rate (%)"})
    ax.set_title("Attrition rate (%) by satisfaction field")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/satisfaction_heatmap.png")
    plt.close(fig)

    # ---- model-dependent figures ----
    artifact = joblib.load("attrition_production_model.pkl")
    cv_results = artifact["cv_results"]
    metrics["cv_results"] = cv_results
    metrics["chosen_model"] = artifact["chosen_model"]
    metrics["test_metrics"] = artifact["test_metrics"]
    metrics["ablation"] = artifact["ablation"]

    # 11. Model comparison bar chart
    names = list(cv_results.keys())
    aucs = [cv_results[n]["roc_auc"] for n in names]
    order_idx = np.argsort(aucs)[::-1]
    names = [names[i] for i in order_idx]
    aucs = [aucs[i] for i in order_idx]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#55A868" if n == artifact["chosen_model"] else "#4C72B0" for n in names]
    ax.bar(names, aucs, color=colors)
    ax.set_ylim(min(aucs) - 0.05, 1.0)
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Model comparison (5-fold CV, SMOTE per fold)")
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/model_comparison.png")
    plt.close(fig)

    # 12. Logistic regression coefficients (top 15 by |coef|)
    pipe = artifact["pipeline"]
    preprocessor = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top15 = coef_df.sort_values("abs_coef", ascending=False).head(15).sort_values("coef")
    metrics["top_coefficients"] = dict(zip(top15["feature"], top15["coef"].round(4)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top15["coef"]]
    ax.barh(top15["feature"], top15["coef"], color=colors)
    ax.set_title("Top 15 logistic regression coefficients\n(red = raises attrition risk, blue = lowers it)")
    ax.set_xlabel("Coefficient (standardized features)")
    fig.tight_layout()
    fig.savefig(f"{ASSETS}/coefficients.png")
    plt.close(fig)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("All figures + metrics.json written.")
    print(json.dumps({k: v for k, v in metrics.items() if k in
                       ["n_rows", "n_left", "attrition_rate", "chosen_model", "test_metrics", "ablation"]}, indent=2))


if __name__ == "__main__":
    main()
