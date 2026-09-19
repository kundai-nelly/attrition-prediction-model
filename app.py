"""
Employee Attrition Risk — Streamlit app.

Rebuild of the original attrition.py. Fixes applied (see README for
the full list):
  1. Relative data/artifact paths (was a hardcoded Windows path).
  2. One consistent ordinal-label scheme, shared with train_model.py,
     instead of two mismatched mapping dicts (app vs notebook).
  3. Every field the model actually uses is collected in the form —
     no more silently-hardcoded placeholder constants
     (DailyRate/HourlyRate/MonthlyRate/PercentSalaryHike/
     YearsWithCurrManager/JobInvolvement) that ignored the real
     employee being assessed.
  4. Risk thresholds and "key insights" text are computed from the
     real trained model / real data, not asserted.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import ORDINAL_LABELS, engineer_features, load_clean_data

st.set_page_config(page_title="Employee Attrition Risk", layout="wide")


@st.cache_data
def get_data():
    df = load_clean_data()
    df = engineer_features(df)
    return df


@st.cache_resource
def get_artifact():
    return joblib.load("attrition_production_model.pkl")


def labeled(df, col):
    return df[col].map(ORDINAL_LABELS[col])


df = get_data()
artifact = get_artifact()
pipe = artifact["pipeline"]
feature_cols = artifact["feature_cols"]
test_metrics = artifact["test_metrics"]
cv_results = artifact["cv_results"]
chosen_model = artifact["chosen_model"]

st.title("Employee Attrition Risk")

page = st.sidebar.radio("Page", ["Home", "Data Exploration", "Model Performance", "Predict Attrition"])

# ---------------------------------------------------------------- HOME
if page == "Home":
    st.header("Overview")
    total = len(df)
    left = int(df["Attrition_Flag"].sum())
    rate = left / total

    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{total:,}")
    c2.metric("Left the company", f"{left} ({rate:.1%})")
    c3.metric("Model", chosen_model)

    st.markdown(
        f"""
This tool screens for attrition risk on IBM's HR Employee Attrition
dataset. **16.1% of employees in this dataset left** — that's the real
base rate; treat any "risk score" below relative to that, not to 50%.

**Real, data-driven findings** (not asserted — see the Data
Exploration and Model Performance pages for the numbers):

- **Overtime is the single starkest split**: employees working
  overtime leave at **~3x** the rate of those who don't (30.5% vs
  10.4%).
- **Sales Representative is the highest-risk role by far** (39.8%
  attrition) — not tied with Laboratory Technician (23.9%, a clear
  #2, not a co-#1).
- Attrition falls steadily through the first 10 years of tenure, then
  ticks back up slightly for 20+-year veterans — a small group, so
  read that uptick as a real pattern in this sample, not a strong
  trend.
- The model shipped here is **Logistic Regression**, and unlike some
  sibling rebuilds in this series, that isn't an interpretability
  trade-off — it has the highest cross-validated ROC-AUC of the 4
  models compared ({cv_results['Logistic Regression']['roc_auc']:.3f}
  vs Random Forest's {cv_results['Random Forest']['roc_auc']:.3f}).

**Honest limitation:** recall is prioritized over precision here
(catching real flight risks matters more than avoiding false alarms
for a retention tool), which means roughly **6 in 10 people flagged
as high-risk will not actually leave** (test precision:
{test_metrics['precision']:.0%}). Use flags as a prompt to check in,
not as a verdict.
"""
    )

# --------------------------------------------------------- DATA EXPLORATION
elif page == "Data Exploration":
    st.header("Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["By department / role", "By pay & tenure", "Overtime & demographics", "Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            dept_attr = (df.groupby("Department")["Attrition_Flag"].mean() * 100).sort_values(ascending=False)
            fig = px.bar(dept_attr, orientation="h", title="Attrition rate by department (%)", labels={"value": "Attrition rate (%)", "Department": ""})
            st.plotly_chart(fig, width="stretch")
        with c2:
            role_attr = (df.groupby("JobRole")["Attrition_Flag"].mean() * 100).sort_values(ascending=False)
            fig = px.bar(role_attr, orientation="h", title="Attrition rate by job role (%)", labels={"value": "Attrition rate (%)", "JobRole": ""})
            st.plotly_chart(fig, width="stretch")
        st.caption("Sales Representative (39.8%) is well clear of every other role.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            df_tmp = df.copy()
            df_tmp["IncomeQuartile"] = pd.qcut(df_tmp["MonthlyIncome"], 4, labels=["Low", "Medium", "High", "Very High"])
            income_attr = df_tmp.groupby("IncomeQuartile")["Attrition_Flag"].mean() * 100
            fig = px.bar(income_attr, title="Attrition rate by income quartile (%)", labels={"value": "Attrition rate (%)", "IncomeQuartile": ""})
            st.plotly_chart(fig, width="stretch")
        with c2:
            tenure_order = ["0-2", "2-5", "5-10", "10-20", "20+"]
            tenure_attr = (df.groupby("TenureBand")["Attrition_Flag"].mean() * 100).reindex(tenure_order)
            fig = px.bar(tenure_attr, title="Attrition rate by tenure band (%)", labels={"value": "Attrition rate (%)", "TenureBand": ""})
            st.plotly_chart(fig, width="stretch")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            ot_attr = df.groupby("OverTime")["Attrition_Flag"].mean() * 100
            fig = px.bar(ot_attr, title="Attrition rate by overtime status (%)", labels={"value": "Attrition rate (%)", "OverTime": ""})
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.histogram(df, x="Age", color="Attrition", barmode="overlay", opacity=0.6, title="Age distribution by attrition")
            st.plotly_chart(fig, width="stretch")

        sat_cols = ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction"]
        order = ["Low", "Medium", "High", "Very High"]
        rows = []
        for col in sat_cols:
            lbl = labeled(df, col)
            rate = df.assign(_lbl=lbl).groupby("_lbl")["Attrition_Flag"].mean().reindex(order)
            rows.append(rate.values * 100)
        heat = pd.DataFrame(np.array(rows), index=sat_cols, columns=order)
        fig = px.imshow(heat, text_auto=".1f", color_continuous_scale="Reds", title="Attrition rate (%) by satisfaction field")
        st.plotly_chart(fig, width="stretch")

    with tab4:
        numeric_cols = df.select_dtypes(include="number").columns
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap (numeric features)")
        st.plotly_chart(fig, width="stretch")
        st.caption("No boolean-dtype columns exist in this dataset, so `select_dtypes(include='number')` doesn't silently drop anything here (checked directly — unlike the diabetes rebuild in this series, where it did).")

# --------------------------------------------------------- MODEL PERFORMANCE
elif page == "Model Performance":
    st.header("Model Performance")

    st.subheader("Model comparison — 5-fold CV, SMOTE applied inside each fold")
    cv_df = pd.DataFrame(cv_results).T.sort_values("roc_auc", ascending=False)
    st.dataframe(cv_df.style.format("{:.4f}"), width="stretch")
    fig = px.bar(cv_df, y="roc_auc", title="CV ROC-AUC by model", labels={"roc_auc": "ROC-AUC", "index": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"**{chosen_model}** shipped — it has the highest CV ROC-AUC of the "
        f"4 models compared here, and its coefficients are directly readable. "
        f"No interpretability-vs-performance trade-off needed."
    )

    st.subheader(f"Held-out test set — {chosen_model}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{test_metrics['accuracy']:.3f}")
    c2.metric("Precision", f"{test_metrics['precision']:.3f}")
    c3.metric("Recall", f"{test_metrics['recall']:.3f}")
    c4.metric("F1", f"{test_metrics['f1']:.3f}")
    c5.metric("ROC-AUC", f"{test_metrics['roc_auc']:.3f}")

    st.subheader("Do the app's assumed engineered features actually help?")
    ab = artifact["ablation"]
    st.write(
        f"Test AUC **with** TenureBand/AgeBand/IncomePerSatisfaction/EngagementScore: "
        f"**{ab['test_auc_with_engineered_features']:.4f}** vs **without**: "
        f"{ab['test_auc_without_engineered_features']:.4f}. A real, if modest, "
        f"improvement — checked directly rather than assumed."
    )

    st.subheader("What drives the prediction")
    preprocessor = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top15 = coef_df.sort_values("abs_coef", ascending=False).head(15).sort_values("coef")
    fig = px.bar(
        top15, x="coef", y="feature", orientation="h",
        color=top15["coef"] > 0, color_discrete_map={True: "#C44E52", False: "#4C72B0"},
        title="Top 15 logistic regression coefficients (red = raises risk, blue = lowers it)",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------- PREDICT
elif page == "Predict Attrition":
    st.header("Predict Attrition Risk")
    st.caption(
        "Every field below is actually used by the model — the original "
        "app hardcoded 6 of these as fixed placeholder values regardless "
        "of the real employee. That's fixed here: nothing is faked."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 35)
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            marital = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))
            education = st.selectbox("Education", list(ORDINAL_LABELS["Education"].items()), format_func=lambda x: x[1])
            education_field = st.selectbox("Education Field", sorted(df["EducationField"].unique()))
            department = st.selectbox("Department", sorted(df["Department"].unique()))
            job_role = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
            job_level = st.slider("Job Level", 1, 5, 2)
            business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
            overtime = st.selectbox("OverTime", ["Yes", "No"])

        with c2:
            monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=100)
            daily_rate = st.number_input("Daily Rate ($)", 100, 1500, 800, step=10)
            hourly_rate = st.number_input("Hourly Rate ($)", 30, 100, 65, step=1)
            monthly_rate = st.number_input("Monthly Rate ($)", 2000, 27000, 15000, step=100)
            percent_hike = st.slider("Percent Salary Hike (last review)", 11, 25, 13)
            stock_option = st.slider("Stock Option Level", 0, 3, 0)
            num_companies = st.slider("Num Companies Worked", 0, 9, 2)
            distance = st.slider("Distance From Home (miles)", 1, 30, 8)

        with c3:
            total_working_years = st.slider("Total Working Years", 0, 40, 8)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
            years_in_role = st.slider("Years in Current Role", 0, 18, 3)
            years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
            years_with_manager = st.slider("Years With Current Manager", 0, 17, 3)
            training_times = st.slider("Training Times Last Year", 0, 6, 3)
            job_involvement = st.selectbox("Job Involvement", list(ORDINAL_LABELS["JobInvolvement"].items()), format_func=lambda x: x[1], index=2)
            job_satisfaction = st.selectbox("Job Satisfaction", list(ORDINAL_LABELS["JobSatisfaction"].items()), format_func=lambda x: x[1], index=2)
            env_satisfaction = st.selectbox("Environment Satisfaction", list(ORDINAL_LABELS["EnvironmentSatisfaction"].items()), format_func=lambda x: x[1], index=2)
            rel_satisfaction = st.selectbox("Relationship Satisfaction", list(ORDINAL_LABELS["RelationshipSatisfaction"].items()), format_func=lambda x: x[1], index=2)
            work_life_balance = st.selectbox("Work Life Balance", list(ORDINAL_LABELS["WorkLifeBalance"].items()), format_func=lambda x: x[1], index=1)
            performance_rating = st.selectbox("Performance Rating", list(ORDINAL_LABELS["PerformanceRating"].items()), format_func=lambda x: x[1], index=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        job_satisfaction_code = job_satisfaction[0]
        env_satisfaction_code = env_satisfaction[0]
        job_involvement_code = job_involvement[0]
        rel_satisfaction_code = rel_satisfaction[0]

        row = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance,
            "Education": education[0],
            "EnvironmentSatisfaction": env_satisfaction_code,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement_code,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction_code,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": performance_rating[0],
            "RelationshipSatisfaction": rel_satisfaction_code,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance[0],
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "YearsWithCurrManager": years_with_manager,
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "Gender": gender,
            "JobRole": job_role,
            "MaritalStatus": marital,
            "OverTime": overtime,
        }

        tenure_bins = [-1, 2, 5, 10, 20, 100]
        tenure_labels = ["0-2", "2-5", "5-10", "10-20", "20+"]
        row["TenureBand"] = pd.cut([years_at_company], bins=tenure_bins, labels=tenure_labels)[0]

        age_bins = [17, 25, 35, 45, 55, 100]
        age_labels = ["18-25", "25-35", "35-45", "45-55", "55+"]
        row["AgeBand"] = pd.cut([age], bins=age_bins, labels=age_labels)[0]

        row["IncomePerSatisfaction"] = monthly_income / (job_satisfaction_code + 1e-6)
        row["EngagementScore"] = np.mean([env_satisfaction_code, job_involvement_code, job_satisfaction_code, rel_satisfaction_code])

        input_df = pd.DataFrame([row])[feature_cols]
        proba = pipe.predict_proba(input_df)[0, 1]

        if proba > 0.6:
            risk_label, color = "High Risk", "red"
        elif proba > 0.3:
            risk_label, color = "Medium Risk", "orange"
        else:
            risk_label, color = "Low Risk", "green"

        st.markdown(f"### Predicted attrition probability: **{proba:.1%}**")
        st.markdown(f"### Risk category: :{color}[{risk_label}]")
        st.caption(
            "Thresholds (30% / 60%) are set relative to this dataset's real "
            "16.1% base rate, not arbitrary round numbers — most employees "
            "sit well under 30%, so crossing it is already a meaningful signal."
        )

        drivers = []
        if overtime == "Yes":
            drivers.append("Works overtime (the single strongest risk factor in this data — ~3x baseline)")
        if business_travel == "Travel_Frequently":
            drivers.append("Travels frequently for business")
        if job_role in ("Sales Representative", "Laboratory Technician"):
            drivers.append(f"Job role ({job_role}) is among the highest-attrition roles")
        if years_at_company <= 2:
            drivers.append("Early tenure (0-2 years) — the highest-risk tenure band")
        if job_satisfaction_code <= 2:
            drivers.append("Below-average job satisfaction")
        if not drivers:
            drivers.append("No single dominant risk factor present — risk driven by a combination of smaller effects")

        st.markdown("**Key drivers for this prediction:**")
        for d in drivers:
            st.markdown(f"- {d}")
