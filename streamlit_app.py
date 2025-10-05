# app.py - Streamlit app for Two-Pot Retirement analysis
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, glob, os, io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report, auc
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation

st.set_page_config(layout="wide", page_title="Two-Pot Retirement — Demo Dashboard")
df=pd.read_csv("C:\Users\Cllr. T.Z. Nyawo\Downloads\New folder\employee")
# -------------------------
# Helper utilities
# -------------------------
def extract_first_csv_from_zip(zip_path, extract_dir="/tmp/extracted_dataset"):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    csvs = glob.glob(os.path.join(extract_dir, '**/*.csv'), recursive=True) + glob.glob(os.path.join(extract_dir, '*.csv'))
    csvs = sorted(csvs, key=os.path.getsize, reverse=True)
    return csvs[0] if len(csvs) else None

def load_dataset_from_uploaded(uploaded_file):
    # uploaded_file is an UploadedFile from st.file_uploader
    name = uploaded_file.name.lower()
    if name.endswith(".zip"):
        # save zip to temp, extract a csv
        tmp_zip = "/tmp/uploaded_archive.zip"
        with open(tmp_zip, "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_path = extract_first_csv_from_zip(tmp_zip, extract_dir="/tmp/extracted_from_upload")
        if csv_path is None:
            st.error("No CSV found inside uploaded ZIP.")
            return None
        return pd.read_csv(csv_path, low_memory=False)
    else:
        # assume CSV (or Excel)
        try:
            return pd.read_csv(uploaded_file, low_memory=False)
        except Exception:
            try:
                return pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                return None

def find_col_by_keywords(df_cols, keywords):
    for col in df_cols:
        low = col.lower()
        for kw in keywords:
            if kw in low:
                return col
    return None

def auto_map_columns(df):
    cols = df.columns.tolist()
    mapping = {}
    mapping['member_id'] = find_col_by_keywords(cols, ['member', 'id', 'emp', 'employee'])
    mapping['start_date'] = find_col_by_keywords(cols, ['date','start','joined','join','hire'])
    mapping['age'] = find_col_by_keywords(cols, ['age'])
    mapping['salary'] = find_col_by_keywords(cols, ['salary','income','pay','monthly'])
    mapping['tenure'] = find_col_by_keywords(cols, ['tenure','years','service','yrs'])
    mapping['balance_access'] = find_col_by_keywords(cols, ['access','accessible','withdrawable','pot1','liquid','cash'])
    mapping['balance_retire'] = find_col_by_keywords(cols, ['retire','retirement','preserve','pot2','locked'])
    mapping['withdraw_event'] = find_col_by_keywords(cols, ['withdraw','withdrawal','attrition','left','exit','churn'])
    mapping['withdraw_date'] = find_col_by_keywords(cols, ['withdraw_date','exit_date','left_date','resign_date','date_of_withdraw'])
    return mapping

def prepare_dataset(df, mapping):
    df2 = df.copy()
    # member id
    if mapping.get('member_id') and mapping['member_id'] in df2.columns:
        df2.rename(columns={mapping['member_id']:'member_id'}, inplace=True)
    else:
        df2['member_id'] = np.arange(1, len(df2)+1)
    # start date
    if mapping.get('start_date') and mapping['start_date'] in df2.columns:
        df2.rename(columns={mapping['start_date']:'start_date'}, inplace=True)
        try:
            df2['start_date'] = pd.to_datetime(df2['start_date'])
        except Exception:
            df2['start_date'] = pd.to_datetime(df2['start_date'], errors='coerce')
    else:
        df2['start_date'] = pd.to_datetime('2018-01-01') + pd.to_timedelta(np.random.randint(0,365*4,size=len(df2)), unit='d')
    # age
    if mapping.get('age') and mapping['age'] in df2.columns:
        df2.rename(columns={mapping['age']:'age'}, inplace=True)
    else:
        df2['age'] = np.random.randint(25,55,size=len(df2))
    # salary
    if mapping.get('salary') and mapping['salary'] in df2.columns:
        df2.rename(columns={mapping['salary']:'salary'}, inplace=True)
    else:
        df2['salary'] = np.random.randint(3000,40000,size=len(df2))
    # tenure
    if mapping.get('tenure') and mapping['tenure'] in df2.columns:
        df2.rename(columns={mapping['tenure']:'tenure'}, inplace=True)
    else:
        df2['tenure'] = np.random.randint(0,30,size=len(df2))
    # balances
    if mapping.get('balance_access') and mapping['balance_access'] in df2.columns:
        df2.rename(columns={mapping['balance_access']:'balance_access'}, inplace=True)
    else:
        df2['balance_access'] = np.random.randint(0,40000,size=len(df2))
    if mapping.get('balance_retire') and mapping['balance_retire'] in df2.columns:
        df2.rename(columns={mapping['balance_retire']:'balance_retire'}, inplace=True)
    else:
        df2['balance_retire'] = np.random.randint(10000,200000,size=len(df2))
    # withdraw_event
    if mapping.get('withdraw_event') and mapping['withdraw_event'] in df2.columns:
        df2.rename(columns={mapping['withdraw_event']:'withdraw_event'}, inplace=True)
        # normalize to 0/1
        df2['withdraw_event'] = df2['withdraw_event'].apply(lambda x: 1 if str(x).strip().lower() in ['1','yes','y','true','t','withdraw','left','attrited'] else 0 if pd.notnull(x) else 0)
        df2['withdraw_event'] = pd.to_numeric(df2['withdraw_event'], errors='coerce').fillna(0).astype(int)
    else:
        # synthetic event based on simple risk
        bal = df2['balance_access'].fillna(0) + df2['balance_retire'].fillna(0)
        risk = 1/(1+np.exp((bal)/1e5 - 1)) + (10 - df2['tenure'].fillna(5))/20
        prob = (risk - risk.min())/(risk.max()-risk.min()+1e-9)
        df2['withdraw_event'] = (np.random.rand(len(df2)) < (0.25*prob)).astype(int)
    # withdraw_date/time_days
    if mapping.get('withdraw_date') and mapping['withdraw_date'] in df2.columns:
        df2.rename(columns={mapping['withdraw_date']:'withdraw_date'}, inplace=True)
        try:
            df2['withdraw_date'] = pd.to_datetime(df2['withdraw_date'])
            df2['time_days'] = (df2['withdraw_date'] - df2['start_date']).dt.days.fillna(0).astype(int)
        except Exception:
            # fallback numeric
            if df2['withdraw_date'].dtype.kind in 'iuf':
                df2['time_days'] = pd.to_numeric(df2['withdraw_date'], errors='coerce').fillna(365*5).astype(int)
            else:
                df2['time_days'] = np.where(df2['withdraw_event']==1, np.random.exponential(scale=600, size=len(df2)).astype(int), 365*5)
    else:
        max_followup_days = 365*5
        df2['time_days'] = np.where(df2['withdraw_event']==1,
                                   np.random.exponential(scale=600, size=len(df2)).astype(int),
                                   max_followup_days)
    df2['time_days'] = pd.to_numeric(df2['time_days'], errors='coerce').fillna(365*5).astype(int)
    df2['event'] = df2['withdraw_event'].astype(int)
    df2['balance_total'] = df2['balance_access'] + df2['balance_retire']
    return df2

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.title("Inputs & Data")
uploaded = st.sidebar.file_uploader("Upload CSV or ZIP (CSV inside)", type=["csv","zip","xlsx"])
use_sample = st.sidebar.checkbox("Use built-in synthetic sample instead", value=False)
run_pipeline = st.sidebar.button("Run analysis")

# sample dataset generator
def make_sample(n=2000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'member_id': np.arange(1, n+1),
        'start_date': pd.to_datetime('2019-01-01') + pd.to_timedelta(np.random.randint(0,365*4,size=n), unit='d'),
        'age': np.random.randint(20,60,size=n),
        'salary': np.random.randint(3000,50000,size=n),
        'tenure': np.random.randint(0,30,size=n),
        'balance_access': np.random.randint(0,40000,size=n),
        'balance_retire': np.random.randint(10000,200000,size=n)
    })
    # synthetic event
    bal = df['balance_access'] + df['balance_retire']
    risk = 1/(1+np.exp((bal)/1e5 - 1)) + (10 - df['tenure'])/20
    prob = (risk - risk.min())/(risk.max()-risk.min()+1e-9)
    df['withdraw_event'] = (np.random.rand(len(df)) < (0.25*prob)).astype(int)
    max_followup_days = 365*5
    df['time_days'] = np.where(df['withdraw_event']==1, np.random.exponential(scale=600,size=len(df)).astype(int), max_followup_days)
    return df

# -------------------------
# Load dataset
# -------------------------
df = None
if use_sample:
    st.sidebar.info("Using built-in synthetic sample.")
    df = make_sample(2000)
elif uploaded is not None:
    st.sidebar.info(f"Loaded {uploaded.name}")
    df = load_dataset_from_uploaded(uploaded)
else:
    st.sidebar.write("Upload a dataset or select built-in sample to begin.")

if df is None:
    st.write("No dataset loaded yet.")
    st.stop()

# show columns and preview
st.header("Dataset preview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head(250))

# automatic mapping (user can override)
st.subheader("Automatic column mapping (best-effort). Override if needed.")
mapping = auto_map_columns(df)
cols = df.columns.tolist()
cols_for_select = ["<none>"] + cols

# show mapping controls
col1, col2 = st.columns(2)
with col1:
    member_col = st.selectbox("member_id", options=cols_for_select, index=cols_for_select.index(mapping['member_id']) if mapping['member_id'] in cols_for_select else 0)
    start_col = st.selectbox("start_date", options=cols_for_select, index=cols_for_select.index(mapping['start_date']) if mapping['start_date'] in cols_for_select else 0)
    age_col = st.selectbox("age", options=cols_for_select, index=cols_for_select.index(mapping['age']) if mapping['age'] in cols_for_select else 0)
    salary_col = st.selectbox("salary", options=cols_for_select, index=cols_for_select.index(mapping['salary']) if mapping['salary'] in cols_for_select else 0)
with col2:
    tenure_col = st.selectbox("tenure", options=cols_for_select, index=cols_for_select.index(mapping['tenure']) if mapping['tenure'] in cols_for_select else 0)
    access_col = st.selectbox("balance_access", options=cols_for_select, index=cols_for_select.index(mapping['balance_access']) if mapping['balance_access'] in cols_for_select else 0)
    retire_col = st.selectbox("balance_retire", options=cols_for_select, index=cols_for_select.index(mapping['balance_retire']) if mapping['balance_retire'] in cols_for_select else 0)
    withdraw_col = st.selectbox("withdraw_event (binary)", options=cols_for_select, index=cols_for_select.index(mapping['withdraw_event']) if mapping['withdraw_event'] in cols_for_select else 0)

# allow override for withdraw_date
withdraw_date_col = st.selectbox("withdraw_date (optional)", options=cols_for_select, index=cols_for_select.index(mapping['withdraw_date']) if mapping['withdraw_date'] in cols_for_select else 0)

# build mapping dict from user selection (use None for '<none>')
user_mapping = {
    'member_id': None if member_col == "<none>" else member_col,
    'start_date': None if start_col == "<none>" else start_col,
    'age': None if age_col == "<none>" else age_col,
    'salary': None if salary_col == "<none>" else salary_col,
    'tenure': None if tenure_col == "<none>" else tenure_col,
    'balance_access': None if access_col == "<none>" else access_col,
    'balance_retire': None if retire_col == "<none>" else retire_col,
    'withdraw_event': None if withdraw_col == "<none>" else withdraw_col,
    'withdraw_date': None if withdraw_date_col == "<none>" else withdraw_date_col
}

st.write("Final mapping to be used:")
st.json(user_mapping)

# run pipeline when user clicks
if st.sidebar.button("Run pipeline with current mapping"):
    with st.spinner("Preparing dataset..."):
        df_prepped = prepare_dataset(df, user_mapping)
    st.success("Dataset prepared.")
    # EDA
    st.subheader("EDA & Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df_prepped.shape[0])
    with col2:
        st.metric("Avg age", f"{df_prepped['age'].mean():.1f}")
    with col3:
        st.metric("Avg balance total", f"{df_prepped['balance_total'].mean():.0f}")

    st.write("Value counts for event (withdraw_event):")
    st.write(df_prepped['event'].value_counts())

    # histograms
    fig, axes = plt.subplots(1,3, figsize=(14,4))
    axes[0].hist(df_prepped['age'].dropna(), bins=20)
    axes[0].set_title("Age distribution")
    axes[1].hist(df_prepped['salary'].dropna(), bins=20)
    axes[1].set_title("Salary")
    axes[2].hist(df_prepped['balance_access'].dropna(), bins=20)
    axes[2].set_title("Accessible balance")
    st.pyplot(fig)

    # Prepare classification label: withdraw within 1 year
    df_prepped['withdraw_1yr'] = ((df_prepped['time_days'] <= 365) & (df_prepped['event'] == 1)).astype(int)

    # Train/test split
    features = ['age', 'salary', 'tenure', 'balance_access', 'balance_retire']
    X = df_prepped[features].fillna(0)
    y = df_prepped['withdraw_1yr']
    if y.sum() == 0:
        st.warning("No positives detected for withdraw_1yr; synthetic labels will be created for demo.")
        # create simple synthetic label
        risk_score = 1/(1+np.exp((df_prepped['balance_access'] + df_prepped['balance_retire'])/1e5 - 1)) + (10 - df_prepped['tenure'])/20
        prob = (risk_score - risk_score.min())/(risk_score.max()-risk_score.min()+1e-9)
        y = (np.random.rand(len(df_prepped)) < (0.25*prob)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic baseline
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict_proba(X_test)[:,1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)
    st.write("Logistic Regression AUC (1yr):", f"{auc_lr:.3f}")

    # LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':31}
    model = lgb.train(params, dtrain, num_boost_round=200)
    y_pred = model.predict(X_test)
    auc_lgb = roc_auc_score(y_test, y_pred)
    st.write("LightGBM AUC (1yr):", f"{auc_lgb:.3f}")

    # classification report
    st.write("Classification report (threshold 0.5):")
    st.text(classification_report(y_test, (y_pred > 0.5).astype(int)))

    # Precision-recall curve plot
    prec, rec, thr = precision_recall_curve(y_test, y_pred)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(rec, prec)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    st.pyplot(fig_pr)

    # SHAP plots
    st.subheader("SHAP feature importance")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(6,4))
        shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.warning(f"SHAP failed: {e}")

    # Survival analysis
    st.subheader("Survival analysis")
    kmf = KaplanMeierFitter()
    T = df_prepped['time_days']
    E = df_prepped['event']
    kmf.fit(T, event_observed=E)
    fig_km, ax_km = plt.subplots()
    kmf.plot_survival_function(ax=ax_km)
    ax_km.set_xlabel("Days")
    ax_km.set_ylabel("Survival probability")
    ax_km.set_title("Kaplan-Meier: survival (no-withdrawal)")
    st.pyplot(fig_km)

    # KM by accessible balance tertiles
    st.write("KM survival by accessible-balance tertiles")
    df_prepped['access_bin'] = pd.qcut(df_prepped['balance_access'].replace(0,1), q=3, labels=['low','mid','high'])
    fig_km_group, ax2 = plt.subplots()
    for grp, grp_df in df_prepped.groupby('access_bin'):
        kmf.fit(grp_df['time_days'], grp_df['event'], label=str(grp))
        kmf.plot_survival_function(ax=ax2)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Survival probability")
    st.pyplot(fig_km_group)

    # Cox PH model
    st.write("Fitting Cox Proportional Hazards model (log balances).")
    cox_df = df_prepped[['time_days','event','age','salary','tenure','balance_access','balance_retire']].copy()
    cox_df['log_balance_access'] = np.log1p(cox_df['balance_access'])
    cox_df['log_balance_retire'] = np.log1p(cox_df['balance_retire'])
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df[['time_days','event','age','tenure','log_balance_access','log_balance_retire']],
                duration_col='time_days', event_col='event')
        st.write("Cox PH summary:")
        st.dataframe(cph.summary)
        # C-index via k-fold
        try:
            cindex = np.mean(k_fold_cross_validation(cph, cox_df, duration_col='time_days', event_col='event', k=5, scoring_method='concordance_index'))
            st.write(f"C-index (5-fold CV): {cindex:.3f}")
        except Exception as e:
            st.warning(f"C-index calculation failed: {e}")
    except Exception as e:
        st.warning(f"Cox model failed: {e}")

    # Predictions & download
    st.subheader("Predictions & Download")
    X_test['pred_prob'] = y_pred
    # Add simple survival median estimate using Cox for each test row (if cph present)
    if 'cph' in locals() and hasattr(cph, "predict_median"):
        try:
            # convert X_test for survival inputs (need log balances)
            df_for_surv = df_prepped.loc[X_test.index].copy()
            df_for_surv['log_balance_access'] = np.log1p(df_for_surv['balance_access'])
            df_for_surv['log_balance_retire'] = np.log1p(df_for_surv['balance_retire'])
            surv_medians = cph.predict_median(df_for_surv[['age','tenure','log_balance_access','log_balance_retire']])
            X_test['pred_median_survival_days'] = surv_medians.values
        except Exception as e:
            X_test['pred_median_survival_days'] = None
    else:
        X_test['pred_median_survival_days'] = None

    st.dataframe(X_test.head(20))

    # download
    csv_bytes = X_test.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="two_pot_predictions.csv", mime="text/csv")

    st.success("Pipeline finished. Inspect figures and use mapping overrides to fine-tune input columns.")

