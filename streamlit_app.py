# app.py - Streamlit app for Two-Pot Retirement analysis
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, glob, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation

st.set_page_config(layout="wide", page_title="Two-Pot Retirement — Demo Dashboard")


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
    name = uploaded_file.name.lower()
    if name.endswith(".zip"):
        tmp_zip = "/tmp/uploaded_archive.zip"
        with open(tmp_zip, "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_path = extract_first_csv_from_zip(tmp_zip, extract_dir="/tmp/extracted_from_upload")
        if csv_path is None:
            st.error("No CSV found inside uploaded ZIP.")
            return None
        return pd.read_csv(csv_path, low_memory=False)
    else:
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
        df2['withdraw_event'] = df2['withdraw_event'].apply(lambda x: 1 if str(x).strip().lower() in ['1','yes','y','true','t','withdraw','left','attrited'] else 0 if pd.notnull(x) else 0)
        df2['withdraw_event'] = pd.to_numeric(df2['withdraw_event'], errors='coerce').fillna(0).astype(int)
    else:
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
use_sample = st.sidebar.checkbox("Use built-in synthetic sample instead", value=True)

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
    st.sidebar.warning("Upload a dataset or select built-in sample to begin.")

if df is None:
    st.write("No dataset loaded yet.")
    st.stop()

# -------------------------
# Dataset preview
# -------------------------
st.header("Dataset preview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head(250))

# -------------------------
# Automatic column mapping
# -------------------------
st.subheader("Automatic column mapping (best-effort). Override if needed.")
mapping = auto_map_columns(df)
cols = df.columns.tolist()
cols_for_select = ["<none>"] + cols

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
withdraw_date_col = st.selectbox("withdraw_date (optional)", options=cols_for_select, index=cols_for_select.index(mapping['withdraw_date']) if mapping['withdraw_date'] in cols_for_select else 0)

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

# -------------------------
# Run

