import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Banking Service Prediction",
    page_icon="🔮",
    layout="wide"
)

# =========================
# PATHS
# =========================
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURE_COL_PATH = "model/feature_columns.pkl"
FINAL_FEATURE_ORDER_PATH = "model/final_feature_order.pkl"
ROC_IMAGE_PATH = "demo_ui/ROC.png"

# =========================
# LOAD OBJECTS
# =========================
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH)
feature_columns = load_pickle(FEATURE_COL_PATH)
final_feature_order = load_pickle(FINAL_FEATURE_ORDER_PATH)

# =========================
# PREPROCESS FUNCTION
# =========================
@st.cache_data
def preprocess_raw(df_raw):
    df = df_raw.copy()

    # binary mapping
    for col in ['default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # one-hot
    df = pd.get_dummies(
        df,
        columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    )

    # align before scale
    df = df.reindex(columns=feature_columns, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)

    # create *_total
    def add_total(prefix):
        cols = [c for c in df_scaled.columns if c.startswith(prefix)]
        df_scaled[f"{prefix[:-1]}_total"] = df_scaled[cols].mean(axis=1)
        return cols

    drop_cols = []
    drop_cols += add_total('job_')
    drop_cols += add_total('marital_')
    drop_cols += add_total('education_')
    drop_cols += add_total('contact_')
    drop_cols += add_total('month_')
    drop_cols += add_total('poutcome_')

    df_scaled.drop(columns=drop_cols, inplace=True)

    # enforce order
    df_scaled = df_scaled.reindex(columns=final_feature_order)

    return df_scaled

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("🔍 Prediction Mode")
mode = st.sidebar.radio(
    "",
    ["Predict by CSV", "Predict by Input"]
)

# =========================
# PIE CHART
# =========================
def draw_pie_chart(df):
    st.subheader("📊 Prediction Distribution")

    counts = (
        df["predicted_using"]
        .value_counts()
        .rename(index={0: "Not Using", 1: "Using"})
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        try:
            st.image(ROC_IMAGE_PATH, caption="ROC Curve")
        except:
            st.info("ROC image not found")

    with col2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

# =====================================================
# 1️⃣ PREDICT BY CSV
# =====================================================
if mode == "Predict by CSV":

    st.title("📂 Predict by CSV")

    file = st.file_uploader("Upload RAW banking CSV", type="csv")

    if file:
        df_raw = pd.read_csv(file)
        st.subheader("Data Preview")
        st.dataframe(df_raw.head())

        try:
            X_input = preprocess_raw(df_raw)

            # DEBUG (có thể comment sau khi chạy ổn)
            # st.write("Model expects:", final_feature_order)
            # st.write("Input columns:", X_input.columns.tolist())

            preds = model.predict(X_input)
            df_raw["predicted_using"] = preds

            st.subheader("✅ Prediction Results")
            st.dataframe(df_raw)

            draw_pie_chart(df_raw)

            # ---- download ----
            csv = df_raw.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download prediction results",
                data=csv,
                file_name="banking_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# =====================================================
# 2️⃣ PREDICT BY INPUT (GIỐNG ẢNH)
# =====================================================
else:
    st.title("✍️ Predict by Input")

    with st.sidebar:
        age = st.number_input("Age", 18, 100, 30)
        balance = st.number_input("Balance", -100000, 1000000, 0)
        duration = st.number_input("Duration", 0, 5000, 300)
        campaign = st.number_input("Campaign", 1, 50, 1)
        pdays = st.number_input("Pdays", -1, 999, -1)
        previous = st.number_input("Previous", 0, 100, 0)

        default = st.selectbox("Default", ["no", "yes"])
        housing = st.selectbox("Housing Loan", ["no", "yes"])
        loan = st.selectbox("Personal Loan", ["no", "yes"])

        job = st.selectbox(
            "Job",
            ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
             'management', 'retired', 'self-employed',
             'services', 'student', 'technician', 'unemployed', 'unknown']
        )

        marital = st.selectbox("Marital", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
        month = st.selectbox(
            "Month",
            ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        )
        poutcome = st.selectbox(
            "Previous Outcome",
            ["unknown", "failure", "other", "success"]
        )

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([{
            "age": age,
            "balance": balance,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "default": default,
            "housing": housing,
            "loan": loan,
            "job": job,
            "marital": marital,
            "education": education,
            "contact": contact,
            "month": month,
            "poutcome": poutcome
        }])

        X = preprocess_raw(input_df)
        pred = model.predict(X)[0]

        st.subheader("Result")
        if pred == 1:
            st.success("✅ Customer WILL use banking service")
        else:
            st.error("❌ Customer will NOT use banking service")
