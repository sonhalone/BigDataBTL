import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Banking Service AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"  # Ẩn sidebar mặc định cho gọn
)

# Custom CSS để làm đẹp giao diện
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 1rem;
        border-bottom: 2px solid #4F8BF9;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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


try:
    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
    feature_columns = load_pickle(FEATURE_COL_PATH)
    final_feature_order = load_pickle(FINAL_FEATURE_ORDER_PATH)
except Exception as e:
    st.error(f"❌ Error loading models. Please check paths. {e}")
    st.stop()


# =========================
# PREPROCESS FUNCTION (GIỮ NGUYÊN)
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
# HEADER
# =========================
st.markdown('<div class="main-header">🏦 Banking Service Prediction System</div>', unsafe_allow_html=True)

# =========================
# MAIN NAVIGATION (TABS)
# =========================
tab_csv, tab_input = st.tabs(["📂 Upload CSV & Batch Predict", "✍️ Single Customer Prediction"])

# =====================================================
# TAB 1: PREDICT BY CSV
# =====================================================
with tab_csv:
    st.markdown('<div class="sub-header">Batch Prediction from File</div>', unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        file = st.file_uploader("Step 1: Upload CSV file", type="csv")

    if file:
        df_raw = pd.read_csv(file)

        with col_preview:
            st.info(f"File uploaded successfully. Rows: {df_raw.shape[0]}")
            with st.expander("👀 View Raw Data"):
                st.dataframe(df_raw.head())

        # Button to trigger prediction
        if st.button("🚀 Run Prediction on CSV", key="btn_csv"):
            try:
                with st.spinner("Processing data and predicting..."):
                    X_input = preprocess_raw(df_raw)
                    preds = model.predict(X_input)
                    df_raw["predicted_using"] = preds

                st.success("Analysis Complete!")

                # --- Result Visualization Layout ---
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

                with res_col1:
                    st.metric("Total Customers", len(df_raw))


                with res_col2:
                    st.markdown("**Model Performance (ROC)**")
                    try:
                        st.image(ROC_IMAGE_PATH, use_container_width=True)
                    except:
                        st.warning("ROC image missing")

                with res_col3:
                    st.markdown("**Prediction Distribution**")
                    counts = (
                        df_raw["predicted_using"]
                        .value_counts()
                        .rename(index={0: "Not Using", 1: "Using"})
                    )
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=['#ff9999', '#66b3ff'])
                    st.pyplot(fig)

                # --- Data Table & Download ---
                st.markdown("### 📋 Detailed Results")
                st.dataframe(df_raw)

                csv = df_raw.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Result CSV",
                    data=csv,
                    file_name="banking_prediction_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =====================================================
# TAB 2: PREDICT BY INPUT
# =====================================================
with tab_input:
    st.markdown('<div class="sub-header">Customer Profile Input</div>', unsafe_allow_html=True)

    # Chia input thành 3 cột logic để dễ nhìn hơn
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.number_input("Age", 18, 100, 30)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
                                   'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])

    with col2:
        st.markdown("#### 💰 Financial Status")
        balance = st.number_input("Annual Balance (€)", -100000, 1000000, 0)
        default = st.selectbox("Has Credit in Default?", ["no", "yes"])
        housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

    with col3:
        st.markdown("#### 📞 Campaign Info")
        contact = st.selectbox("Contact Communication", ["cellular", "telephone", "unknown"])
        month = st.selectbox("Last Contact Month",
                             ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Duration (sec)", 0, 5000, 300)
        campaign = st.number_input("Campaign Contacts", 1, 50, 1)

        # Nhóm 2 cái ít dùng vào expander cho gọn
        with st.expander("Advanced History"):
            pdays = st.number_input("Pdays (-1 if new)", -1, 999, -1)
            previous = st.number_input("Previous Contacts", 0, 100, 0)
            poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "other", "success"])

    st.write("---")

    # Nút bấm và Kết quả căn giữa
    c_btn, c_res = st.columns([1, 2])

    with c_btn:
        predict_btn = st.button("🔮 ANALYZE CUSTOMER", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "age": age, "balance": balance, "duration": duration, "campaign": campaign,
            "pdays": pdays, "previous": previous, "default": default, "housing": housing,
            "loan": loan, "job": job, "marital": marital, "education": education,
            "contact": contact, "month": month, "poutcome": poutcome
        }])

        try:
            X = preprocess_raw(input_df)
            pred = model.predict(X)[0]

            with c_res:
                if pred == 1:
                    st.success("✅ **HIGH POTENTIAL**: Customer is likely to use the banking service.")
                    st.balloons()
                else:
                    st.error("❌ **LOW POTENTIAL**: Customer is unlikely to use the service.")
        except Exception as e:
            st.error(f"Error processing input: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>Banking Prediction System | Powered by Streamlit & Scikit-learn</div>",
    unsafe_allow_html=True)