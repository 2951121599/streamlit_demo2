import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# Set page config with custom icon
st.set_page_config(
    page_title="Stroke Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🧠 Risk prediction for Stroke")
st.markdown("""
    本工具使用miRNA表达数据来预测脑卒中风险，并通过SHAP可视化提供机制解释。
    在侧边栏调整miRNA表达水平，探索它们对疾病进展和诊断标志物的影响。
""")

# Load and prepare background data
@st.cache_data
def load_background_data():
    df = pd.read_excel('data/10feature_train.xlsx')
    return df[[
    'hsa-miR-29b-1-5p', 'hsa-miR-486-5p', 'hsa-miR-23a-3p', 'hsa-miR-296-5p', 'hsa-miR-551b-3p', 
    'hsa-miR-92a-3p', 'hsa-miR-581', 'hsa-miR-154-5p', 'hsa-miR-769-5p', 'hsa-miR-99b-3p'
]]

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('data/MODEL.h5')

# Initialize data and model
background_data = load_background_data()
model = load_model()

# Default values for miRNAs
default_values = {
    'hsa-miR-29b-1-5p': 1.000,
    'hsa-miR-486-5p': 0.000,
    'hsa-miR-23a-3p': 1.000,
    'hsa-miR-296-5p': 0.678,
    'hsa-miR-551b-3p': 1.000,
    'hsa-miR-92a-3p': 1.000,
    'hsa-miR-581': 1.000,
    'hsa-miR-154-5p': 0.781,
    'hsa-miR-769-5p': 0.777,
    'hsa-miR-99b-3p': 1.000
}

cut_off = 0.1346

# Sidebar configuration
st.sidebar.header("🧬 miRNA Expression Inputs")
st.sidebar.markdown("Adjust expression levels of stroke-related miRNAs:")

# Reset button
if st.sidebar.button("Reset to Defaults", key="reset"):
    st.session_state.update(default_values)

# Dynamic two-column layout for 10 miRNAs
mirna_features = list(default_values.keys())
mirna_values = {}
cols = st.sidebar.columns(2)

for i, mirna in enumerate(mirna_features):
    with cols[i % 2]:
        mirna_values[mirna] = st.number_input(
            mirna,
            min_value=float(background_data[mirna].min()),
            max_value=float(background_data[mirna].max()),
            value=default_values[mirna],
            step=0.001,
            format="%.3f",
            key=mirna
        )

# Prepare input data
def prepare_input_data():
    return pd.DataFrame([mirna_values])

# Main analysis
if st.button("🧠 Analyze miRNA Impacts", key="calculate"):
    input_df = prepare_input_data()
    
    # Prediction
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    st.header("📈 Diagnostic Prediction")    
    st.metric("Stroke Probability", f"{prediction:.4f}", 
             delta="Positive" if prediction >= cut_off else "Negative",
             delta_color="inverse")
    
    # SHAP explanation
    explainer = shap.DeepExplainer(model, background_data.values)
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    base_value = float(explainer.expected_value[0].numpy())

    # Visualization tabs
    tab1, tab2 = st.tabs(["Decision Plot", "Mechanistic Insights"])

    with tab1:
        st.subheader("Feature Impact Visualization")
        col1, col2 = st.columns([2, 2])  # 创建两个列，图像放在较小的列中
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))  # 设置图像大小
            shap.decision_plot(base_value, shap_values, input_df.columns, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab2:
        st.subheader("Mechanistic Insights")
        st.markdown("""
        **Key Stroke-related Pathways:**
        """)
        importance_df = pd.DataFrame({'miRNA': input_df.columns, 'SHAP Value': shap_values})
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))