import streamlit as st
import json
import pandas as pd
from deid_pipeline import load_ner_model, deidentify_json

# Streamlit Config
st.set_page_config(
    page_title="🩺 Clinical Deidentification Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp, .block-container {
        background-color: #16181C !important;
        color: #e2e8f0; !important
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    textarea {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-family: 'Fira Code', monospace;
        font-size: 0.9rem;
        border: 1px solid #334155;
    }
    .stFileUploader {
        background-color: #1e293b !important;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stJson {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 8px;
        border-radius: 6px;
    }
    .stDataFrame {
        background: #1e293b;
        border-radius: 6px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
        margin: auto;
        max-width: 95% !important;  /* Expand main app container */
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    div[data-testid="column"] {
        width: 50% !important;
        flex: 1 1 50% !important;
        padding: 0 1rem !important;
    }
    .stJson {
        max-height: 600px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>🩺 Transformer-Based Clinical Deidentification with Regex-Based Safeguards</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Redact Protected Health Information (PHI) from Synthea-style FHIR JSON using a fine-tuned transformer model. Compliant with HIPAA's Safe Harbor Policy</div>", unsafe_allow_html=True)

#Load model (cached)
@st.cache_resource
def get_model():
    return load_ner_model()

ner = get_model()

# Input JSON Section
st.markdown("### 📋 Paste or Upload FHIR Input")

user_input = st.text_area(
    "Input JSON",
    placeholder='{\n  "type": "collection",\n  "entry": [ ... ]\n}',
    height=280,
    label_visibility="collapsed"
)

uploaded = st.file_uploader("Upload FHIR file", type=["json"])
if uploaded:
    user_input = uploaded.read().decode("utf-8")

# Run Button
run_btn = st.button("🔍 Run Deidentification")

# Deidentification Execution
if run_btn:
    if not user_input.strip():
        st.warning("Please paste or upload a valid JSON file before running.")
        st.stop()
    try:
        data = json.loads(user_input)
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format. Please check your input.")
        st.stop()

    with st.spinner("Running deidentification... ⏳"):
        redacted_json, entity_table = deidentify_json(data, ner)

    st.success("✅ Deidentification complete!")

    # 🧾 Output JSON + Table
    st.markdown("---")
    st.markdown("### 🔍 Results")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🩹 Redacted JSON")
        st.json(redacted_json)

    with col2:
        st.markdown("#### 📊 Detected PHI Entities")
        if entity_table:
            df = pd.DataFrame(entity_table)
            if not df.empty:
                label_colors = {
                    "NAME": "#f4a261",
                    "LOCATION": "#2a9d8f",
                    "CONTACT": "#e9c46a",
                    "ID": "#e76f51",
                    "DATE": "#8ecae6",
                    "WEB": "#9b5de5",
                }

                def color_label(label):
                    color = label_colors.get(label, "#94a3b8")
                    return f"background-color: {color}; color: #0f172a; border-radius: 5px; padding: 3px 6px"

                def color_source(source):
                    if source == "model":
                        return "background-color: #093545; color: #b6f0e0; border-radius: 5px; padding: 3px 6px"
                    elif source == "keypath":
                        return "background-color: #4b2e83; color: #f0e6ff; border-radius: 5px; padding: 3px 6px"
                    elif source == "regex":
                        return "background-color: #3b0b0b; color: #ffd6d6; border-radius: 5px; padding: 3px 6px"
                    return ""

                if "label" in df.columns and "source" in df.columns:
                    df_styled = df.style.applymap(
                        lambda v: color_label(v) if v in label_colors else "",
                        subset=["label"]
                    ).applymap(
                        lambda v: color_source(v) if v in ["model", "keypath", "regex"] else "",
                        subset=["source"]
                    )
                else:
                    df_styled = df
                st.dataframe(df_styled, use_container_width=True, height=450)
            else:
                st.info("No PHI entities detected.")
        else:
            st.info("No PHI entities detected.")
else:
    st.info("Paste or upload a JSON file, then click **Run Deidentification** to begin.")