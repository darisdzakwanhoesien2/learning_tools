import streamlit as st
import pandas as pd
from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(page_title="Keyword Visualizer", layout="wide")
st.title("📘 Keyword & Concept Visualizer")

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"

# =====================================
# DATA LOADING
# =====================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"❌ Dataset not found: {DATA_PATH}")
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)

        required_cols = {"Term", "Answer"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ dataset.csv must contain columns: {required_cols}")
            st.stop()

        # Clean whitespace
        df["Term"] = df["Term"].astype(str).str.strip()
        df["Answer"] = df["Answer"].astype(str).str.strip()

        return df

    except Exception as e:
        st.error(f"❌ Failed to load dataset.csv: {e}")
        st.stop()


df = load_data()

# =====================================
# SIDEBAR CONTROLS
# =====================================

st.sidebar.header("🔎 Search & Filters")

search_term = st.sidebar.text_input("Search term")

filtered_df = df.copy()
if search_term:
    filtered_df = filtered_df[
        filtered_df["Term"].str.contains(search_term, case=False, na=False)
    ]

if filtered_df.empty:
    st.warning("No matching terms found.")
    st.stop()

selected_term = st.sidebar.selectbox(
    "Select a term",
    filtered_df["Term"].tolist()
)

# =====================================
# MAIN LAYOUT
# =====================================

tab1, tab2, tab3 = st.tabs(["📋 Table", "📖 Definition Viewer", "📊 Word Analysis"])

# -------------------------------------
# TAB 1 — TABLE VIEW
# -------------------------------------

with tab1:
    st.subheader("Keyword Table")
    st.dataframe(filtered_df, use_container_width=True)

# -------------------------------------
# TAB 2 — DEFINITION VIEWER
# -------------------------------------

with tab2:
    st.subheader(f"📌 {selected_term}")

    row = df[df["Term"] == selected_term].iloc[0]

    st.markdown("### 📖 Definition")
    st.info(row["Answer"])

# -------------------------------------
# TAB 3 — WORD FREQUENCY ANALYSIS
# -------------------------------------

with tab3:
    st.subheader("📊 Keyword Frequency (All Definitions)")

    combined_text = " ".join(
        df["Answer"].fillna("").tolist()
    ).lower()

    words = re.findall(r"[a-z']+", combined_text)
    stopwords = {
        "the", "a", "of", "and", "to", "is", "it", "that",
        "by", "for", "from", "where", "as", "are", "an"
    }

    words = [w for w in words if w not in stopwords and len(w) > 3]
    freq = Counter(words).most_common(15)

    if freq:
        labels, counts = zip(*freq)

        fig, ax = plt.subplots()
        ax.barh(labels[::-1], counts[::-1])
        ax.set_title("Top Keywords in Definitions")
        st.pyplot(fig)
    else:
        st.warning("No keywords found.")

# =====================================
# FOOTER
# =====================================

st.caption("Dataset loaded from data/dataset.csv")
