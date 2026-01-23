import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from core.classifiers import centroid_classifier, mahalanobis_classifier

st.set_page_config(layout="wide")
st.title("🧠 Classification Playground (Assignment 5)")

uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    feature_cols = st.multiselect("Select Feature Columns", df.columns[:-1], default=df.columns[:2])
    label_col = st.selectbox("Select Label Column", df.columns, index=len(df.columns)-1)

    X = df[feature_cols].values
    y = df[label_col].values

    split_ratio = st.slider("Train/Test Split (%)", 50, 90, 70)
    n_train = int(len(X) * split_ratio / 100)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    clf_type = st.selectbox("Select Classifier", ["Nearest Centroid", "Mahalanobis"])

    if st.button("🚀 Train & Evaluate"):

        if clf_type == "Nearest Centroid":
            y_pred, clf = centroid_classifier(X_train, y_train, X_test)

        else:
            y_pred = mahalanobis_classifier(X_train, y_train, X_test)

        acc = np.mean(y_pred == y_test)
        st.metric("Accuracy", f"{acc:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))
        disp.plot(ax=ax)
        st.pyplot(fig)

        # Scatter visualization
        st.subheader("📈 Feature Space")

        fig, ax = plt.subplots()
        for label in np.unique(y):
            mask = y == label
            ax.scatter(X[mask,0], X[mask,1], label=label)

        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        **Explainability**
        - Nearest Centroid: assigns sample to closest class centroid.
        - Mahalanobis: accounts for variance and feature correlation.
        - Confusion matrix shows misclassification patterns.
        """)

else:
    st.info("👆 Upload a CSV dataset to begin.")