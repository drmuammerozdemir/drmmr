import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr
from io import BytesIO


st.title('🔬 ROC AUC & Correlation Heatmap Dashboard (.csv, .txt, .sav)')

uploaded_file = st.file_uploader("Upload CSV, TXT, or SPSS (.sav)", type=["csv", "txt", "sav"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'txt':
        df = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
    elif file_extension == 'sav':
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.read())
        df, meta = pyreadstat.read_sav("temp.sav")

    st.write('Data Preview:', df.head())

    st.sidebar.header("Global Plot Options")
    palette_choice = st.sidebar.selectbox(
        "Heatmap Color Palette",
        ["coolwarm", "vlag", "rocket", "mako", "icefire"]
    )

    st.sidebar.header("Select Analysis")
    analysis_type = st.sidebar.radio("Choose Analysis", 
                                      ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"])

    if analysis_type == "Correlation Heatmap":
        correlation_vars = st.sidebar.multiselect(
            "Select variables for Correlation Matrix (numeric)",
            options=df.columns,
            default=df.select_dtypes(include=[np.number]).columns.tolist()
        )

        if len(correlation_vars) < 2:
            st.warning("Select at least 2 numeric variables.")
            st.stop()

        heatmap_title = st.sidebar.text_input("Heatmap Title", value="Spearman Correlation Heatmap")
        custom_names = {}
        for col in correlation_vars:
            new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
            custom_names[col] = new_name

        footnote = st.text_area("Add footnote below the plot", value="")

        df_corr = df[correlation_vars].apply(pd.to_numeric, errors='coerce')
        df_corr = df_corr.dropna()
        df_corr.rename(columns=custom_names, inplace=True)

        corr, _ = spearmanr(df_corr)
        columns = df_corr.columns
        corr_df = pd.DataFrame(corr, index=columns, columns=columns)

        mask = np.triu(np.ones_like(corr_df, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_df, mask=mask, cmap=palette_choice, center=0,
            annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        plt.title(heatmap_title)
        plt.xlabel("")
        plt.ylabel("")
        st.pyplot(fig)

        if footnote:
            st.markdown(f"**Note:** {footnote}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(), file_name="heatmap.png", mime="image/png")
        buf.seek(0)
        fig.savefig(buf, format="jpg", bbox_inches="tight")
        st.download_button("Download JPG", buf.getvalue(), file_name="heatmap.jpg", mime="image/jpeg")

    if analysis_type == "Single ROC Curve":
        outcome_var = st.sidebar.selectbox("Select Outcome Variable (0/1)", options=df.columns)
        predictor_var = st.sidebar.selectbox("Select Predictor Variable (numeric)", options=df.columns)

        plot_title = st.sidebar.text_input("ROC Title", "ROC Curve")
        x_label = st.sidebar.text_input("X-axis Label", "Specificity (FPR)")
        y_label = st.sidebar.text_input("Y-axis Label", "Sensitivity (TPR)")
        custom_name = st.sidebar.text_input(f"Rename '{predictor_var}'", value=predictor_var)
        footnote = st.text_area("Add footnote below the plot", value="")

        y_true = pd.to_numeric(df[outcome_var], errors='coerce')
        y_scores = pd.to_numeric(df[predictor_var], errors='coerce')
        mask = ~y_true.isna() & ~y_scores.isna()
        y_true = y_true[mask].astype(int)
        y_scores = y_scores[mask].astype(float)
        y_true = y_true.replace({2: 0, 1: 1})

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f'{custom_name} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        ax.legend(loc="lower right")
        st.pyplot(fig)

        if footnote:
            st.markdown(f"**Note:** {footnote}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(), file_name="roc.png", mime="image/png")
        buf.seek(0)
        fig.savefig(buf, format="jpg", bbox_inches="tight")
        st.download_button("Download JPG", buf.getvalue(), file_name="roc.jpg", mime="image/jpeg")

    if analysis_type == "Multiple ROC Curves":
        outcome_var = st.sidebar.selectbox("Select Outcome Variable (0/1)", options=df.columns)
        predictor_vars = st.sidebar.multiselect("Select Predictor Variables (numeric)", options=df.columns)

        plot_title = st.sidebar.text_input("ROC Title", "Multiple ROC Curves")
        x_label = st.sidebar.text_input("X-axis Label", "Specificity (FPR)")
        y_label = st.sidebar.text_input("Y-axis Label", "Sensitivity (TPR)")
        footnote = st.text_area("Add footnote below the plot", value="")

        custom_names = {}
        for col in predictor_vars:
            new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
            custom_names[col] = new_name

        fig, ax = plt.subplots(figsize=(7, 6))
        for var in predictor_vars:
            y_true = pd.to_numeric(df[outcome_var], errors='coerce')
            y_scores = pd.to_numeric(df[var], errors='coerce')
            mask = ~y_true.isna() & ~y_scores.isna()
            y_true = y_true[mask].astype(int)
            y_scores = y_scores[mask].astype(float)
            y_true = y_true.replace({2: 0, 1: 1})

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{custom_names[var]} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        ax.legend(loc="lower right")
        st.pyplot(fig)

        if footnote:
            st.markdown(f"**Note:** {footnote}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(), file_name="multi_roc.png", mime="image/png")
        buf.seek(0)
        fig.savefig(buf, format="jpg", bbox_inches="tight")
        st.download_button("Download JPG", buf.getvalue(), file_name="multi_roc.jpg", mime="image/jpeg")