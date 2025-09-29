import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
import math
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr, mannwhitneyu, norm
from io import BytesIO

# --- ROC & Tanısal Ölçütler yardımcıları ---
rng = np.random.default_rng(42)

def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2)
    phat = successes / n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    return (max(0, center - half), min(1, center + half))

def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    base_auc = auc(fpr, tpr)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        aucs.append(auc(fpr_b, tpr_b))
    if len(aucs) == 0:
        return base_auc, (np.nan, np.nan)
    lo = np.quantile(aucs, alpha/2)
    hi = np.quantile(aucs, 1 - alpha/2)
    return base_auc, (lo, hi)

def youden_best_threshold(fpr, tpr, thr):
    j = tpr - fpr
    j_ix = int(np.argmax(j))
    return thr[j_ix], tpr[j_ix], 1 - fpr[j_ix]

def confusion_from_threshold(y_true_bin, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    TP = int(((y_pred==1) & (y_true_bin==1)).sum())
    TN = int(((y_pred==0) & (y_true_bin==0)).sum())
    FP = int(((y_pred==1) & (y_true_bin==0)).sum())
    FN = int(((y_pred==0) & (y_true_bin==1)).sum())
    return TP, TN, FP, FN

def diag_metrics_with_ci(y_true_bin, y_score, thr, alpha=0.05):
    TP, TN, FP, FN = confusion_from_threshold(y_true_bin, y_score, thr)
    sens = TP / (TP + FN) if (TP+FN)>0 else np.nan
    spec = TN / (TN + FP) if (TN+FP)>0 else np.nan
    ppv  = TP / (TP + FP) if (TP+FP)>0 else np.nan
    npv  = TN / (TN + FN) if (TN+FN)>0 else np.nan
    sens_ci = wilson_ci(TP, TP+FN, alpha)
    spec_ci = wilson_ci(TN, TN+FP, alpha)
    ppv_ci  = wilson_ci(TP, TP+FP, alpha) if (TP+FP)>0 else (np.nan, np.nan)
    npv_ci  = wilson_ci(TN, TN+FN, alpha) if (TN+FN)>0 else (np.nan, np.nan)
    return (sens, sens_ci), (spec, spec_ci), (ppv, ppv_ci), (npv, npv_ci)

def format_auc_with_ci(a, ci):
    return f"{a:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"

def format_rate_with_ci(x, ci):
    if np.isnan(x) or np.isnan(ci[0]) or np.isnan(ci[1]):
        return "NA"
    return f"{x*100:.0f} ({ci[0]*100:.1f}–{ci[1]*100:.1f})"

def make_diag_summary_table(result_dict_ordered_cols):
    rows = [
        ("AUC (95% CI)", "auc_ci"),
        ("p-Value", "p"),
        ("Cut-off", "cut"),
        ("Sensitivity (95% CI)", "sens"),
        ("Specificity (95% CI)", "spec"),
        ("PPV (95% CI)", "ppv"),
        ("NPV (95% CI)", "npv"),
    ]
    data = { "": [r[0] for r in rows] }
    for col, vals in result_dict_ordered_cols.items():
        data[col] = [vals[rkey] for _, rkey in rows]
    return pd.DataFrame(data)
# --- son ---

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

    # NEW: Download DPI for images
    download_dpi = st.sidebar.number_input("Download DPI", min_value=72, max_value=1200, value=300, step=10)

    st.sidebar.header("Select Analysis")
    analysis_type = st.sidebar.radio("Choose Analysis", 
                                      ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"])

    if analysis_type == "Correlation Heatmap":
        correlation_vars = st.sidebar.multiselect(
            "Select variables for Correlation Matrix (numeric)",
            options=df.columns,
            default=df.select_dtypes(include=[np.number]).columns.tolist()
        )

        # NEW: Correlation method selector
        corr_method = st.sidebar.selectbox("Correlation Method", ["Spearman", "Pearson"])

        if len(correlation_vars) < 2:
            st.warning("Select at least 2 numeric variables.")
            st.stop()

        heatmap_title = st.sidebar.text_input("Heatmap Title", value="Correlation Heatmap")
        custom_names = {}
        for col in correlation_vars:
            new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
            custom_names[col] = new_name

        footnote = st.text_area("Add footnote below the plot", value="")

        df_corr = df[correlation_vars].apply(pd.to_numeric, errors='coerce')
        df_corr = df_corr.dropna()
        df_corr.rename(columns=custom_names, inplace=True)

        # NEW: Compute correlation by selected method (using pandas .corr)
        method_key = "spearman" if corr_method.lower() == "spearman" else "pearson"
        corr_df = df_corr.corr(method=method_key)

        mask = np.triu(np.ones_like(corr_df, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_df, mask=mask, cmap=palette_choice, center=0,
            annot=True, fmt=".2f", square=True, linewidths=.5,
            cbar_kws={"shrink": .75}, ax=ax, annot_kws={"size":8}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        plt.title(heatmap_title)
        plt.xlabel("")
        plt.ylabel("")
        st.pyplot(fig)
# === ROC Özet Tablosu (Single) ===
classes = np.sort(y_true.unique())
if len(classes) != 2:
    st.error(f"ROC için ikili sonuç gerekli. Sınıflar: {classes}")
else:
    y_bin = (y_true == classes[-1]).astype(int).to_numpy()
    y_sc  = y_scores.to_numpy()

    auc_val, auc_ci = bootstrap_auc_ci(y_bin, y_sc, n_boot=2000, alpha=0.05, random_state=42)
    pos = y_sc[y_bin==1]; neg = y_sc[y_bin==0]
    if len(pos)>0 and len(neg)>0:
        pval = mannwhitneyu(pos, neg, alternative='two-sided').pvalue
        p_disp = f"{pval:.3g}" if pval >= 0.001 else "<0.001"
    else:
        p_disp = "NA"

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    best_thr, best_sens, best_spec = youden_best_threshold(fpr, tpr, thr)

    (sens, sens_ci), (spec, spec_ci), (ppv, ppv_ci), (npv, npv_ci) = diag_metrics_with_ci(y_bin, y_sc, best_thr)

    colname = custom_name  # sidebar'da verdiğiniz ad
    summary = {
        colname: {
            "auc_ci": format_auc_with_ci(auc_val, auc_ci),
            "p": p_disp,
            "cut": f"{best_thr:.3g}",
            "sens": format_rate_with_ci(sens, sens_ci),
            "spec": format_rate_with_ci(spec, spec_ci),
            "ppv":  format_rate_with_ci(ppv,  ppv_ci),
            "npv":  format_rate_with_ci(npv,  npv_ci),
        }
    }
    df_summary = make_diag_summary_table(summary)

    st.subheader("ROC curve analysis and statistical diagnostic measures")
    st.dataframe(df_summary, use_container_width=True)
    st.download_button(
        "Download summary (CSV)",
        df_summary.to_csv(index=False).encode('utf-8'),
        "roc_summary.csv",
        "text/csv"
    )

        if footnote:
            st.markdown(f"**Note:** {footnote}")

        # Use selected DPI for downloads
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=download_dpi)
        st.download_button("Download PNG", buf.getvalue(), file_name="heatmap.png", mime="image/png")
        buf.seek(0)
        fig.savefig(buf, format="jpg", bbox_inches="tight", dpi=download_dpi)
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
# === ROC Özet Tablosu (Multiple) ===
y_true_all = pd.to_numeric(df[outcome_var], errors='coerce')
classes = np.sort(y_true_all.dropna().unique())
if len(classes) != 2:
    st.error(f"ROC için ikili sonuç gerekli. Sınıflar: {classes}")
else:
    y_bin_all = (y_true_all == classes[-1]).astype(int)

    results = {}
    for var in predictor_vars:
        y_scores = pd.to_numeric(df[var], errors='coerce')
        mask = y_scores.notna() & y_true_all.notna()
        yb = y_bin_all[mask].to_numpy()
        ys = y_scores[mask].astype(float).to_numpy()
        if len(np.unique(yb)) < 2:
            continue

        auc_val, auc_ci = bootstrap_auc_ci(yb, ys, n_boot=2000, alpha=0.05, random_state=42)
        pos = ys[yb==1]; neg = ys[yb==0]
        pval = mannwhitneyu(pos, neg, alternative='two-sided').pvalue
        p_disp = f"{pval:.3g}" if pval >= 0.001 else "<0.001"

        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thr = roc_curve(yb, ys)
        best_thr, best_sens, best_spec = youden_best_threshold(fpr, tpr, thr)
        (sens, sens_ci), (spec, spec_ci), (ppv, ppv_ci), (npv, npv_ci) = diag_metrics_with_ci(yb, ys, best_thr)

        colname = custom_names.get(var, var)
        results[colname] = {
            "auc_ci": format_auc_with_ci(auc_val, auc_ci),
            "p": p_disp,
            "cut": f"{best_thr:.3g}",
            "sens": format_rate_with_ci(sens, sens_ci),
            "spec": format_rate_with_ci(spec, spec_ci),
            "ppv":  format_rate_with_ci(ppv,  ppv_ci),
            "npv":  format_rate_with_ci(npv,  npv_ci),
        }

    if len(results) == 0:
        st.warning("Geçerli değişken bulunamadı.")
    else:
        df_summary = make_diag_summary_table(results)
        st.subheader("ROC curve analysis and statistical diagnostic measures")
        st.dataframe(df_summary, use_container_width=True)
        st.download_button(
            "Download summary (CSV)",
            df_summary.to_csv(index=False).encode('utf-8'),
            "roc_multi_summary.csv",
            "text/csv"
        )
        if footnote:
            st.markdown(f"**Note:** {footnote}")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("Download PNG", buf.getvalue(), file_name="multi_roc.png", mime="image/png")
        buf.seek(0)
        fig.savefig(buf, format="jpg", bbox_inches="tight")
        st.download_button("Download JPG", buf.getvalue(), file_name="multi_roc.jpg", mime="image/jpeg")


