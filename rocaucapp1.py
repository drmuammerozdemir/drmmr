# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadstat
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr, mannwhitneyu, norm, chi2
from io import BytesIO
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="ROC AUC & Correlation Heatmap", layout="wide")
st.title('🔬 ROC AUC & Correlation Heatmap Dashboard (.csv, .txt, .sav, .xls, .xlsx)')

# =========================
# Yardımcı fonksiyonlar
# =========================
def wilson_ci(successes, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha/2)
    phat = successes / n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + z**2/(4*n))/n)) / denom
    return (max(0, center - half), min(1, center + half))

def bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
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

def confusion_from_threshold(y_true_bin, y_score, thr, greater_is_positive=True):
    if greater_is_positive:
        y_pred = (y_score >= thr).astype(int)
    else:
        y_pred = (y_score <= thr).astype(int)
    TP = int(((y_pred==1) & (y_true_bin==1)).sum())
    TN = int(((y_pred==0) & (y_true_bin==0)).sum())
    FP = int(((y_pred==1) & (y_true_bin==0)).sum())
    FN = int(((y_pred==0) & (y_true_bin==1)).sum())
    return TP, TN, FP, FN

def diag_metrics_with_ci(y_true_bin, y_score, thr, alpha=0.05, greater_is_positive=True):
    TP, TN, FP, FN = confusion_from_threshold(y_true_bin, y_score, thr, greater_is_positive)
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
    if any(map(np.isnan, ci)):
        return f"{a:.3f} (NA–NA)"
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
        ("LR+", "lr_pos"),
        ("LR-", "lr_neg"),
        ("DOR", "dor"),
    ]
    data = { "": [r[0] for r in rows] }
    for col, vals in result_dict_ordered_cols.items():
        data[col] = [vals[rkey] for _, rkey in rows]
    return pd.DataFrame(data)

# DeLong testi yardımcıları
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = i + (j - i + 1) / 2.
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _compute_auc(y_true, y_pred):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _compute_midrank(y_pred)
    R_pos = ranks[y_true == 1]
    auc_val = (np.sum(R_pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc_val

def _compute_structural_components(y_true, y_pred):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    pos_scores = y_pred[y_true == 1]
    neg_scores = y_pred[y_true == 0]
    v_pos = np.array([np.mean(neg_scores < s) for s in pos_scores])
    v_neg = np.array([np.mean(pos_scores > s) for s in neg_scores])
    return v_pos, v_neg

def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    auc1 = _compute_auc(y_true, y_pred1)
    auc2 = _compute_auc(y_true, y_pred2)
    v_pos1, v_neg1 = _compute_structural_components(y_true, y_pred1)
    v_pos2, v_neg2 = _compute_structural_components(y_true, y_pred2)
    s_pos = np.var(v_pos1 - v_pos2, ddof=1)
    s_neg = np.var(v_neg1 - v_neg2, ddof=1)
    var = (s_pos / n_pos) + (s_neg / n_neg)
    if var == 0:
        z = np.inf * np.sign(auc1 - auc2)
    else:
        z = (auc1 - auc2) / np.sqrt(var)
    p_value = 2. * norm.sf(np.abs(z))
    return z, p_value

# =========================
# Dosya yükleme
# =========================
uploaded_file = st.file_uploader(
    "Upload CSV, TXT, SPSS (.sav), or Excel (.xls, .xlsx)",
    type=["csv", "txt", "sav", "xls", "xlsx"]
)

df = None
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in ('csv', 'txt'):
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-9')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
    elif file_extension in ('xls', 'xlsx'):
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception:
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, engine='xlrd')
            except Exception as e:
                st.error(f"Excel dosyası okunamadı (Hata: {e}).")
    elif file_extension == 'sav':
        with open("temp.sav", "wb") as f:
            f.write(uploaded_file.read())
        df, meta = pyreadstat.read_sav("temp.sav")

if df is not None:
    st.write('**Data Preview:**')
    st.dataframe(df.head(), width='stretch')

# =========================
# Sidebar
# =========================
st.sidebar.header("Global Plot Options")
palette_choice = st.sidebar.selectbox(
    "Heatmap Color Palette",
    ["coolwarm", "vlag", "rocket", "mako", "icefire"]
)
download_dpi = st.sidebar.number_input(
    "Download DPI",
    min_value=72, max_value=1200, value=300, step=10
)
st.sidebar.header("Select Analysis")
analysis_type = st.sidebar.radio(
    "Choose Analysis",
    ["Correlation Heatmap", "Single ROC Curve", "Multiple ROC Curves"]
)

# =========================
# Correlation Heatmap (GÜNCELLENMİŞ BÖLÜM)
# =========================
if df is not None and analysis_type == "Correlation Heatmap":
    correlation_vars = st.sidebar.multiselect(
        "Select variables for Correlation Matrix (numeric)",
        options=df.columns.tolist(),
        default=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    )

    if len(correlation_vars) < 2:
        st.warning("Select at least 2 variables.")
        st.stop()

    corr_method = st.sidebar.selectbox("Correlation Method", ["Spearman", "Pearson"], index=0)
    method_key = "spearman" if corr_method.lower() == "spearman" else "pearson"
    heatmap_title = st.sidebar.text_input("Heatmap Title", value=f"{corr_method} Correlation Heatmap")

    custom_names = {}
    for col in correlation_vars:
        new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
        custom_names[col] = new_name

    footnote = st.text_area("Add footnote below the plot", value="")

    # Veriyi hazırla
    num_data = df[correlation_vars].apply(pd.to_numeric, errors='coerce').dropna()
    
    # 1. Isı Haritası Hesaplamaları
    corr_df = num_data.corr(method=method_key)
    corr_df.rename(columns=custom_names, index=custom_names, inplace=True)
    mask_tri = np.triu(np.ones_like(corr_df, dtype=bool))

    # Görselleştirme
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df, mask=mask_tri, cmap=palette_choice, center=0,
        annot=True, fmt=".2f", square=False, linewidths=.5,
        cbar_kws={"shrink": .75}, ax=ax, annot_kws={"size":9}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(heatmap_title, pad=15)
    st.pyplot(fig)

    # 2. İSTATİSTİKSEL DETAY TABLOSU (RHO VE P-DEĞERLERİ)
    st.subheader(f"{corr_method} Correlation Details (Rho & p-values)")
    
    # Tüm çiftler için hesaplama yap
    results_list = []
    cols = correlation_vars
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v1, v2 = cols[i], cols[j]
            if method_key == "spearman":
                rho, p = spearmanr(num_data[v1], num_data[v2])
            else:
                from scipy.stats import pearsonr
                rho, p = pearsonr(num_data[v1], num_data[v2])
            
            # Anlamlılık yıldızlarını belirle
            stars = ""
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
            
            results_list.append({
                "Variable 1": custom_names[v1],
                "Variable 2": custom_names[v2],
                "Coefficient (r/ρ)": f"{rho:.3f}",
                "p-value": f"{p:.4g}",
                "Significance": stars
            })
    
    detay_df = pd.DataFrame(results_list)
    st.dataframe(detay_df, use_container_width=True)

    # CSV İndirme Butonu
    st.download_button(
        label="Download Correlation Table (CSV)",
        data=detay_df.to_csv(index=False).encode('utf-8'),
        file_name="correlation_details.csv",
        mime="text/csv"
    )

    if footnote:
        st.markdown(f"**Note:** {footnote}")
# =========================
# Single ROC
# =========================
if df is not None and analysis_type == "Single ROC Curve":
    outcome_var = st.sidebar.selectbox("Select Outcome Variable (0/1)", options=df.columns)
    predictor_var = st.sidebar.selectbox("Select Predictor Variable (numeric)", options=df.columns)

    plot_title = st.sidebar.text_input("ROC Title", "ROC Curve")
    x_label = st.sidebar.text_input("X-axis Label", "False Positive Rate (1 − Specificity)")
    y_label = st.sidebar.text_input("Y-axis Label", "True Positive Rate (Sensitivity)")
    custom_name = st.sidebar.text_input(f"Rename '{predictor_var}'", value=predictor_var)
    score_dir = st.sidebar.radio("Score direction", ["Auto-detect (recommended)", "Higher values indicate disease (+)", "Lower values indicate disease (−)"])

    footnote = st.text_area("Add footnote below the plot", value="")

    y_true_raw = pd.to_numeric(df[outcome_var], errors='coerce')
    y_scores_raw = pd.to_numeric(df[predictor_var], errors='coerce')
    valid_mask = y_true_raw.notna() & y_scores_raw.notna()
    y_true = y_true_raw[valid_mask].astype(float)
    y_scores = y_scores_raw[valid_mask].astype(float)

    classes = np.sort(y_true.unique())
    if len(classes) != 2:
        st.error(f"ROC için ikili sonuç gerekli. Bulunan sınıflar: {classes}")
        st.stop()

    pos_label = classes[-1]
    y_bin = (y_true == pos_label).astype(int).to_numpy()
    st.caption(f"Pozitif sınıf = {pos_label} (otomatik).")

    ys_arr = y_scores.to_numpy()

    # YÖN BELİRLEME
    if score_dir.startswith("Auto"):
        fpr_test, tpr_test, _ = roc_curve(y_bin, ys_arr)
        auc_test = auc(fpr_test, tpr_test)
        higher_is_positive = (auc_test >= 0.5)
    else:
        higher_is_positive = score_dir.startswith("Higher")

    ys_for_roc = ys_arr if higher_is_positive else -ys_arr

    fpr, tpr, thr_tmp = roc_curve(y_bin, ys_for_roc)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    if len(fpr) > 3:
        fpr_smooth = np.linspace(0, 1, 200)
        interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
        tpr_smooth = np.maximum.accumulate(interp_func(fpr_smooth))
        ax.plot(fpr_smooth, tpr_smooth, lw=2, label=f'{custom_name} (AUC = {roc_auc:.3f})')
    else:
        ax.plot(fpr, tpr, lw=2, label=f'{custom_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    if footnote:
        st.markdown(f"**Note:** {footnote}")

    auc_val, auc_ci = bootstrap_auc_ci(y_bin, ys_for_roc, n_boot=1000, alpha=0.05, random_state=42)
    pos_scores = ys_for_roc[y_bin==1]
    neg_scores = ys_for_roc[y_bin==0]
    pval = mannwhitneyu(pos_scores, neg_scores, alternative='two-sided').pvalue
    p_disp = f"{pval:.3g}" if pval >= 0.001 else "<0.001"

    best_thr_internal, _, _ = youden_best_threshold(fpr, tpr, thr_tmp)
    thr_display = best_thr_internal if higher_is_positive else -best_thr_internal
    (sens, sens_ci), (spec, spec_ci), (ppv, ppv_ci), (npv, npv_ci) = diag_metrics_with_ci(
        y_bin, ys_arr, thr_display, greater_is_positive=higher_is_positive
    )

    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else np.nan
    lr_neg = (1 - sens) / spec if spec > 0 else np.nan
    dor = lr_pos / lr_neg if (not np.isnan(lr_neg) and lr_neg > 0 and not np.isnan(lr_pos)) else np.nan

    cut_rule = "≥" if higher_is_positive else "≤"
    summary = {
        custom_name: {
            "auc_ci": format_auc_with_ci(auc_val, auc_ci),
            "p": p_disp,
            "cut": f"{cut_rule} {thr_display:.3g}",
            "sens": format_rate_with_ci(sens, sens_ci),
            "spec": format_rate_with_ci(spec, spec_ci),
            "ppv":  format_rate_with_ci(ppv,  ppv_ci),
            "npv":  format_rate_with_ci(npv,  npv_ci),
            "lr_pos": f"{lr_pos:.2f}" if not np.isnan(lr_pos) else "NA",
            "lr_neg": f"{lr_neg:.2f}" if not np.isnan(lr_neg) else "NA",
            "dor": f"{dor:.2f}" if not np.isnan(dor) else "NA",
        }
    }
    df_summary = make_diag_summary_table(summary)

    st.subheader("ROC curve analysis and statistical diagnostic measures")
    st.dataframe(df_summary, width='stretch')
    st.download_button(
        "Download summary (CSV)",
        df_summary.to_csv(index=False).encode('utf-8'),
        "roc_summary.csv",
        "text/csv"
    )

    for ext, mime in [('png','image/png'), ('jpg','image/jpeg')]:
        buf = BytesIO()
        fig.savefig(buf, format=ext, bbox_inches="tight", dpi=download_dpi)
        st.download_button(f"Download {ext.upper()}", buf.getvalue(),
                            file_name=f"roc.{ext}", mime=mime)

# =========================
# Multiple ROC
# =========================
if df is not None and analysis_type == "Multiple ROC Curves":
    outcome_var = st.sidebar.selectbox("Select Outcome Variable (0/1)", options=df.columns)
    predictor_vars = st.sidebar.multiselect("Select Predictor Variables (numeric)", options=df.columns)
    plot_title = st.sidebar.text_input("ROC Title", "Multiple ROC Curves")
    x_label = st.sidebar.text_input("X-axis Label", "False Positive Rate (1 − Specificity)")
    y_label = st.sidebar.text_input("Y-axis Label", "True Positive Rate (Sensitivity)")
    footnote = st.text_area("Add footnote below the plot", value="")

    score_dir_multi = st.sidebar.radio(
        "Score direction (applies to all predictors)",
        ["Auto-detect per predictor (recommended)",
         "Higher values indicate disease (+)",
         "Lower values indicate disease (−)"]
    )

    custom_names = {}
    for col in predictor_vars:
        new_name = st.sidebar.text_input(f"Rename '{col}'", value=col)
        custom_names[col] = new_name

    if len(predictor_vars) == 0:
        st.warning("En az bir kestirici seçin.")
        st.stop()

    y_true_all = pd.to_numeric(df[outcome_var], errors='coerce')
    classes = np.sort(y_true_all.dropna().unique())
    if len(classes) != 2:
        st.error(f"ROC için ikili sonuç gerekli. Sınıflar: {classes}")
        st.stop()
    pos_label = classes[-1]
    y_bin_all = (y_true_all == pos_label).astype(int)
    st.caption(f"Pozitif sınıf = {pos_label} (otomatik).")

    fig, ax = plt.subplots(figsize=(7, 6))
    results = {}
    delong_data_store = {}  # var -> (yb, ys_for_roc)

    for var in predictor_vars:
        y_scores_raw = pd.to_numeric(df[var], errors='coerce')
        valid_mask = y_scores_raw.notna() & y_true_all.notna()
        yb = y_bin_all[valid_mask].to_numpy()
        ys = y_scores_raw[valid_mask].astype(float).to_numpy()

        if len(np.unique(yb)) < 2:
            st.warning(f"'{var}' için iki sınıf bulunamadı, atlandı.")
            continue

        # YÖN BELİRLEME
        fpr_h, tpr_h, thr_h = roc_curve(yb, ys)
        auc_h = auc(fpr_h, tpr_h)

        if score_dir_multi.startswith("Auto"):
            current_higher_is_positive = (auc_h >= 0.5)
        elif score_dir_multi.startswith("Higher"):
            current_higher_is_positive = True
        else:
            current_higher_is_positive = False

        if current_higher_is_positive:
            fpr, tpr, thr_tmp = fpr_h, tpr_h, thr_h
            my_auc = auc_h
            ys_for_roc = ys
            cut_rule = "≥"
        else:
            ys_for_roc = -ys
            fpr, tpr, thr_tmp = roc_curve(yb, ys_for_roc)
            my_auc = auc(fpr, tpr)
            cut_rule = "≤"
        if len(fpr) > 3:
            fpr_smooth = np.linspace(0, 1, 200)
            interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(0, 1))
            tpr_smooth = np.maximum.accumulate(interp_func(fpr_smooth))
            ax.plot(fpr_smooth, tpr_smooth, lw=2, label=f"{custom_names.get(var,var)} (AUC = {my_auc:.3f})")
        else:
            ax.plot(fpr, tpr, lw=2, label=f"{custom_names.get(var,var)} (AUC = {my_auc:.3f})")

        # METRİKLER
        best_thr_internal, _, _ = youden_best_threshold(fpr, tpr, thr_tmp)
        thr_display = best_thr_internal if current_higher_is_positive else -best_thr_internal

        (sens, sens_ci), (spec, spec_ci), (ppv, ppv_ci), (npv, npv_ci) = diag_metrics_with_ci(
            yb, ys, thr_display, greater_is_positive=current_higher_is_positive
        )

        pos_s = ys_for_roc[yb == 1]
        neg_s = ys_for_roc[yb == 0]
        try:
            p_val = mannwhitneyu(pos_s, neg_s, alternative='two-sided').pvalue
            p_disp = f"{p_val:.3g}" if p_val >= 0.001 else "<0.001"
        except Exception:
            p_disp = "NA"

        _, auc_ci = bootstrap_auc_ci(yb, ys_for_roc, n_boot=1000, alpha=0.05, random_state=42)

        lr_pos = sens / (1 - spec) if (1 - spec) > 0 else np.nan
        lr_neg = (1 - sens) / spec if spec > 0 else np.nan
        dor = lr_pos / lr_neg if (not np.isnan(lr_neg) and lr_neg > 0 and not np.isnan(lr_pos)) else np.nan

        colname = custom_names.get(var, var)
        results[colname] = {
            "auc_ci": format_auc_with_ci(my_auc, auc_ci),
            "p": p_disp,
            "cut": f"{cut_rule} {thr_display:.3g}",
            "sens": format_rate_with_ci(sens, sens_ci),
            "spec": format_rate_with_ci(spec, spec_ci),
            "ppv":  format_rate_with_ci(ppv,  ppv_ci),
            "npv":  format_rate_with_ci(npv,  npv_ci),
            "lr_pos": f"{lr_pos:.2f}" if not np.isnan(lr_pos) else "NA",
            "lr_neg": f"{lr_neg:.2f}" if not np.isnan(lr_neg) else "NA",
            "dor": f"{dor:.2f}" if not np.isnan(dor) else "NA",
        }

        # DeLong için sakla
        delong_data_store[var] = current_higher_is_positive

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    if footnote:
        st.markdown(f"**Note:** {footnote}")

    if len(results) > 0:
        df_summary = make_diag_summary_table(results)
        st.subheader("ROC curve analysis and statistical diagnostic measures")
        st.dataframe(df_summary, width='stretch')
        st.download_button(
            "Download summary (CSV)",
            df_summary.to_csv(index=False).encode('utf-8'),
            "roc_multi_summary.csv",
            "text/csv"
        )

    for ext, mime in [('png','image/png'), ('jpg','image/jpeg')]:
        buf = BytesIO()
        fig.savefig(buf, format=ext, bbox_inches="tight", dpi=download_dpi)
        st.download_button(f"Download {ext.upper()}", buf.getvalue(),
                            file_name=f"multi_roc.{ext}", mime=mime)

    # =========================
    # DeLong Test
    # =========================
    if len(delong_data_store) >= 2:
        st.subheader("DeLong Test for AUC Comparison (Paired Data)")
        st.info(
            "**DİKKAT:** İki AUC değerini istatistiksel olarak karşılaştırmak (DeLong testi) "
            "için, her iki testin de **aynı denek grubu** üzerinde çalışılmış olması gerekir. "
            "Aşağıdaki p-değerleri, yalnızca **tüm seçili belirteçler için tam veriye sahip olan** "
            "denekler (listwise deletion) kullanılarak hesaplanmıştır."
        )

        paired_cols = [outcome_var] + list(delong_data_store.keys())
        paired_data = df[paired_cols].apply(pd.to_numeric, errors='coerce').dropna()

        if len(paired_data) < 5:
            st.warning("DeLong testi için yeterli eşleştirilmiş veri yok.")
        else:
            y_true_paired = (paired_data[outcome_var] == pos_label).astype(int).to_numpy()

            predictor_custom_names_map = {custom_names.get(var, var): var for var in delong_data_store.keys()}
            options = list(predictor_custom_names_map.keys())

            ref_name_selected = st.selectbox(
                "DeLong Testi için Referans (Ref) Belirteci Seçin:",
                options=options,
                index=0
            )

            ref_var = predictor_custom_names_map[ref_name_selected]
            ref_dir = delong_data_store[ref_var]
            ref_raw = paired_data[ref_var].to_numpy()
            ref_scores_roc = ref_raw if ref_dir else -ref_raw

            delong_results = []
            for comp_var in delong_data_store.keys():
                if comp_var == ref_var:
                    continue
                comp_name = custom_names.get(comp_var, comp_var)
                comp_dir = delong_data_store[comp_var]
                comp_raw = paired_data[comp_var].to_numpy()
                comp_scores_roc = comp_raw if comp_dir else -comp_raw

                try:
                    z_stat, p_value = delong_roc_test(y_true_paired, ref_scores_roc, comp_scores_roc)
                    p_display = f"{p_value:.4g}" if p_value >= 0.0001 else "<0.0001"
                    delong_results.append({
                        "Comparison": f"{ref_name_selected} (Ref) vs {comp_name}",
                        "Z": f"{z_stat:.3f}",
                        "p-value": p_display,
                        "N (paired)": len(y_true_paired)
                    })
                except Exception as e:
                    delong_results.append({
                        "Comparison": f"{ref_name_selected} (Ref) vs {comp_name}",
                        "Z": "NA",
                        "p-value": f"Hata: {e}",
                        "N (paired)": len(y_true_paired)
                    })

            if delong_results:
                st.dataframe(pd.DataFrame(delong_results), width='stretch')

# =========================
# Yüklenmemiş dosya durumu
# =========================
if df is None:
    st.info("Başlamak için sol üstten bir dosya yükleyin (.csv, .txt, .sav, .xls, .xlsx).")
