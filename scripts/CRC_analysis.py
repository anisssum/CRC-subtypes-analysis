import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

os.makedirs("figures", exist_ok=True)

def get_top_variable_genes(expr_df, n_top=5000):
    gene_var = expr_df.var(axis=1)
    top_genes = gene_var.nlargest(n_top).index
    expr_top = expr_df.loc[top_genes]
    X = expr_top.T
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index

SAMPLE_TYPE_CODES = {
    "01": "Primary Tumor",
    "02": "Recurrent Tumor",
    "03": "Primary Blood Cancer",
    "05": "Additional New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal"
}

# ========================== Load expression data ==========================
expr = pd.read_csv("TCGA.COADREAD.sampleMap_HiSeqV2", sep="\t", index_col=0)
print(f"Initial expression data shape: {expr.shape}")

# ========================== Gene statistics ==========================
expr_stats = pd.DataFrame({
    'mean': expr.mean(axis=1),
    'std': expr.std(axis=1),
    'min': expr.min(axis=1),
    'max': expr.max(axis=1)
})

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(expr_stats['std'], bins=50, edgecolor='black')
plt.xlabel('Standard deviation')
plt.ylabel('Number of genes')
plt.title('Gene variance distribution')

plt.subplot(1, 3, 2)
plt.boxplot([expr.iloc[i].values for i in range(min(50, expr.shape[0]))],
            flierprops={'markersize': 2})
plt.xlabel('First 50 genes')
plt.xticks([])
plt.ylabel('Expression values')
plt.title('Scale of different genes')

plt.subplot(1, 3, 3)
plt.scatter(expr_stats['mean'], expr_stats['std'], alpha=0.5)
plt.xlabel('Mean value')
plt.ylabel('Standard deviation')
plt.title('Mean vs Std')

plt.tight_layout()
plt.savefig("figures/Expr_statistics.png", dpi=300)
plt.close()

# ========================== Sample statistics ==========================
sample_mean = expr.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].boxplot([expr.iloc[:, i] for i in range(200)],
                flierprops={'markersize': 1, 'alpha': 0.3})
axes[0].set_title(f'Distribution across 200/{expr.shape[1]} samples')
axes[0].set_xlabel('Samples')
axes[0].set_ylabel('log2(expression)')
axes[0].set_xticks([])

axes[1].hist(sample_mean, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(sample_mean.mean(), color='red', linestyle='--',
                label=f"Mean = {sample_mean.mean():.2f}")
axes[1].set_xlabel('Sample mean expression')
axes[1].set_ylabel('Number of samples')
axes[1].set_title('Distribution of sample means')
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/Sample_statistics.png", dpi=300)
plt.close()

# ==================== Formal outlier criterion (sample level) ====================

q1, q3 = sample_mean.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_low, iqr_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

z_scores = (sample_mean - sample_mean.mean()) / sample_mean.std()

iqr_outliers = sample_mean[(sample_mean < iqr_low) | (sample_mean > iqr_high)]
z_outliers = sample_mean[z_scores.abs() > 3]

consensus_outliers = sorted(set(iqr_outliers.index) & set(z_outliers.index))

print(f"IQR-rule outlier candidates ({len(iqr_outliers)}): {list(iqr_outliers.index)}")
print(f"Z-score (>3 SD) outlier candidates ({len(z_outliers)}): {list(z_outliers.index)}")

expr = expr.drop(columns=consensus_outliers)
print(f"Removed {len(consensus_outliers)} sample(s) failing both outlier "
        f"criteria. Expression shape after removal: {expr.shape}")
# ========================== Technical batch check ==========================

def parse_tcga_barcode(sample_id):
    parts = str(sample_id).split("-")
    tss_code = parts[1]
    type_code = re.sub(r"[^0-9]", "", parts[3])[:2]
    sample_type = SAMPLE_TYPE_CODES.get(type_code, f"Other ({type_code})")
    return tss_code, sample_type

barcode_info = pd.DataFrame(
    [parse_tcga_barcode(s) for s in expr.columns],
    columns=["TSS", "sample_type"],
    index=expr.columns
)

# ========================== Filter top variable genes + PCA/tSNE/UMAP ==========================
X_scaled, sample_order = get_top_variable_genes(expr, n_top=5000)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)
X_tsne = tsne.fit_transform(X_scaled)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], s=20, alpha=0.7)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], s=20, alpha=0.7)
axes[1].set_xlabel("t-SNE1")
axes[1].set_ylabel("t-SNE2")
axes[1].set_title("t-SNE")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(X_umap[:, 0], X_umap[:, 1], s=20, alpha=0.7)
axes[2].set_xlabel("UMAP1")
axes[2].set_ylabel("UMAP2")
axes[2].set_title("UMAP")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/Dimensionality_Reduction_Comparison.png", dpi=300)
plt.close()

# ==================== PCA/UMAP colored by TSS and by parsed sample type ====================
barcode_plot = barcode_info.loc[sample_order].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

tss_categories = barcode_plot["TSS"].unique()
cmap_tss = plt.get_cmap("tab20", len(tss_categories))
for i, tss in enumerate(tss_categories):
    mask = barcode_plot["TSS"] == tss
    axes[0].scatter(X_umap[mask.values, 0], X_umap[mask.values, 1],
                     s=15, alpha=0.7, color=cmap_tss(i))
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].set_title(f"UMAP colored by TSS, {len(tss_categories)} sites")
axes[0].grid(True, alpha=0.3)

type_categories = barcode_plot["sample_type"].unique()
cmap_type = plt.get_cmap("tab10", len(type_categories))
for i, st in enumerate(type_categories):
    mask = barcode_plot["sample_type"] == st
    axes[1].scatter(X_umap[mask.values, 0], X_umap[mask.values, 1],
                     s=15, alpha=0.7, color=cmap_type(i), label=st)
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].set_title("UMAP colored by sample type")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/UMAP_batch_and_sample_type.png", dpi=300, bbox_inches="tight")
plt.close()

umap_values = X_umap[:, 0]
umap_negative_mask = umap_values < 0
umap_positive_mask = umap_values >= 0

n_negative = umap_negative_mask.sum()
n_positive = umap_positive_mask.sum()

print(f"PC1 split: {n_negative} samples with < 0, {n_positive} samples with >= 0")

tss_negative = barcode_plot.loc[umap_negative_mask, "TSS"].value_counts()
tss_positive = barcode_plot.loc[umap_positive_mask, "TSS"].value_counts()

all_tss = sorted(set(tss_negative.index) | set(tss_positive.index))
contingency_table = []
for tss in all_tss:
    neg_count = tss_negative.get(tss, 0)
    pos_count = tss_positive.get(tss, 0)
    contingency_table.append([neg_count, pos_count])

contingency_table = np.array(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test of independence between PC1 sign and TSS: "
        f"chi2={chi2:.2f}, df={dof}, p={p:.3e}")

# ========================== Keep only Primary Tumor ==========================

primary_tumor_mask = barcode_info["sample_type"] == "Primary Tumor"
print(f"Samples by parsed type before filtering:\n{barcode_info['sample_type'].value_counts()}")

expr = expr.loc[:, primary_tumor_mask]
print(f"Expression data after keeping only Primary Tumor: {expr.shape}")

# ========================== Load clinical data ==========================
clinical_df = pd.read_csv("TCGA.COADREAD.sampleMap%2FCOADREAD_clinicalMatrix", sep="\t")
print(f"Initial clinical data shape: {clinical_df.shape}")

sample_ids = expr.columns.tolist()

clinical_df = clinical_df[
    clinical_df["sampleID"].isin(sample_ids)
].copy()

# ========================== Select relevant clinical columns ==========================
clinical_df = clinical_df[
    [
        "sampleID",
        "_primary_disease",
        "CDE_ID_3226963",
        "pathologic_stage",
        "age_at_initial_pathologic_diagnosis",
        "gender",
        "anatomic_neoplasm_subdivision",
    ]
]

clinical_df = clinical_df.rename(
    columns={
        "_primary_disease": "disease",
        "CDE_ID_3226963": "MSI_status",
        "pathologic_stage": "stage",
        "age_at_initial_pathologic_diagnosis": "age",
        "gender": "sex",
        "anatomic_neoplasm_subdivision": "localization",
    }
)

# ========================== Filter by MSI status ==========================
valid_status = ["MSS", "MSI-L", "MSI-H"]

clinical_df = clinical_df[
    clinical_df["MSI_status"].isin(valid_status)
].copy()

valid_samples = clinical_df["sampleID"].tolist()

expr = expr[valid_samples]
print(f"Expression data after filtering MSI status: {expr.shape}")

# ========================== Map stage values ==========================
stage_map = {
    "Stage I": "I",
    "Stage IA": "I",

    "Stage II": "II",
    "Stage IIA": "II",
    "Stage IIB": "II",
    "Stage IIC": "II",

    "Stage III": "III",
    "Stage IIIA": "III",
    "Stage IIIB": "III",
    "Stage IIIC": "III",

    "Stage IV": "IV",
    "Stage IVA": "IV",
    "Stage IVB": "IV",
}

clinical_df["stage"] = clinical_df["stage"].map(stage_map)

clinical_df = clinical_df.dropna(subset=["stage"])

expr = expr[clinical_df["sampleID"].tolist()]
print(f"Expression data after stage filtering: {expr.shape}")

clinical_df.to_csv("clinical_clean.csv", index=False)
expr.to_csv("expression_clean.csv")

# ========================== Descriptive statistics ==========================

report = []

report.append(clinical_df["disease"].value_counts().to_string())
report.append("")

report.append(clinical_df["sex"].value_counts().to_string())
report.append("")

report.append(clinical_df["age"].describe().to_string())
report.append("")

report.append(clinical_df["localization"].value_counts().to_string())
report.append("")

report.append(clinical_df["MSI_status"].value_counts().to_string())
report.append("")

report.append(clinical_df["stage"].value_counts().sort_index().to_string())
report.append("")

with open("statistics_summary.txt", "w") as f:
    f.write("\n".join(report))

# ==================== Overall clinical plots ====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

clinical_df["disease"].value_counts().plot(kind="bar", ax=axes[0, 0])
axes[0, 0].set_ylabel("Number of samples")
axes[0, 0].set_title("Diagnosis")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0, ha="right")
axes[0, 0].set_xlabel("")

clinical_df["sex"].value_counts().plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_ylabel("Number of patients")
axes[0, 1].set_title("Sex")
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
axes[0, 1].set_xlabel("") 

axes[0, 2].hist(clinical_df["age"], bins=20, edgecolor="black")
axes[0, 2].set_xlabel("")
axes[0, 2].set_ylabel("Number of patients")
axes[0, 2].set_title("Age distribution")

clinical_df["localization"].value_counts().plot(kind="bar", ax=axes[1, 0])
axes[1, 0].set_ylabel("Number of patients")
axes[1, 0].set_title("Tumor localization")
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
axes[1, 0].set_xlabel("")

clinical_df["MSI_status"].value_counts().plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_ylabel("Number of patients")
axes[1, 1].set_title("MSI status")
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
axes[1, 1].set_xlabel("")

clinical_df["stage"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 2])
axes[1, 2].set_ylabel("Number of patients")
axes[1, 2].set_title("Tumor stage")
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)
axes[1, 2].set_xlabel("")

plt.tight_layout()
plt.savefig("figures/All_Statistics_Combined.png", dpi=300)
plt.close()

# ========================== PCA colored by localization ==========================
X_scaled_loc, sample_order_loc = get_top_variable_genes(expr, n_top=5000)

pca = PCA(n_components=2, random_state=42)
X_pca_loc = pca.fit_transform(X_scaled_loc)

localization_by_sample = clinical_df.set_index("sampleID").loc[sample_order_loc, "localization"]

plot_df = pd.DataFrame({
    "PC1": X_pca_loc[:, 0],
    "PC2": X_pca_loc[:, 1],
    "localization": localization_by_sample.values
})

plt.figure(figsize=(8, 6))

for loc in plot_df["localization"].unique():
    mask = plot_df["localization"] == loc
    plt.scatter(
        plot_df.loc[mask, "PC1"],
        plot_df.loc[mask, "PC2"],
        label=loc,
        s=20
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA (top-5000 HVG, scaled) colored by tumor localization")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("figures/PCA_localization.png", dpi=300)
plt.close()