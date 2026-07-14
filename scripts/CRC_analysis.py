import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

os.makedirs("figures", exist_ok=True)

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
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].boxplot([expr.iloc[:, i] for i in range(200)],
                flierprops={'markersize': 1, 'alpha': 0.3})
axes[0].set_title(f'Distribution across 200/{expr.shape[1]} samples')
axes[0].set_xlabel('Samples')
axes[0].set_ylabel('log2(expression)')
axes[0].set_xticks([])

axes[1].hist(expr.mean(axis=0), bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(expr.mean(axis=0).mean(), color='red', linestyle='--',
                label=f"Mean = {expr.mean(axis=0).mean():.2f}")
axes[1].set_xlabel('Sample mean expression')
axes[1].set_ylabel('Number of samples')
axes[1].set_title('Distribution of sample means')
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/Sample_statistics.png", dpi=300)
plt.close()

# ========================== Filter top variable genes ==========================
gene_var = expr.var(axis=1)
top_genes = gene_var.nlargest(5000).index
expr_filtered = expr.loc[top_genes]

# ========================== Standardization ==========================
X = expr_filtered.T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================== PCA ==========================
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

# 1. PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], s=20, alpha=0.7)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA")
axes[0].grid(True, alpha=0.3)

# 2. t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], s=20, alpha=0.7)
axes[1].set_xlabel("t-SNE1")
axes[1].set_ylabel("t-SNE2")
axes[1].set_title("t-SNE")
axes[1].grid(True, alpha=0.3)

# 3. UMAP
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], s=20, alpha=0.7)
axes[2].set_xlabel("UMAP1")
axes[2].set_ylabel("UMAP2")
axes[2].set_title("UMAP")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/Dimensionality_Reduction_Comparison.png", dpi=300)
plt.close()

# ========================== Load clinical data ==========================
clinical_df = pd.read_csv("TCGA.COADREAD.sampleMap%2FCOADREAD_clinicalMatrix", sep="\t")
print(f"Initial clinical data shape: {clinical_df.shape}")

sample_ids = expr.columns.tolist()

clinical_df = clinical_df[
    clinical_df["sampleID"].isin(sample_ids)
].copy()

# ========================== UMAP colored by sample_type ==========================
sample_order = X.index.tolist()

plot_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "sampleID": sample_order
})

plot_df = plot_df.merge(clinical_df[['sampleID', 'sample_type']], on='sampleID', how='left')

plt.figure(figsize=(10, 8))

plot_df['sample_type'] = plot_df['sample_type'].fillna("Unknown").astype(str)

categories = plot_df['sample_type'].unique()
cmap = plt.get_cmap("tab10", len(categories))

for i, cat in enumerate(categories):
    mask = plot_df['sample_type'] == cat
    plt.scatter(
        plot_df.loc[mask, "UMAP1"],
        plot_df.loc[mask, "UMAP2"],
        color=cmap(i),
        label=cat
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

plt.legend(
    title="Sample Type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig("figures/UMAP_by_sample_type.png", dpi=300, bbox_inches="tight")
plt.close()

# ========================== Leave only Primary Tumor ==========================
clinical_df = clinical_df[clinical_df["sample_type"] == "Primary Tumor"].copy()

expr = expr[clinical_df["sampleID"].tolist()]
print(f"Expression data after removing normal tissue: {expr.shape}")

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

# ==================== Единый график со всеми подграфиками ====================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Diagnosis
clinical_df["disease"].value_counts().plot(kind="bar", ax=axes[0, 0])
axes[0, 0].set_ylabel("Number of samples")
axes[0, 0].set_title("Diagnosis")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0, ha="right")
axes[0, 0].set_xlabel("")

# Sex
clinical_df["sex"].value_counts().plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_ylabel("Number of patients")
axes[0, 1].set_title("Sex")
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
axes[0, 1].set_xlabel("") 

# Age
axes[0, 2].hist(clinical_df["age"], bins=20, edgecolor="black")
axes[0, 2].set_xlabel("")
axes[0, 2].set_ylabel("Number of patients")
axes[0, 2].set_title("Age distribution")

# Localization
clinical_df["localization"].value_counts().plot(kind="bar", ax=axes[1, 0])
axes[1, 0].set_ylabel("Number of patients")
axes[1, 0].set_title("Tumor localization")
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
axes[1, 0].set_xlabel("")

# MSI status
clinical_df["MSI_status"].value_counts().plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_ylabel("Number of patients")
axes[1, 1].set_title("MSI status")
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha="right")
axes[1, 1].set_xlabel("")

# 6. Stage
clinical_df["stage"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 2])
axes[1, 2].set_ylabel("Number of patients")
axes[1, 2].set_title("Tumor stage")
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)
axes[1, 2].set_xlabel("")

plt.tight_layout()
plt.savefig("figures/All_Statistics_Combined.png", dpi=300)
plt.close()

# ========================== PCA colored by localization ==========================
X = expr.T

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "localization": clinical_df["localization"]
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
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig("figures/PCA_localization.png", dpi=300)
plt.close()