import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal

# ==================== LOAD DATA ====================

cms = pd.read_csv("cms_result/cms_results.csv")
clinical = pd.read_csv("clinical_clean.csv")
expr = pd.read_csv("expression_clean.csv", index_col=0)

genes = ["EGFR", "ERBB2", "MYC"]

expr_gene = expr.loc[genes].T
expr_gene.index.name = "sampleID"
expr_gene.reset_index(inplace=True)

data = (
    cms.merge(clinical, on="sampleID")
       .merge(expr_gene, on="sampleID")
)

data = data[data['CMS'].notna()].copy()

cms_order = sorted(data['CMS'].unique())
data['CMS'] = pd.Categorical(data['CMS'], categories=cms_order, ordered=True)

print(data.shape)

# ==================== CREATE SUBPLOTS ====================

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

plot_idx = 0
stats = []

# 1. MSI_status
var = "MSI_status"
table = pd.crosstab(data["CMS"], data[var])
chi2, p, dof, expected = chi2_contingency(table)
stats.append([var, "Chi-square", p])
ax = axes[plot_idx]
table.plot(kind="bar", stacked=True, ax=ax)
ax.set_title(f"{var}\np={p:.3e}")
ax.set_ylabel("Samples")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 2. sex
var = "sex"
table = pd.crosstab(data["CMS"], data[var])
chi2, p, dof, expected = chi2_contingency(table)
stats.append([var, "Chi-square", p])
ax = axes[plot_idx]
table.plot(kind="bar", stacked=True, ax=ax)
ax.set_title(f"{var}\np={p:.3e}")
ax.set_ylabel("Samples")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 3. localization
var = "localization"
table = pd.crosstab(data["CMS"], data[var])
chi2, p, dof, expected = chi2_contingency(table)
stats.append([var, "Chi-square", p])
ax = axes[plot_idx]
table.plot(kind="bar", stacked=True, ax=ax)
ax.set_title(f"{var}\np={p:.3e}")
ax.set_ylabel("Samples")
ax.set_xlabel("")
ax.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 4. stage
var = "stage"
table = pd.crosstab(data["CMS"], data[var])
chi2, p, dof, expected = chi2_contingency(table)
stats.append([var, "Chi-square", p])
ax = axes[plot_idx]
table.plot(kind="bar", stacked=True, ax=ax)
ax.set_title(f"{var}\np={p:.3e}")
ax.set_ylabel("Samples")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 5. Age
ax = axes[plot_idx]
sns.boxplot(data=data, x="CMS", y="age", ax=ax)
sns.stripplot(data=data, x="CMS", y="age", color="black", alpha=0.4, ax=ax)
p = kruskal(*[g["age"].dropna() for _, g in data.groupby("CMS", observed=False)]).pvalue
stats.append(["Age", "Kruskal-Wallis", p])
ax.set_title(f"Age\np={p:.3e}")
ax.set_ylabel("Age")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 6. EGFR
gene = "EGFR"
ax = axes[plot_idx]
sns.boxplot(data=data, x="CMS", y=gene, ax=ax)
sns.stripplot(data=data, x="CMS", y=gene, color="black", alpha=0.35, ax=ax)
p = kruskal(*[g[gene].dropna() for _, g in data.groupby("CMS", observed=False)]).pvalue
stats.append([gene, "Kruskal-Wallis", p])
ax.set_title(f"{gene} expression\np={p:.3e}")
ax.set_ylabel("Expression")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 7. ERBB2
gene = "ERBB2"
ax = axes[plot_idx]
sns.boxplot(data=data, x="CMS", y=gene, ax=ax)
sns.stripplot(data=data, x="CMS", y=gene, color="black", alpha=0.35, ax=ax)
p = kruskal(*[g[gene].dropna() for _, g in data.groupby("CMS", observed=False)]).pvalue
stats.append([gene, "Kruskal-Wallis", p])
ax.set_title(f"{gene} expression\np={p:.3e}")
ax.set_ylabel("Expression")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

# 8. MYC
gene = "MYC"
ax = axes[plot_idx]
sns.boxplot(data=data, x="CMS", y=gene, ax=ax)
sns.stripplot(data=data, x="CMS", y=gene, color="black", alpha=0.35, ax=ax)
p = kruskal(*[g[gene].dropna() for _, g in data.groupby("CMS", observed=False)]).pvalue
stats.append([gene, "Kruskal-Wallis", p])
ax.set_title(f"{gene} expression\np={p:.3e}")
ax.set_ylabel("Expression")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
plot_idx += 1

if plot_idx < len(axes):
    fig.delaxes(axes[plot_idx])

fig.text(0.5, 0.02, 'CMS Subtype', ha='center', fontsize=16)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.07)

plt.savefig("figures/CMS/all_features.png", dpi=300, bbox_inches='tight')
plt.close()

# ==================== SAVE STATISTICS ====================

stats = pd.DataFrame(stats, columns=["Variable", "Test", "p_value"])
stats.to_csv("cms_result/statistics.csv", index=False)
print(stats)