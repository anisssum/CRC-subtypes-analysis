import os
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

cms = pd.read_csv("cms_result/cms_results.csv")

survival = pd.read_csv("survival%2FCOADREAD_survival.txt", sep = "\t")
survival.rename(columns={'sample': 'sampleID'}, inplace=True)

data = cms.merge(
    survival[["sampleID", "OS", "OS.time"]],
    on="sampleID",
    how="inner"
)

data = data.dropna()

# ===================== KAPLAN-MEIER =====================

kmf = KaplanMeierFitter()

plt.figure(figsize=(8,7))

for cms_type in sorted(data["CMS"].unique()):
    subset = data[data["CMS"] == cms_type]
    kmf.fit(
        durations=subset["OS.time"],
        event_observed=subset["OS"],
        label=cms_type
    )
    kmf.plot_survival_function(ci_show=True)

# ===================== LOG-RANK TEST =====================

results = multivariate_logrank_test(
    event_durations=data["OS.time"],
    groups=data["CMS"],
    event_observed=data["OS"]
)

p = results.p_value

plt.title(f"Log-rank p = {p:.3e}")
plt.xlabel("Days")
plt.ylabel("Overall survival probability")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "figures/CMS/Kaplan_Meier_OS.png",
    dpi=300
)

plt.close()

with open("cms_result/survival_statistics.txt", "w") as f:
    f.write(f"Log-rank p-value = {p:.6g}\n\n")
    f.write(str(results.summary))
