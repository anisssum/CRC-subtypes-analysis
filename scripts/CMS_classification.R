#!/usr/bin/env Rscript

library(CMScaller)

expr <- read.csv(
  "expression_clean.csv",
  row.names = 1,
  check.names = FALSE
)

png("./figures/CMS/heatmap_CMS.png",
    width = 1000,
    height = 800,
    res = 150)

res <- CMScaller(
  expr,
  RNAseq = FALSE,
  FDR = 0.05,
  rowNames = "symbol",
  seed = 42,
  doPlot = TRUE
)

dev.off()

cms_results <- data.frame(
  sampleID = colnames(expr),
  CMS = res$prediction
)

write.csv(
  cms_results,
  "./cms_result/cms_results.csv",
  row.names = FALSE
)

write.csv(
  res,
  "./cms_result/cms_full_results.csv",
  row.names = TRUE
)

# =========================== Prediction statistics ===========================

cms_stats <- as.data.frame(table(cms_results$CMS))
colnames(cms_stats) <- c("CMS", "Count")

write.csv(
  cms_stats,
  "./cms_result/cms_statistics.csv",
  row.names = FALSE
)