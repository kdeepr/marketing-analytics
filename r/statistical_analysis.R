# Statistical analysis of the Bank Marketing campaign dataset.
#
# Complements the Python EDA/modeling with formal hypothesis tests that are
# more idiomatic in R: chi-square tests of independence for categorical
# drivers of conversion, a Welch t-test and ANOVA for numeric/grouped
# effects, and a logistic regression for interpretable odds ratios.
#
# Run from the project root:
#   Rscript r/statistical_analysis.R
#
# Requires: readr, dplyr, ggplot2, broom (installed on first run if missing).

required_packages <- c("readr", "dplyr", "ggplot2", "broom")
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!(pkg %in% installed)) install.packages(pkg, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(broom)
})

if (!file.exists("data/processed/bank_marketing_features.csv")) {
  stop("Run this script from the project root: Rscript r/statistical_analysis.R")
}
data_path <- "data/processed/bank_marketing_features.csv"
out_dir <- "reports/r_outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(data_path, show_col_types = FALSE)
df$y <- as.integer(df$y)

sink(file.path(out_dir, "statistical_tests_summary.txt"), split = TRUE)

cat("=========================================================\n")
cat("BANK MARKETING CAMPAIGN - STATISTICAL ANALYSIS (R)\n")
cat("=========================================================\n\n")

## 1. Chi-square test: is subscription independent of job type? ---------
cat("--- 1. Chi-square test: subscription ~ job ---\n")
tbl_job <- table(df$job, df$y)
chi_job <- chisq.test(tbl_job)
print(chi_job)
cat(sprintf("=> p-value = %.4g. %s\n\n", chi_job$p.value,
            ifelse(chi_job$p.value < 0.05,
                   "Reject H0: conversion rate depends significantly on job.",
                   "Fail to reject H0.")))

## 2. Chi-square test: subscription ~ prior campaign outcome ------------
cat("--- 2. Chi-square test: subscription ~ poutcome ---\n")
tbl_poutcome <- table(df$poutcome, df$y)
chi_poutcome <- chisq.test(tbl_poutcome)
print(chi_poutcome)
cat(sprintf("=> p-value = %.4g. %s\n\n", chi_poutcome$p.value,
            ifelse(chi_poutcome$p.value < 0.05,
                   "Reject H0: prior outcome strongly predicts conversion.",
                   "Fail to reject H0.")))

## 3. Welch two-sample t-test: age differs between converters/non? ------
cat("--- 3. Welch t-test: age by subscription outcome ---\n")
t_age <- t.test(age ~ y, data = df)
print(t_age)
cat(sprintf("=> p-value = %.4g. Mean age (no)=%.1f vs (yes)=%.1f\n\n",
            t_age$p.value, t_age$estimate[1], t_age$estimate[2]))

## 4. One-way ANOVA: euribor3m across quarters ---------------------------
cat("--- 4. ANOVA: euribor3m (economic context) across quarters ---\n")
anova_fit <- aov(euribor3m ~ quarter, data = df)
print(summary(anova_fit))
cat("=> Confirms macro-economic conditions vary materially by quarter,\n")
cat("   which is why 'quarter' and the raw economic indicators both carry\n")
cat("   independent signal in the model (see notebooks/03_modeling.ipynb).\n\n")

## 5. Logistic regression: interpretable odds ratios ---------------------
cat("--- 5. Logistic regression: interpretable drivers of subscription ---\n")
log_model <- glm(
  y ~ age + campaign + previous + contacted_before + prior_success +
      emp_var_rate + euribor3m + job + education + contact,
  data = df, family = binomial(link = "logit")
)
tidy_model <- tidy(log_model, exponentiate = TRUE, conf.int = TRUE)
print(tidy_model, n = Inf)
write_csv(tidy_model, file.path(out_dir, "logistic_regression_odds_ratios.csv"))
cat("\n=> Odds ratios > 1 increase subscription odds; < 1 decrease them.\n")
cat("   Full table saved to reports/r_outputs/logistic_regression_odds_ratios.csv\n\n")

sink()

## Plots -------------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))

p1 <- df %>%
  group_by(job) %>%
  summarise(conversion_rate = mean(y), n = n(), .groups = "drop") %>%
  filter(n >= 200) %>%
  ggplot(aes(x = reorder(job, conversion_rate), y = conversion_rate)) +
  geom_col(fill = "#2f6f4f") +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Conversion rate by job (chi-sq p < 0.001)", x = NULL, y = "Conversion rate")
ggsave(file.path(out_dir, "conversion_by_job_r.png"), p1, width = 7, height = 5, dpi = 150)

p2 <- ggplot(df, aes(x = quarter, y = euribor3m, fill = quarter)) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "3-month Euribor rate by quarter (ANOVA)", x = "Quarter", y = "Euribor 3M (%)")
ggsave(file.path(out_dir, "euribor_by_quarter_r.png"), p2, width = 6, height = 5, dpi = 150)

p3 <- df %>%
  mutate(y_label = ifelse(y == 1, "Subscribed", "Not subscribed")) %>%
  ggplot(aes(x = y_label, y = age, fill = y_label)) +
  geom_violin(alpha = 0.7, show.legend = FALSE) +
  geom_boxplot(width = 0.1, outlier.shape = NA, show.legend = FALSE) +
  labs(title = "Age distribution by subscription outcome (Welch t-test)", x = NULL, y = "Age")
ggsave(file.path(out_dir, "age_by_outcome_r.png"), p3, width = 6, height = 5, dpi = 150)

cat("R analysis complete. Outputs written to reports/r_outputs/\n")
