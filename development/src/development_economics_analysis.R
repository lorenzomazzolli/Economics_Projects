# =========================================================
# DEVELOPMENT ECONOMICS PROJECT
# FULL SCRIPT - MODEL 1 + MODEL 2 + WORLD SAMPLE
# INPUT: Development/data/processed/csv
# OUTPUT: Development/outputs/charts
#         Development/outputs/results_summary/results_summary.txt
# =========================================================

# -------------------------
# STEP 0 — PATHS
# -------------------------

project_dir <- "/Users/lorenzomazzolli/Desktop/GITHUB/Economics_Projects/development"
data_dir <- file.path(project_dir, "data", "processed", "csv")
output_dir <- file.path(project_dir, "outputs")
charts_dir <- file.path(output_dir, "charts")
results_dir <- file.path(output_dir, "results_summary")
results_txt <- file.path(results_dir, "results_summary.txt")

charts_model1 <- file.path(charts_dir, "model1_level")
charts_model2 <- file.path(charts_dir, "model2_growth")
charts_world  <- file.path(charts_dir, "world_sample")

dir.create(charts_model1, showWarnings = FALSE, recursive = TRUE)
dir.create(charts_model2, showWarnings = FALSE, recursive = TRUE)
dir.create(charts_world,  showWarnings = FALSE, recursive = TRUE)
dir.create(results_dir,   showWarnings = FALSE, recursive = TRUE)

# clean old png outputs to avoid duplicates
old_png_m1 <- list.files(charts_model1, pattern = "\\.png$", full.names = TRUE)
old_png_m2 <- list.files(charts_model2, pattern = "\\.png$", full.names = TRUE)
old_png_ws <- list.files(charts_world,  pattern = "\\.png$", full.names = TRUE)

if (length(old_png_m1) > 0) file.remove(old_png_m1)
if (length(old_png_m2) > 0) file.remove(old_png_m2)
if (length(old_png_ws) > 0) file.remove(old_png_ws)

# Reset txt output
cat("DEVELOPMENT ECONOMICS PROJECT - RESULTS SUMMARY\n",
    "=============================================\n\n",
    file = results_txt)

# -------------------------
# HELPER FUNCTIONS
# -------------------------

append_txt <- function(..., file = results_txt) {
  cat(..., "\n", file = file, append = TRUE, sep = "")
}

append_capture <- function(object, title = NULL, file = results_txt) {
  if (!is.null(title)) {
    cat("\n", title, "\n", file = file, append = TRUE, sep = "")
    cat(paste(rep("-", nchar(title)), collapse = ""), "\n",
        file = file, append = TRUE, sep = "")
  }
  captured <- capture.output(print(object))
  cat(paste(captured, collapse = "\n"), "\n\n", file = file, append = TRUE)
}

save_scatter <- function(data, xvar, yvar, labels_var,
                         color_vector, color_legend_labels, color_title,
                         xlab_short, ylab_short, title_long,
                         filename, folder,
                         fit_type = c("linear", "quadratic"),
                         fit_threshold = 0.5) {
  
  fit_type <- match.arg(fit_type)
  
  x_all <- data[[xvar]]
  y_all <- data[[yvar]]
  lab_all <- data[[labels_var]]
  
  ok <- complete.cases(x_all, y_all, lab_all)
  x <- x_all[ok]
  y <- y_all[ok]
  labels_vec <- lab_all[ok]
  col_vec <- color_vector[ok]
  
  this_cor <- suppressWarnings(cor(x, y, use = "complete.obs"))
  show_fit <- !is.na(this_cor) && abs(this_cor) >= fit_threshold
  
  png(filename = file.path(folder, filename),
      width = 2700, height = 1800, res = 240)
  
  par(mar = c(6, 6, 8, 10), xpd = NA)
  
  main_wrapped <- paste(strwrap(title_long, width = 78), collapse = "\n")
  
  plot(x, y,
       xlab = xlab_short,
       ylab = ylab_short,
       main = main_wrapped,
       pch = 19,
       col = col_vec,
       cex = 1.2,
       cex.axis = 1.2,
       cex.lab = 1.4,
       cex.main = 1.4)
  
  text(x, y,
       labels = labels_vec,
       pos = 4,
       offset = 0.30,
       cex = 0.65,
       col = col_vec)
  
  if (show_fit) {
    x_seq <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = 100)
    
    if (fit_type == "linear") {
      model <- lm(y ~ x)
      y_hat <- predict(model, newdata = data.frame(x = x_seq))
    }
    
    if (fit_type == "quadratic") {
      model <- lm(y ~ x + I(x^2))
      y_hat <- predict(model, newdata = data.frame(x = x_seq))
    }
    
    keep <- y_hat >= min(y, na.rm = TRUE) & y_hat <= max(y, na.rm = TRUE)
    
    lines(x_seq[keep], y_hat[keep],
          lwd = 2.5,
          lty = 1)
  }
  
  if (length(color_legend_labels) > 0) {
    legend(x = par("usr")[2] + 0.03 * diff(par("usr")[1:2]),
           y = par("usr")[4],
           legend = color_legend_labels,
           col = seq_along(color_legend_labels),
           pch = 19,
           title = color_title,
           bty = "n",
           cex = 0.95,
           xjust = 0,
           yjust = 1)
  }
  
  dev.off()
}

# -------------------------
# STEP 1 — LOAD DATASETS
# -------------------------

df <- read.csv2(file.path(data_dir, "Model_1_Regression_Table.csv"),
                stringsAsFactors = FALSE)

df_growth <- read.csv2(file.path(data_dir, "Model_2_Regression_Table.csv"),
                       stringsAsFactors = FALSE)

df_world <- read.csv2(file.path(data_dir, "Model_1_Regression_Table_World.csv"),
                      stringsAsFactors = FALSE)

df_z <- read.csv2(file.path(data_dir, "Model_1_Z_Score_Regression_Table.csv"),
                  stringsAsFactors = FALSE)

append_txt("STEP 1 — LOAD DATASETS")
append_txt("----------------------")
append_txt("Files found in data_dir:")
append_capture(list.files(data_dir), "List of files in data/processed")

append_txt("Main sample rows (Model 1): ", nrow(df))
append_txt("Growth sample rows (Model 2): ", nrow(df_growth))
append_txt("World sample rows: ", nrow(df_world))
append_txt("Z-score sample rows: ", nrow(df_z), "\n")

# -------------------------
# STEP 2 — INSPECT MAIN SAMPLE
# -------------------------

append_txt("STEP 2 — INSPECT MAIN SAMPLE")
append_txt("----------------------------")
append_capture(names(df), "Model 1 - Original Column Names")
append_capture(str(df), "Model 1 - Structure")
append_capture(summary(df), "Model 1 - Summary Statistics")
append_capture(colSums(is.na(df)), "Model 1 - Missing Values by Column")
append_txt("Model 1 dimensions: ", nrow(df), " rows x ", ncol(df), " columns\n")

# -------------------------
# STEP 3 — RENAME VARIABLES
# -------------------------

names(df) <- c(
  "region",
  "core_growth_model",
  "structural_qualifier",
  "institution_stability",
  "country_name",
  "country_code",
  "log_gdp_pc_2023",
  "governance_2023",
  "hci_plus_2025",
  "gcf_2023",
  "trade_2023",
  "resource_rents_2021",
  "fdi_2023",
  "manufacturing_va_2023",
  "services_va_2023"
)

names(df_growth) <- c(
  "region",
  "core_growth_model",
  "structural_qualifier",
  "institution_stability",
  "country_name",
  "country_code",
  "gdp_pc_growth",
  "log_gdp_pc_2000",
  "governance_avg",
  "hci_plus_avg",
  "gcf_avg",
  "trade_avg",
  "resource_rents_avg",
  "fdi_avg"
)

names(df_world) <- c(
  "region",
  "cluster",
  "country_name",
  "country_code",
  "log_gdp_pc_2023",
  "governance_2023",
  "hci_plus_2025",
  "gcf_2023",
  "trade_2023",
  "resource_rents_2021",
  "manufacturing_va_2023",
  "services_va_2023"
)

df_world$cluster <- as.character(df_world$cluster)
df_world$region  <- as.character(df_world$region)

df_world$cluster[df_world$cluster == "Central America & Caribbean"] <- "C. Am. & Carib."
df_world$cluster[df_world$cluster == "Central Asia & Caucasus"] <- "C. Asia & Cauc."

df_world$region[df_world$region == "Central America & Caribbean"] <- "C. Am. & Carib."
df_world$region[df_world$region == "Central Asia & Caucasus"] <- "C. Asia & Cauc."

names(df_z) <- c(
  "region",
  "core_growth_model",
  "structural_qualifier",
  "institution_stability",
  "country_name",
  "country_code",
  "log_gdp_pc_2023",
  "governance_z",
  "hci_plus_z",
  "gcf_z",
  "trade_z",
  "resource_rents_z",
  "fdi_z",
  "manufacturing_va_z",
  "services_va_z"
)

append_txt("STEP 3 — RENAMED VARIABLES")
append_txt("--------------------------")
append_capture(names(df), "Model 1 - Renamed Columns")
append_capture(names(df_growth), "Model 2 - Renamed Columns")
append_capture(names(df_world), "World Sample - Renamed Columns")
append_capture(names(df_z), "Z-score Sample - Renamed Columns")

# -------------------------
# STEP 4 — PREPARE NUMERIC SUBSETS
# -------------------------

num_df <- df[, c(
  "log_gdp_pc_2023",
  "governance_2023",
  "hci_plus_2025",
  "gcf_2023",
  "trade_2023",
  "resource_rents_2021",
  "fdi_2023",
  "manufacturing_va_2023",
  "services_va_2023"
)]

num_df_growth <- df_growth[, c(
  "gdp_pc_growth",
  "log_gdp_pc_2000",
  "governance_avg",
  "hci_plus_avg",
  "gcf_avg",
  "trade_avg",
  "resource_rents_avg",
  "fdi_avg"
)]

num_df_world <- df_world[, c(
  "log_gdp_pc_2023",
  "governance_2023",
  "hci_plus_2025",
  "gcf_2023",
  "trade_2023",
  "resource_rents_2021",
  "manufacturing_va_2023",
  "services_va_2023"
)]

df_world_rents10 <- subset(df_world, resource_rents_2021 >= 10)

append_txt("STEP 4 — NUMERIC SUBSETS")
append_txt("------------------------")
append_capture(summary(num_df), "Model 1 - Summary Statistics (Numeric)")
append_capture(summary(num_df_growth), "Model 2 - Summary Statistics (Numeric)")
append_capture(summary(num_df_world), "World Sample - Summary Statistics (Numeric)")

# -------------------------
# STEP 5 — CLUSTERING VECTORS
# -------------------------

growth_colors <- as.numeric(as.factor(df$core_growth_model))
region_colors <- as.numeric(as.factor(df$region))
stability_colors <- as.numeric(as.factor(df$institution_stability))

growth_colors_m2 <- as.numeric(as.factor(df_growth$core_growth_model))
stability_colors_m2 <- as.numeric(as.factor(df_growth$institution_stability))
region_colors_m2 <- as.numeric(as.factor(df_growth$region))

df_world$cluster <- as.character(df_world$cluster)

df_world$cluster[df_world$cluster == "Central America & Caribbean"] <- "C. Am. & Carib."
df_world$cluster[df_world$cluster == "Central Asia & Caucasus"] <- "C. Asia & Cauc."

cluster_colors_world <- as.numeric(as.factor(df_world$cluster))

df_world_rents10 <- subset(df_world, resource_rents_2021 >= 10)
cluster_colors_world_rents10 <- as.numeric(as.factor(df_world_rents10$cluster))

append_txt("STEP 5 — CLUSTERING VECTORS")
append_txt("---------------------------")
append_capture(table(df$core_growth_model), "Model 1 - Growth Model Distribution")
append_capture(table(df$region), "Model 1 - Region Distribution")
append_capture(table(df$institution_stability), "Model 1 - Institutional Stability Distribution")

# -------------------------
# STEP 6 — CORRELATION MATRICES
# -------------------------

corr_model1 <- round(cor(num_df, use = "complete.obs"), 3)
corr_model2 <- round(cor(num_df_growth, use = "complete.obs"), 3)
corr_world <- round(cor(num_df_world, use = "complete.obs"), 3)

append_txt("STEP 6 — CORRELATION MATRICES")
append_txt("-----------------------------")
append_capture(corr_model1, "Model 1 - Full Correlation Matrix")
append_capture(corr_model2, "Model 2 - Full Correlation Matrix")
append_capture(corr_world, "World Sample - Full Correlation Matrix")

append_txt("Model 1 - Selected Pairwise Correlations")
append_txt("---------------------------------------")

selected_corrs <- c(
  paste("log_gdp_pc_2023 vs governance_2023 =",
        round(cor(df$log_gdp_pc_2023, df$governance_2023, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs hci_plus_2025 =",
        round(cor(df$log_gdp_pc_2023, df$hci_plus_2025, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs gcf_2023 =",
        round(cor(df$log_gdp_pc_2023, df$gcf_2023, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs trade_2023 =",
        round(cor(df$log_gdp_pc_2023, df$trade_2023, use = "complete.obs"), 3)),
  paste("governance_2023 vs hci_plus_2025 =",
        round(cor(df$governance_2023, df$hci_plus_2025, use = "complete.obs"), 3)),
  paste("governance_2023 vs gcf_2023 =",
        round(cor(df$governance_2023, df$gcf_2023, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs manufacturing_va_2023 =",
        round(cor(df$log_gdp_pc_2023, df$manufacturing_va_2023, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs services_va_2023 =",
        round(cor(df$log_gdp_pc_2023, df$services_va_2023, use = "complete.obs"), 3)),
  paste("manufacturing_va_2023 vs trade_2023 =",
        round(cor(df$manufacturing_va_2023, df$trade_2023, use = "complete.obs"), 3)),
  paste("services_va_2023 vs trade_2023 =",
        round(cor(df$services_va_2023, df$trade_2023, use = "complete.obs"), 3)),
  paste("services_va_2023 vs hci_plus_2025 =",
        round(cor(df$services_va_2023, df$hci_plus_2025, use = "complete.obs"), 3)),
  paste("log_gdp_pc_2023 vs resource_rents_2021 =",
        round(cor(df$log_gdp_pc_2023, df$resource_rents_2021, use = "complete.obs"), 3)),
  paste("governance_2023 vs resource_rents_2021 =",
        round(cor(df$governance_2023, df$resource_rents_2021, use = "complete.obs"), 3)),
  paste("manufacturing_va_2023 vs resource_rents_2021 =",
        round(cor(df$manufacturing_va_2023, df$resource_rents_2021, use = "complete.obs"), 3))
)

append_txt(paste(selected_corrs, collapse = "\n"), "\n")

# -------------------------
# STEP 7 — MODEL 1 SCATTER PLOTS
# -------------------------

append_txt("STEP 7 — SCATTER PLOT DIAGNOSTICS (MODEL 1)")
append_txt("------------------------------------------")

# Block A — Institution-Based Relationships
save_scatter(
  data = df,
  xvar = "governance_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "Governance",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Governance score (-2.5 to 2.5, 2023) — clustered by Institutional Stability",
  filename = "model1_gdp_vs_governance_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "hci_plus_2025",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "HCI+",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs HCI+ overall score (0-325, 2025) — clustered by Institutional Stability",
  filename = "model1_gdp_vs_hci_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "gcf_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "GCF",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Gross capital formation (% of GDP, 2023) — clustered by Institutional Stability",
  filename = "model1_gdp_vs_gcf_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "trade_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "Trade",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Trade (% of GDP, 2023) — clustered by Institutional Stability",
  filename = "model1_gdp_vs_trade_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "hci_plus_2025",
  yvar = "governance_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "HCI+",
  ylab_short = "Governance",
  title_long = "Governance score (-2.5 to 2.5, 2023) vs HCI+ overall score (0-325, 2025) — clustered by Institutional Stability",
  filename = "model1_governance_vs_hci_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "gcf_2023",
  yvar = "governance_2023",
  labels_var = "country_code",
  color_vector = stability_colors,
  color_legend_labels = levels(as.factor(df$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "GCF",
  ylab_short = "Governance",
  title_long = "Governance score (-2.5 to 2.5, 2023) vs Gross capital formation (% of GDP, 2023) — clustered by Institutional Stability",
  filename = "model1_governance_vs_gcf_stability.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

# Block B — Structure of the Economy
save_scatter(
  data = df,
  xvar = "manufacturing_va_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = growth_colors,
  color_legend_labels = levels(as.factor(df$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "Manufacturing",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Manufacturing value added (% of GDP, 2023) — clustered by Core Growth Model",
  filename = "model1_gdp_vs_manufacturing_growthmodel.png",
  folder = charts_model1,
  fit_type = "quadratic",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "services_va_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = growth_colors,
  color_legend_labels = levels(as.factor(df$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "Services",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Services value added (% of GDP, 2023) — clustered by Core Growth Model",
  filename = "model1_gdp_vs_services_growthmodel.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "trade_2023",
  yvar = "manufacturing_va_2023",
  labels_var = "country_code",
  color_vector = growth_colors,
  color_legend_labels = levels(as.factor(df$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "Trade",
  ylab_short = "Manufacturing",
  title_long = "Manufacturing value added (% of GDP, 2023) vs Trade (% of GDP, 2023) — clustered by Core Growth Model",
  filename = "model1_manufacturing_vs_trade_growthmodel.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "trade_2023",
  yvar = "services_va_2023",
  labels_var = "country_code",
  color_vector = growth_colors,
  color_legend_labels = levels(as.factor(df$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "Trade",
  ylab_short = "Services",
  title_long = "Services value added (% of GDP, 2023) vs Trade (% of GDP, 2023) — clustered by Core Growth Model",
  filename = "model1_services_vs_trade_growthmodel.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "hci_plus_2025",
  yvar = "services_va_2023",
  labels_var = "country_code",
  color_vector = growth_colors,
  color_legend_labels = levels(as.factor(df$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "HCI+",
  ylab_short = "Services",
  title_long = "Services value added (% of GDP, 2023) vs HCI+ overall score (0-325, 2025) — clustered by Core Growth Model",
  filename = "model1_services_vs_hci_growthmodel.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

# Block C — Resource Curse
save_scatter(
  data = df,
  xvar = "resource_rents_2021",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = region_colors,
  color_legend_labels = levels(as.factor(df$region)),
  color_title = "Region",
  xlab_short = "Natural resource rents",
  ylab_short = "Log GDP per capita",
  title_long = "Log GDP per capita (constant 2021 international $, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by Region",
  filename = "model1_gdp_vs_resource_rents_region.png",
  folder = charts_model1,
  fit_type = "quadratic",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "resource_rents_2021",
  yvar = "governance_2023",
  labels_var = "country_code",
  color_vector = region_colors,
  color_legend_labels = levels(as.factor(df$region)),
  color_title = "Region",
  xlab_short = "Natural resource rents",
  ylab_short = "Governance",
  title_long = "Governance score (-2.5 to 2.5, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by Region",
  filename = "model1_governance_vs_resource_rents_region.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df,
  xvar = "resource_rents_2021",
  yvar = "manufacturing_va_2023",
  labels_var = "country_code",
  color_vector = region_colors,
  color_legend_labels = levels(as.factor(df$region)),
  color_title = "Region",
  xlab_short = "Natural resource rents",
  ylab_short = "Manufacturing",
  title_long = "Manufacturing value added (% of GDP, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by Region",
  filename = "model1_manufacturing_vs_resource_rents_region.png",
  folder = charts_model1,
  fit_type = "linear",
  fit_threshold = 0.5
)

append_txt("Model 1 scatter plots saved in: ", charts_model1, "\n")

# -------------------------
# STEP 8 — WORLD SAMPLE SCATTER PLOTS
# -------------------------

append_txt("STEP 8 — SCATTER PLOT DIAGNOSTICS (WORLD SAMPLE)")
append_txt("-----------------------------------------------")

save_scatter(
  data = df_world,
  xvar = "governance_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Governance",
  ylab_short = "Log GDP per capita",
  title_long = "World sample: Log GDP per capita (constant 2021 international $, 2023) vs Governance score (-2.5 to 2.5, 2023) — clustered by World Cluster",
  filename = "world_gdp_vs_governance_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "hci_plus_2025",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "HCI+",
  ylab_short = "Log GDP per capita",
  title_long = "World sample: Log GDP per capita (constant 2021 international $, 2023) vs HCI+ overall score (0-325, 2025) — clustered by World Cluster",
  filename = "world_gdp_vs_hci_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "governance_2023",
  yvar = "hci_plus_2025",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Governance",
  ylab_short = "HCI+",
  title_long = "World sample: HCI+ overall score (0-325, 2025) vs Governance score (-2.5 to 2.5, 2023) — clustered by World Cluster",
  filename = "world_hci_vs_governance_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "services_va_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Services",
  ylab_short = "Log GDP per capita",
  title_long = "World sample: Log GDP per capita (constant 2021 international $, 2023) vs Services value added (% of GDP, 2023) — clustered by World Cluster",
  filename = "world_gdp_vs_services_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "trade_2023",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Trade",
  ylab_short = "Log GDP per capita",
  title_long = "World sample: Log GDP per capita (constant 2021 international $, 2023) vs Trade (% of GDP, 2023) — clustered by World Cluster",
  filename = "world_gdp_vs_trade_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "trade_2023",
  yvar = "services_va_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Trade",
  ylab_short = "Services",
  title_long = "World sample: Services value added (% of GDP, 2023) vs Trade (% of GDP, 2023) — clustered by World Cluster",
  filename = "world_services_vs_trade_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world,
  xvar = "trade_2023",
  yvar = "manufacturing_va_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world,
  color_legend_labels = levels(as.factor(df_world$cluster)),
  color_title = "World Cluster",
  xlab_short = "Trade",
  ylab_short = "Manufacturing",
  title_long = "World sample: Manufacturing value added (% of GDP, 2023) vs Trade (% of GDP, 2023) — clustered by World Cluster",
  filename = "world_manufacturing_vs_trade_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world_rents10,
  xvar = "resource_rents_2021",
  yvar = "log_gdp_pc_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world_rents10,
  color_legend_labels = levels(as.factor(df_world_rents10$cluster)),
  color_title = "World Cluster",
  xlab_short = "Natural resource rents (>=10%)",
  ylab_short = "Log GDP per capita",
  title_long = "World sample filtered at natural resource rents >= 10% of GDP: Log GDP per capita vs Total natural resource rents (% of GDP, 2021) — clustered by World Cluster",
  filename = "world_rents10_gdp_vs_resource_rents_cluster.png",
  folder = charts_world,
  fit_type = "quadratic",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world_rents10,
  xvar = "resource_rents_2021",
  yvar = "governance_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world_rents10,
  color_legend_labels = levels(as.factor(df_world_rents10$cluster)),
  color_title = "World Cluster",
  xlab_short = "Natural resource rents (>=10%)",
  ylab_short = "Governance",
  title_long = "World sample filtered at natural resource rents >= 10% of GDP: Governance score (-2.5 to 2.5, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by World Cluster",
  filename = "world_rents10_governance_vs_resource_rents_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world_rents10,
  xvar = "resource_rents_2021",
  yvar = "services_va_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world_rents10,
  color_legend_labels = levels(as.factor(df_world_rents10$cluster)),
  color_title = "World Cluster",
  xlab_short = "Natural resource rents (>=10%)",
  ylab_short = "Services",
  title_long = "World sample filtered at natural resource rents >= 10% of GDP: Services value added (% of GDP, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by World Cluster",
  filename = "world_rents10_services_vs_resource_rents_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_world_rents10,
  xvar = "resource_rents_2021",
  yvar = "manufacturing_va_2023",
  labels_var = "country_code",
  color_vector = cluster_colors_world_rents10,
  color_legend_labels = levels(as.factor(df_world_rents10$cluster)),
  color_title = "World Cluster",
  xlab_short = "Natural resource rents (>=10%)",
  ylab_short = "Manufacturing",
  title_long = "World sample filtered at natural resource rents >= 10% of GDP: Manufacturing value added (% of GDP, 2023) vs Total natural resource rents (% of GDP, 2021) — clustered by World Cluster",
  filename = "world_rents10_manufacturing_vs_resource_rents_cluster.png",
  folder = charts_world,
  fit_type = "linear",
  fit_threshold = 0.5
)

append_txt("World sample scatter plots saved in: ", charts_world, "\n")

# -------------------------
# STEP 9 — MODEL 2 SCATTER PLOTS
# -------------------------

append_txt("STEP 9 — SCATTER PLOT DIAGNOSTICS (MODEL 2)")
append_txt("------------------------------------------")

save_scatter(
  data = df_growth,
  xvar = "log_gdp_pc_2000",
  yvar = "gdp_pc_growth",
  labels_var = "country_code",
  color_vector = growth_colors_m2,
  color_legend_labels = levels(as.factor(df_growth$core_growth_model)),
  color_title = "Core Growth Model",
  xlab_short = "Initial log GDP per capita (2000)",
  ylab_short = "GDP per capita growth",
  title_long = "GDP per capita growth (% average annual change) vs Initial log GDP per capita (2000) — clustered by Core Growth Model",
  filename = "model2_growth_vs_initial_gdp_growthmodel.png",
  folder = charts_model2,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_growth,
  xvar = "governance_avg",
  yvar = "gdp_pc_growth",
  labels_var = "country_code",
  color_vector = stability_colors_m2,
  color_legend_labels = levels(as.factor(df_growth$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "Average governance",
  ylab_short = "GDP per capita growth",
  title_long = "GDP per capita growth (% average annual change) vs Average governance score (2000-2023) — clustered by Institutional Stability",
  filename = "model2_growth_vs_governance_stability.png",
  folder = charts_model2,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_growth,
  xvar = "hci_plus_avg",
  yvar = "gdp_pc_growth",
  labels_var = "country_code",
  color_vector = stability_colors_m2,
  color_legend_labels = levels(as.factor(df_growth$institution_stability)),
  color_title = "Institutional Stability",
  xlab_short = "Average HCI+",
  ylab_short = "GDP per capita growth",
  title_long = "GDP per capita growth (% average annual change) vs Average HCI+ score (2000-2023) — clustered by Institutional Stability",
  filename = "model2_growth_vs_hci_stability.png",
  folder = charts_model2,
  fit_type = "linear",
  fit_threshold = 0.5
)

save_scatter(
  data = df_growth,
  xvar = "trade_avg",
  yvar = "gdp_pc_growth",
  labels_var = "country_code",
  color_vector = region_colors_m2,
  color_legend_labels = levels(as.factor(df_growth$region)),
  color_title = "Region",
  xlab_short = "Average trade",
  ylab_short = "GDP per capita growth",
  title_long = "GDP per capita growth (% average annual change) vs Average trade openness (% of GDP, 2000-2023) — clustered by Region",
  filename = "model2_growth_vs_trade_region.png",
  folder = charts_model2,
  fit_type = "linear",
  fit_threshold = 0.5
)

append_txt("Model 2 scatter plots saved in: ", charts_model2, "\n")

# -------------------------
# STEP 10 — MODEL 1 REGRESSIONS
# -------------------------

append_txt("STEP 10 — REGRESSION ANALYSIS (MODEL 1)")
append_txt("---------------------------------------")

m1 <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023, data = df)
m2a <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023 + manufacturing_va_2023, data = df)
m2b <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023 + services_va_2023, data = df)
m2c <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023 + manufacturing_va_2023 + services_va_2023, data = df)
m3 <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023 + resource_rents_2021 + governance_2023:resource_rents_2021, data = df)
m4 <- lm(log_gdp_pc_2023 ~ governance_2023 + hci_plus_2025 + gcf_2023 + trade_2023 + resource_rents_2021 + fdi_2023, data = df)

append_capture(summary(m1), "Model 1.1 — Baseline Development Model")
append_capture(summary(m2a), "Model 1.2a — Structure of the Economy (Manufacturing only)")
append_capture(summary(m2b), "Model 1.2b — Structure of the Economy (Services only)")
append_capture(summary(m2c), "Model 1.2c — Structure of the Economy (Manufacturing and Services)")
append_capture(summary(m3), "Model 1.3 — Resource Curse with Interaction")
append_capture(summary(m4), "Model 1.4 — Expanded Baseline with Resources and FDI")

# -------------------------
# STEP 11 — WORLD SAMPLE ROBUSTNESS
# -------------------------

append_txt("STEP 11 — WORLD SAMPLE CORRELATION ROBUSTNESS")
append_txt("---------------------------------------------")

append_capture(summary(df_world), "World Sample - Summary Statistics")
append_capture(colSums(is.na(df_world)), "World Sample - Missing Values by Column")

# -------------------------
# STEP 12 — Z-SCORE ROBUSTNESS
# -------------------------

append_txt("STEP 12 — Z-SCORE ROBUSTNESS")
append_txt("-----------------------------")

m1_z <- lm(log_gdp_pc_2023 ~ governance_z + hci_plus_z + gcf_z + trade_z, data = df_z)
append_capture(summary(m1_z), "Model 1 Z-score Baseline")

# -------------------------
# STEP 13 — GROWTH MODEL
# -------------------------

append_txt("STEP 13 — GROWTH BASELINE (MODEL 2)")
append_txt("-----------------------------------")

append_capture(names(df_growth), "Model 2 - Column Names")
append_capture(str(df_growth), "Model 2 - Structure")
append_capture(summary(df_growth), "Model 2 - Summary")
append_capture(colSums(is.na(df_growth)), "Model 2 - Missing Values by Column")

m_growth <- lm(gdp_pc_growth ~ log_gdp_pc_2000 + governance_avg + hci_plus_avg + gcf_avg + trade_avg, data = df_growth)
append_capture(summary(m_growth), "Model 2 — Growth Baseline")

# -------------------------
# STEP 14 — FINAL MESSAGE
# -------------------------

append_txt("Coding completed successfully.\n")
append_txt("Model 1 charts folder: ", charts_model1)
append_txt("Model 2 charts folder: ", charts_model2)
append_txt("World sample charts folder: ", charts_world)
append_txt("Results summary file: ", results_txt)