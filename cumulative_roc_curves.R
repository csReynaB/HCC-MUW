# Load required packages
library(survivalROC)
library(ggplot2)
library(dplyr)

risk_scores <- read.csv("Data/risk_scores_time_event_train_df.csv")
antigen_scores <- read.csv("Data/antigen_scores_time_event_train_df.csv")

time <- 18
# Compute the ROC curves at 12 months for both predictors
roc12_risk <- survivalROC(Stime = risk_scores$OS.months,
                          status = risk_scores$OS.Status,
                          marker = risk_scores$X42,
                          predict.time = time,
                          method = "KM")

roc12_antigen <- survivalROC(Stime = antigen_scores$OS.months,
                          status = antigen_scores$OS.Status,
                          marker = antigen_scores$X42,
                          predict.time = time,
                          method = "KM")


# Create data frames for ggplot from the survivalROC output
roc12_risk_df <- data.frame(
  FP = roc12_risk$FP,
  TP = roc12_risk$TP,
  Predictor = paste0("Risk scores (AUC=", round(roc12_risk$AUC, 2), ")")
)

roc12_antigen_df <- data.frame(
  FP = roc12_antigen$FP,
  TP = roc12_antigen$TP,
  Predictor = paste0("Antigen scores (AUC=", round(roc12_antigen$AUC, 2), ")")
)

# Combine the data frames
roc12_df <- rbind(roc12_risk_df, roc12_antigen_df)

# Plot using ggplot2
ggplot(roc12_df, aes(x = FP, y = TP, color = Predictor)) +
  geom_line(size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = paste0("Time-dependent ROC Curve at ", time,"  Months"),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()


##########################################
##########################################
fp_grid <- seq(0, 1, length.out = 100)

# Function to interpolate a ROC curve to the common FP grid:
interpolate_roc <- function(fp, tp, common_fp){
  interp_tp <- approx(x = fp, y = tp, xout = common_fp, method = "linear", yleft = 0, yright = 1)$y
  # Ensure interpolation stays in [0,1]
  pmin(pmax(interp_tp, 0), 1)
}

# Assume first three columns are sampleName, OS.months, OS.Status
risk_repeat_cols <- colnames(risk_scores)[4:ncol(risk_scores)]
antigen_repeat_cols <- colnames(antigen_scores)[4:ncol(antigen_scores)]


# Initialize lists for storing ROC curves and AUCs
roc_risk_list <- list()
auc_risk_list <- c()

for(i in seq_along(risk_repeat_cols)) {
  score_col <- risk_repeat_cols[i]
  marker <- risk_scores[[score_col]]
  
  roc_obj <- survivalROC(
    Stime = risk_scores$OS.months,
    status = risk_scores$OS.Status,
    marker = marker,
    predict.time = time,
    method = "KM"
  )
  
  interp_tp <- interpolate_roc(roc_obj$FP, roc_obj$TP, fp_grid)
  roc_df <- data.frame(FP = fp_grid, TP = interp_tp)
  
  # Create a data frame for this ROC curve
  temp_df <- data.frame(
    FP = roc_df$FP,
    TP = roc_df$TP,
    Predictor = "Risk scores",
    Repeat = paste0("Repeat_", i)
  )
  temp_df$AUC <- roc_obj$AUC  # the AUC value for this repeat
  roc_risk_list[[i]] <- temp_df
  auc_risk_list[i] <- roc_obj$AUC
}

# Process antigen scores similarly
roc_antigen_list <- list()
auc_antigen_list <- c()

for(i in seq_along(antigen_repeat_cols)) {
  score_col <- antigen_repeat_cols[i]
  marker <- antigen_scores[[score_col]]
  
  roc_obj <- survivalROC(
    Stime = antigen_scores$OS.months,
    status = antigen_scores$OS.Status,
    marker = marker,
    predict.time = time,
    method = "KM"
  )
  
  interp_tp <- interpolate_roc(roc_obj$FP, roc_obj$TP, fp_grid)
  roc_df <- data.frame(FP = fp_grid, TP = interp_tp)

  temp_df <- data.frame(
    FP = roc_df$FP,
    TP = roc_df$TP,
    Predictor = "Antigen scores",
    Repeat = paste0("Repeat_", i)
  )
  temp_df$AUC <- roc_obj$AUC
  roc_antigen_list[[i]] <- temp_df
  auc_antigen_list[i] <- roc_obj$AUC
}


# Combine all ROC curves from risk and antigen scores
roc_risk_df <- do.call(rbind, roc_risk_list)
roc_antigen_df <- do.call(rbind, roc_antigen_list)
roc_all_df <- rbind(roc_risk_df, roc_antigen_df)

# Calculate the mean ROC curve for risk scores (group by FP)
mean_risk_df <- roc_risk_df %>%
  group_by(FP) %>%
  summarise(TP = mean(TP)) %>%
  mutate(Predictor = "Risk scores",
         AUC = mean(auc_risk_list))

# Calculate the mean ROC curve for antigen scores
mean_antigen_df <- roc_antigen_df %>%
  group_by(FP) %>%
  summarise(TP = mean(TP)) %>%
  mutate(Predictor = "Antigen scores",
         AUC = mean(auc_antigen_list))
mean_scores_df <- rbind(mean_risk_df, mean_antigen_df)


mean_auc_antigen <- round(unique(mean_scores_df$AUC[mean_scores_df$Predictor == "Antigen scores"]), 2)
mean_auc_risk <- round(unique(mean_scores_df$AUC[mean_scores_df$Predictor == "Risk scores"]), 2)


new_labels <- c("Antigen scores" = paste0("Antigen scores\n(AUC=", mean_auc_antigen, ")"),
                "Risk scores" = paste0("\nRisk scores\n(AUC=", mean_auc_risk, ")"))
ggplot(mean_scores_df, aes(x = FP, y = TP, color = Predictor)) +
  # Plot individual repeats with a lower alpha (transparency)
  geom_line(size = 1, alpha = 1) +
  geom_line(data = roc_all_df, aes(x = FP, y = TP, color = Predictor, group = interaction(Predictor, Repeat) ),
            size = 1, alpha = 0.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = paste0("Time-dependent ROC Curve at ", time, " Months"),
       x = "False Positive Rate",
       y = "True Positive Rate") +
  scale_color_manual(values = c("Antigen scores" = "cadetblue", "Risk scores" = "darkgreen"),
                     labels = new_labels) +
  theme_linedraw()

paste0("Antigen scores (AUC=", round(mean_scores_df$AUC, 2), ")")


roc_all_df %>%
  group_by(Predictor) %>% 
  summarise(mean(AUC))
