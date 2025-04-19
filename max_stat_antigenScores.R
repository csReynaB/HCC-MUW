library("maxstat")
library("survival")


setwd("/home/creyna/Vogl-lab_Projects_git/HCC/")
df <- read.csv("antigen_scores_test.csv")
head(df)
# Create a survival object
surv_obj <- Surv(df$OS.months, df$OS.Status)
df$Antigen.Score..Scaled.
# Run the maxstat test to find the optimal cutpoint for antigen_score
result <- maxstat.test(surv_obj ~ `Antigen.Score..Scaled.`, data = df,
                       smethod = "LogRank",  # using log-rank test statistic
                       pmethod = "condMC",   # conditional Monte Carlo p-value estimation
                       B = 9999)
                       #minprop = 0.1,        # minimum 10% of observations in each group
                       #maxprop = 0.9)        # maximum 90% of observations in each group

# View the optimal cutpoint
print(result$estimate)

as.vector(result$estimate)
####
df <- read.csv("clinical_meta_HCC-MUW_extraInfo.csv")
head(df)
# Create a survival object
surv_obj <- Surv(df$OS.months, df$OS.Status)
# Run the maxstat test to find the optimal cutpoint for antigen_score
result <- maxstat.test(surv_obj ~ `Model.2.Score`, data = df,
                       smethod = "LogRank",  # using log-rank test statistic
                       pmethod = "condMC",   # conditional Monte Carlo p-value estimation
                       B = 9999)
#minprop = 0.1,        # minimum 10% of observations in each group
#maxprop = 0.9)        # maximum 90% of observations in each group

# View the optimal cutpoint
print(result$estimate)
result
