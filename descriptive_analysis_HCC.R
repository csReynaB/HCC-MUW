library("stats")  # For chi-squared test
library("multcomp")  # For multiple testing correction
library("dplyr")
library("ggplot2")
library("ggsignif")
library("plotly")

library("ggvenn")
library("nnet")
library("MatchIt")

library(reticulate)
use_condaenv("ML_env", required = TRUE)
py_config()

#LIB_METADATA <- 'all_libraries_with_important_info.pkl'
LIB_METADATA <- 'Metadata/aligent_twist_with_important_info_excludeSARS.pkl'
#setwd("/home/creyna/Vogl-lab_Projects_git/HCC/phip_seq_DB")

hcc_muw_db <- read.csv("phipseqDB/hcc_muw_with_controls_phipseq_DB.csv", sep = ",")

exist <- read.csv("Data/exist_noSARSpeptides.csv", header = TRUE, row.names = 1)
shannon_diversity <- vegan::diversity(t(exist), index = "shannon", base = 2)
samples_stats <- data.frame(SampleName = names(shannon_diversity), count_filter = colSums(exist), Shannon_diversity = shannon_diversity)

metadata <- readxl::read_excel("Metadata/HCC_MUW_metadata_allSamples.xlsx", sheet = 1) 
metadata <- merge(metadata, samples_stats, by = "SampleName", all.x = TRUE)


## Quality Check
hcc_muw_db <- hcc_muw_db %>%
  mutate(sample_type = case_when(
    grepl("Anchor", sample_name) ~ "Anchor",
    grepl("Mock", sample_name) ~ "Mock",
    grepl("R14P01", sample_name) ~ "HCC-ICI",
    grepl("R22P04", sample_name) ~ "HCC-TKI",
    TRUE ~ "Controls"
  ))
hcc_muw_db[hcc_muw_db$sample_name %in% "R14P01_79_0106492769_HCC_MUW_A_T_C2",]$sample_type <- "Controls"


df_summary <- hcc_muw_db %>%
  group_by(sample_name, sample_type, Run_plate, sequencing_run) %>%
  summarise(count = n()) %>%
  rename(SampleName = sample_name) %>%
  ungroup() %>%
  left_join(samples_stats, by = "SampleName")
factor(df_summary$sample_type)

#Reorder sample_type as a factor with the desired order
df_summary$sample_type <- factor(df_summary$sample_type, 
                                 levels = c("HCC-ICI", "HCC-TKI", "Controls","Anchor", "Mock"))

custom_colors <- c("HCC-ICI" = "dodgerblue3",
                   "HCC-TKI" = "palegreen3",
                   "Controls" = "red3",
                   "Anchor" = "gray", 
                   "Mock" = "violet")  


##########3
df_counts <- metadata %>%
  group_by(group_test) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(group_test, "\n(n = ", sample_count,")"), group_test))

# View the updated dataframe
ggplot(metadata, aes(x = group_test, y = count_filter, fill = group_test)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("Controls", "HCC")), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)


#########
df_counts <- df_summary %>%
  group_by(sample_type, Run_plate) %>%
  summarize(sample_count = n(), .groups = 'drop')  # Count the number of samples per group


# View the updated dataframe
ggplot(df_summary, aes(x = sample_type, y = count, fill = sample_type)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  geom_text(data = df_counts, aes(x = sample_type, y = max(df_summary$count) + 50, label = paste("n =", sample_count)),
            position = position_dodge(width = 0.75), vjust = 0, alpha = 0.45, size = 3) +  # Adjust y offset as needed
  
  facet_grid(~ Run_plate, scales = "free_x", space = "free_x",
             labeller = labeller(Run_plate = c("P01" = "SQR 7 Plate 1", "P02" = "SQR 7 Plate 2", "4.0" = "SQR15 Plate 4"))) + # Separate plots with better spacing between P01 and P02
  
  labs(
    x = "Samples",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  #scale_x_discrete(labels = levels(df_summary$sample_type_label)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels 45 degrees
        strip.background = element_blank(),  # Remove the background of facet labels (P01, P02)
        strip.text = element_text(face = "bold"),  # Make the facet labels bold
        panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("HCC-ICI", "HCC-TKI"),
                                                c("HCC-ICI", "Controls"),
                                                c("HCC-TKI", "Controls")), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  


##################
df_counts <- df_summary %>%
  group_by(sample_type) %>%
  summarize(sample_count = n())

# View the updated dataframe
ggplot(df_summary, aes(x = sample_type, y = count, fill = sample_type)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  geom_text(data = df_counts, aes(x = sample_type, y = max(df_summary$count_filter) + 50, label = paste("n =", sample_count)),
            position = position_dodge(width = 0.75), vjust = 0, alpha = 0.45, size = 3) +  # Adjust y offset as needed
  
  labs(
    x = "Samples",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels 45 degrees
        panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("HCC-ICI", "HCC-TKI"),
                                                c("HCC-ICI", "Controls"),
                                                c("HCC-TKI", "Controls")), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  



################
df_counts <- metadata %>%
  group_by(subgroups) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(subgroups, "\n(n = ", sample_count,")"), subgroups))

# View the updated dataframe
ggplot(metadata, aes(x = subgroups, y = count_filter, fill = subgroups)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("Controls", "HCC-ICI"),
                                                c("Controls", "HCC-TKI"),
                                                c("HCC-ICI", "HCC-TKI")), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  
ggsave("plots/counts_tratments.png", width = 4, height = 4, dpi=300)

###############
metadata <- metadata %>%
  mutate(subgroup_orr = case_when(
    subgroups == "Controls" ~ "Controls",
    subgroups == "HCC-ICI" & ORR_ctg == "ORR" ~ "HCC-ICI\nORR",
    subgroups == "HCC-ICI" & ORR_ctg == "non ORR" ~ "HCC-ICI\nnon ORR",
    subgroups == "HCC-TKI" & ORR_ctg == "ORR" ~ "HCC-TKI\nORR",
    subgroups == "HCC-TKI" & ORR_ctg == "non ORR" ~ "HCC-TKI\nnon ORR",
    TRUE ~ NA_character_
  )) 

df_counts <- metadata %>%
  # Exclude rows where new_group is NA
  filter(!is.na(subgroup_orr))  %>%
  group_by(subgroup_orr) %>%
  summarize(sample_count = n())

x_labels <- with(df_counts, setNames(paste0(subgroup_orr, "\n(n = ", sample_count, ")"), subgroup_orr))
# View the updated dataframe
ggplot(metadata%>%
         filter(!is.na(subgroup_orr)), 
       aes(x = subgroup_orr, y = count_filter, fill =  subgroup_orr)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  #facet_grid(~ subgroups, scales = "free_x", space = "free_x") +
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  #scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(
                               c("Controls", "HCC-ICI\nnon ORR"),
                               c("Controls", "HCC-ICI\nORR"),
                               c("Controls", "HCC-TKI\nnon ORR"),
                               c("Controls", "HCC-TKI\nORR"),
                               c("HCC-ICI\nORR", "HCC-ICI\nnon ORR"),
                               c("HCC-TKI\nORR", "HCC-TKI\nnon ORR")
                             ), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  



################
metadata <- metadata %>%
  mutate(subgroup_dcr = case_when(
    subgroups == "Controls" ~ "Controls",
    subgroups == "HCC-ICI" & DCR_ctg == "DCR" ~ "HCC-ICI\nDCR",
    subgroups == "HCC-ICI" & DCR_ctg == "non DCR" ~ "HCC-ICI\nnon DCR",
    subgroups == "HCC-TKI" & DCR_ctg == "DCR" ~ "HCC-TKI\nDCR",
    subgroups == "HCC-TKI" & DCR_ctg == "non DCR" ~ "HCC-TKI\nnon DCR",
    TRUE ~ NA_character_
  )) 

df_counts <- metadata %>%
  # Exclude rows where new_group is NA
  filter(!is.na(subgroup_dcr))  %>%
  group_by(subgroup_dcr) %>%
  summarize(sample_count = n())

x_labels <- with(df_counts, setNames(paste0(subgroup_dcr, "\n(n = ", sample_count, ")"), subgroup_dcr))
# View the updated dataframe
ggplot(metadata%>%
         filter(!is.na(subgroup_dcr)), 
       aes(x = subgroup_dcr, y = count_filter, fill =  subgroup_dcr)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  #facet_grid(~ subgroups, scales = "free_x", space = "free_x") +
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Count of Enriched Peptides",
    fill = "Samples") +
  #scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(
                               c("Controls", "HCC-ICI\nDCR"),
                               c("Controls", "HCC-ICI\nnon DCR"),
                               c("Controls", "HCC-TKI\nDCR"),
                               c("Controls", "HCC-TKI\nnon DCR"),
                               c("HCC-ICI\nDCR", "HCC-ICI\nnon DCR"),
                               c("HCC-TKI\nDCR", "HCC-TKI\nnon DCR")
                             ), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  


#######################
df_counts <- metadata %>%
  group_by(group_test) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(group_test, "\n(n = ", sample_count,")"), group_test))

# View the updated dataframe
ggplot(metadata, aes(x = group_test, y = Shannon_diversity, fill = group_test)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Alpha Diversity (Shannon iIndex)",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("Controls", "HCC")
                             ), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  
ggsave("plots/diversity_treatments.png", width = 4, height = 4, dpi=300)




###
df_counts <- metadata %>%
  group_by(subgroups) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(subgroups, "\n(n = ", sample_count,")"), subgroups))

# View the updated dataframe
ggplot(metadata, aes(x = subgroups, y = Shannon_diversity, fill = subgroups)) +
  geom_boxplot(show.legend = F) +
  geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = F) +  
  # Add the counts on top of the boxplots
  labs(
    x = "Group Test",
    y = "Alpha Diversity (Shannon iIndex)",
    fill = "Samples") +
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  scale_x_discrete(labels = x_labels) +  # Use the custom labels
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),  # Rotate x-axis labels 45 degrees
    panel.grid = element_blank()) + # Remove the gridlines
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = list(c("Controls", "HCC-ICI"),
                                                c("Controls", "HCC-TKI"),
                                                c("HCC-ICI", "HCC-TKI")), 
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  
ggsave("plots/diversity_treatments.png", width = 4, height = 4, dpi=300)



######################
metadata <- metadata %>%
  mutate(Age_group = cut(Age, breaks = seq(20, 95, by = 5), right = FALSE))  # Modify age groups as needed
metadata$Sex_ctg <- factor(metadata$Sex, levels = c(0, 1), labels = c("Female", "Male"))

chisq_result <- chisq.test(table(metadata$subgroups, metadata$Sex_ctg))
print(chisq_result)

# Assuming your merged grouping variable is 'new_group' in metadata_updated
model <- multinom(subgroups ~ Age + Sex, data = metadata)
summary(model)

aov_model <- aov(Age ~ subgroups * Sex_ctg, data = metadata)
aov_results <- as.data.frame(summary(aov_model)[[1]])
# Extract and tidy the ANOVA results
tidy_aov <- broom::tidy(aov_model)
tidy_aov$p.value <- round(tidy_aov$p.value, 3)
knitr::kable(tidy_aov, digits = 3, caption = "Two-way ANOVA Results for Age by Controls/HCC-ICI/HCC-TKI and Gender")

#Age:The significant negative association for Age in the HCC-TKI group suggests that age might be a factor distinguishing this subgroup from Controls. In other words, if HCC-TKI patients tend to be younger, this may indicate an age bias between subgroups.
#Sex: Neither subgroup shows a significant difference by sex (the coefficients for Sex_ctgMale are very small or non-significant), so there doesn’t appear to be a sex bias based on these results.

data_summary <- metadata %>%
  group_by(subgroups, Sex_ctg, Age_group) %>%
  summarize(count = n(), .groups = 'drop') %>%
  ungroup() %>%
  mutate(count = ifelse(Sex_ctg == "Male", -count, count))  # Make Male counts negative for mirroring

df_counts <- metadata %>%
  group_by(subgroups) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(subgroups, "\n(n = ", sample_count,")"), subgroups))

ggplot(data_summary, aes(x = ifelse(subgroups == "Controls", count, -count), y = Age_group, fill = Sex_ctg)) +
  geom_bar(stat = "identity", position = "identity", width = 0.85) +  # Slightly narrower bars for elegance
  scale_x_continuous(
    labels = function(x) ifelse(x %in% c(-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15), abs(x), ""),
    breaks = seq(-15, 15, by = 3)
  ) +
  scale_fill_manual(values = RColorBrewer::brewer.pal(3, "Set2")[c(1, 2)]) +  # Custom colors with better contrast
  facet_grid(~ subgroups, scales = "free_x", space = "free_x", labeller = labeller(subgroups = x_labels)) +
  labs(
    x = "Counts",
    y = "Age Group",
    fill = NULL  # Remove legend title
  ) +
  theme_minimal(base_size = 11) +  # Use a minimal theme for elegance and set a base font size
  theme(
    legend.position = "top",  # Place legend at the top
    legend.text = element_text(size = 11),  # Increase legend text size for readability
    strip.background = element_blank(),  # Remove facet background
    strip.text = element_text(face = "bold", size = 10),  # Make facet labels bold
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"),  # Lighter, dashed vertical grid lines
    panel.grid.major.y = element_blank(),  # Remove horizontal grid lines for a cleaner look
    axis.text.y.left = element_text(size = 10, face = "italic"),  # Style y-axis labels
    axis.text.y.right = element_blank(),  # Remove right-side y-axis text
    axis.ticks.y.right = element_blank(),  # Remove right-side y-axis ticks
    axis.title.x = element_text(margin = margin(t = 10), face = "bold"),  # Add margin and bold to x-axis title
    axis.title.y = element_text(face = "bold"),  # Bold y-axis title
    legend.box.spacing = unit(0.2, "lines")  # Reduce spacing around the legend box
  )
ggsave("plots/sexAgedistribution.png", width = 4.6, height = 5, dpi=300, bg="white")


####################
# Perform a chi-square test
chisq_result <- chisq.test(table(metadata$group_test, metadata$Sex_ctg))
print(chisq_result)

aov_model <- aov(Age ~ group_test * Sex_ctg, data = metadata)
aov_results <- as.data.frame(summary(aov_model)[[1]])
tidy_aov <- broom::tidy(aov_model)
tidy_aov$p.value <- round(tidy_aov$p.value, 3)
knitr::kable(tidy_aov, digits = 3, caption = "Two-way ANOVA Results for Age by Controls/HCC and Sex")

# Assuming your merged grouping variable is 'new_group' in metadata_updated
model <- multinom(group_test ~ Age + Sex, data = metadata)
summary(model)

#The model suggests that neither Age nor Sex has a strong or statistically significant effect on the odds of being in the HCC group versus Controls, given the magnitude of the coefficients relative to their standard errors.
#The negative coefficient for Age indicates a very slight trend that older age might be associated with lower odds of being in the HCC group, but the effect is very small.
#The positive coefficient for Sex suggests a slight increase in odds for being in the HCC group if the individual is male, but this effect is also very small.
#Overall, based on these coefficients, there is little evidence from this model that either Age or Se

data_summary$Age_group
data_summary <- metadata %>%
  group_by(group_test, Sex_ctg, Age_group) %>%
  summarize(count = n(), .groups = 'drop') %>%
  ungroup() %>%
  mutate(count = ifelse(Sex_ctg == "Male", -count, count))  # Make Male counts negative for mirroring

df_counts <- metadata %>%
  group_by(group_test) %>%
  summarize(sample_count = n())
x_labels <- with(df_counts, setNames(paste0(group_test, "\n(n = ", sample_count,")"), group_test))

ggplot(data_summary, aes(x = ifelse(group_test == "Controls", count, -count), y = Age_group, fill = Sex_ctg)) +
  geom_bar(stat = "identity", position = "identity", width = 0.85) +  # Slightly narrower bars for elegance
  scale_x_continuous(
    labels = function(x) ifelse(x %in% c(-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15), abs(x), ""),
    breaks = seq(-15, 15, by = 3)
  ) +
  scale_fill_manual(values = RColorBrewer::brewer.pal(3, "Set2")[c(1, 2)]) +  # Custom colors with better contrast
  facet_grid(~ group_test, scales = "free_x", space = "free_x", labeller = labeller(group_test = x_labels)) +
  labs(
    x = "Counts",
    y = "Age Group",
    fill = NULL  # Remove legend title
  ) +
  theme_minimal(base_size = 11) +  # Use a minimal theme for elegance and set a base font size
  theme(
    legend.position = "top",  # Place legend at the top
    legend.text = element_text(size = 11),  # Increase legend text size for readability
    strip.background = element_blank(),  # Remove facet background
    strip.text = element_text(face = "bold", size = 10),  # Make facet labels bold
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    panel.grid.major.x = element_line(color = "grey90", linetype = "dashed"),  # Lighter, dashed vertical grid lines
    panel.grid.major.y = element_blank(),  # Remove horizontal grid lines for a cleaner look
    axis.text.y.left = element_text(size = 10, face = "italic"),  # Style y-axis labels
    axis.text.y.right = element_blank(),  # Remove right-side y-axis text
    axis.ticks.y.right = element_blank(),  # Remove right-side y-axis ticks
    axis.title.x = element_text(margin = margin(t = 10), face = "bold"),  # Add margin and bold to x-axis title
    axis.title.y = element_text(face = "bold"),  # Bold y-axis title
    legend.box.spacing = unit(0.2, "lines")  # Reduce spacing around the legend box
  )
ggsave("plots/sexAgedistribution.png", width = 4.6, height = 5, dpi=300, bg="white")

######
metadata <- metadata %>%
  mutate(group_test_dcr = case_when(
    group_test == "Controls" ~ "Controls",
    group_test == "HCC" & DCR_ctg == "DCR" ~ "HCC - Responders",
    group_test == "HCC" & DCR_ctg == "non DCR" ~ "HCC - non Responders",
    TRUE ~ NA_character_
  )) 

metadata <- metadata %>%
  mutate(subgroup_dcr = case_when(
    subgroups == "Controls" ~ "Controls",
    subgroups == "HCC-ICI" & DCR_ctg == "DCR" ~ "HCC-ICI Responders",
    subgroups == "HCC-ICI" & DCR_ctg == "non DCR" ~ "HCC-ICI non Responders",
    subgroups == "HCC-TKI" & DCR_ctg == "DCR" ~ "HCC-TKI\nDCR",
    subgroups == "HCC-TKI" & DCR_ctg == "non DCR" ~ "HCC-TKI\nnon DCR",
    TRUE ~ NA_character_
  )) 

#- Prevalence Filtering: To ensure robust results, a prevalence filter of 5% was applied separately to each group test. This means that a peptide must be present in at least 5% of samples for all groups within each group test (e.g., 5% in both Controls AND HCC) to be considered in the comparison.
tmp <- as.data.frame(t(exist)) %>%
  tibble::rownames_to_column("SampleName")%>%
  dplyr::left_join(metadata %>%
                     dplyr::select(SampleName,  subgroup_dcr) #group_test, subgroups, subgroup_orr, 
                   ,by = "SampleName")
#group_test_columns <- colnames(tmp)[grepl("^subgroups", colnames(tmp))]
group_test_columns <- c("subgroup_dcr")#"subgroups", "group_test") #, "subgroup_orr", "subgroup_dcr")

# Calculate the total number of samples for each group within each group_test column
num_samples_per_group <- list()
for (group_col in group_test_columns) {
  num_samples_per_group[[group_col]] <- metadata %>%
    filter(!is.na(!!sym(group_col))) %>%  # Filter out NA values in the group_test column
    group_by(!!sym(group_col)) %>%
    summarise(n = n(), .groups = "drop") %>%
    tibble::deframe()  # Convert the summarized result to a named vector (group -> count)
}



prevalence_threshold <- 2
percentage_group_test_list <- list()
for (group_col in group_test_columns) {
  # Calculate percentage of presence for each peptide within each group
  percentage_df <- tmp %>%
    filter(!is.na(!!sym(group_col))) %>%  
    tidyr::gather(key = "Peptide", value = "Presence", -SampleName, -all_of(group_test_columns)) %>%
    group_by(!!sym(group_col), Peptide) %>%
    summarise(Percent = mean(Presence) * 100, 
              # Count = sum(Presence),           # Calculate count of presence
              .groups = "drop") %>%
    tidyr::spread(key = !!sym(group_col), value = Percent) %>%
    left_join(as.data.frame(reticulate::py_load_object(LIB_METADATA, pickle = "pickle")) %>%
                tibble::rownames_to_column(var = "Peptide") %>%
                dplyr::select(Peptide, Description, `full name`, Organism_complete_name), 
              by = "Peptide")
  
  # percentage_df <- percentage_df %>%
  #   left_join(twist,  by = "Peptide") %>%
  #   mutate(`full name` = coalesce(`full name.x`, `full name.y`)) %>%
  #   select(-`full name.x`, -`full name.y`)
  
  # Identify all percentage columns dynamically
  percentage_columns <- setdiff(colnames(percentage_df), c("Peptide", "Description", "full name", "Organism_complete_name"))
  
  # Apply the prevalence filter on all identified percentage columns
  percentage_df <- percentage_df %>%
    filter(if_any(all_of(percentage_columns), ~ . >= prevalence_threshold)) #if_all
  
  # Sort the dataframe based on "Controls" if the column exists
  if ("Controls" %in% colnames(percentage_df)) {
    percentage_df <- percentage_df %>%
      arrange(Controls)
  }
  
  # Add the count columns dynamically based on percentages and number of samples per group
  for (column in percentage_columns) {
    num_samples_group <- num_samples_per_group[[group_col]][[column]]  # Get the number of samples for the current group
    percentage_df[[paste0(column, "_count")]] <- round(percentage_df[[column]] / 100 * num_samples_group, 0)
  }
  
  # Save the dataframe to the list
  percentage_group_test_list[[group_col]] <- percentage_df
}


# Function to automate the entire process for each group_test in the list
automate_group_test_analysis <- function(percentage_group_test_list, num_samples_per_group) {
  results_list <- list()
  
  # Loop through each group_test column in the list
  for (group_col in names(percentage_group_test_list)) {
    # Get the dataframe for the current group_test
    df <- percentage_group_test_list[[group_col]]
    
    # Identify unique groups within this group_test column
    groups <- colnames(df)[grepl("_count$", colnames(df))] %>%
      sub("_count$", "", .)
    
    # Ensure there are at least two groups for pairwise comparison
    if (length(groups) < 2) next
    
    
    # Generate pairwise comparisons for each group_test column
    comparison_results <- list()
    for (i in 1:(length(groups) - 1)) {
      for (j in (i + 1):length(groups)) {
        group1 <- groups[i]
        group2 <- groups[j]
        
        # Calculate p-values and ratios for each pair of groups
        ratio_vals <- c()
        pvals_chisq <- c()
        epsilon <- 0.001
        # Loop over each row in the filtered dataframe
        for (k in 1:nrow(df)) {
          # Extract the values for the two groups being compared
          val1 <- df[[paste0(group1, "_count")]][k]
          val2 <- df[[paste0(group2, "_count")]][k]
          
          # Adjust zero values by setting them to 1 (similar to Python logic)
          if (val1 == 0) val1 <- 1
          if (val2 == 0) val2 <- 1
          
          # Construct the contingency table using counts from num_samples_per_group
          chitable <- matrix(c(val1 + 1, num_samples_per_group[[group_col]][[group1]] - val1 + 1, 
                               val2 + 1, num_samples_per_group[[group_col]][[group2]] - val2 + 1), 
                             nrow = 2, byrow = TRUE)
          
          # Perform the Fisher's test on the constructed contingency table
          test_result <- fisher.test(chitable)
          
          # Store the resulting p-value
          pvals_chisq <- c(pvals_chisq, test_result$p.value)
          
          # Calculate the ratio (handling potential zeroes)
          if (val1 == 0 | val2 == 0) {
            ratio_vals[k] <- NA
          } else {
            if (val1 >= val2) {
              ratio_vals[k] <- val1 / val2 - 1
            } else {
              ratio_vals[k] <- -(val2 / val1 - 1)
            }
          }
        }
        
        # Add p-values and ratios to the dataframe
        comparison_df <- df %>%
          mutate(
            Delta_ratio = ratio_vals,
            ratio = log10((!!sym(group1) + epsilon) / (!!sym(group2) + epsilon)),
            pvals_not_corr = pvals_chisq
            
          ) %>%
          mutate(
            Significant = ifelse(pvals_not_corr < 0.05, "Yes", "No"),
            pvals_bh = p.adjust(pvals_not_corr, method = "BH"),
            passed_bh = p.adjust(pvals_not_corr, method = "BH") < 0.05,
            Significant_bh = ifelse(passed_bh, "Yes", "No")
          )  %>%
          dplyr::arrange(desc(ratio), ratio) %>%
          dplyr::select(Peptide, Description, `full name`, everything())
        
        # Generate scatter plot for the comparison
        p <- ggplot(comparison_df, aes(x = !!sym(group1), y = !!sym(group2), Peptide = Peptide, Description = Description, Organism=Organism_complete_name)) + #text = paste('Peptide:', Peptide, '<br>Description:', Description))) +
          geom_point(data = subset(comparison_df, Significant == "No"), 
                     aes(color = "not significant"), alpha = 0.35) +
          geom_point(data = subset(comparison_df, Significant == "Yes"), 
                     aes(color = "significant prior correction"), alpha = 0.6) +
          geom_point(data = subset(comparison_df, Significant_bh == "Yes"), 
                     aes(color = "significant post FDR correction"), alpha = 1) +
          scale_color_manual(values = c("not significant" = "steelblue", 
                                        "significant prior correction" = "forestgreen", 
                                        "significant post FDR correction" = "firebrick")) +
          labs(
            x = paste0("% ", group1, " in whom\na peptide is significantly bound\n(n = ",
                       num_samples_per_group[[group_col]][[group1]], ")"),
            y = paste0("% ", group2, " in whom\na peptide is significantly bound\n(n = ", 
                       num_samples_per_group[[group_col]][[group2]], ")"),
            color = "Significance"
          ) +
          theme_bw(base_size = 12) +  # Use a minimal theme for elegance and set a base font size
          theme(
            legend.position = "none",
            aspect.ratio = 1,
            panel.grid.major = element_blank(),  # Remove major grid lines
            panel.grid.minor = element_blank(),  # Remove minor grid lines
            panel.border = element_rect(colour = "black", fill = NA),  # Keep border if desired
            plot.margin = margin(t = 10, r = 15, b = 15, l = 10, unit = "pt"),  # Adjust margins in point
            #plot.margin = margin(c(0, 0, 0, 0.5), "cm"),  # Reduce margins around the plot
            axis.text.y.left = element_text(size = 10, face = "italic"),  # Style y-axis labels
            axis.text.y.right = element_blank(),  # Remove right-side y-axis text
            axis.ticks.y.right = element_blank(),  # Remove right-side y-axis ticks
            axis.title.x = element_text(face = "bold"),  # Add margin and bold to x-axis title
            axis.title.y = element_text(face = "bold")  # Bold y-axis title
          )
        
        # Convert to plotly for interactivity
        interactive_plot <- ggplotly(p, tooltip = c("Peptide", "Description", "Organism", group1, group2)) %>%
          layout(
            #height = 475,  # Define the plot height in pixels
            #width = 475,
            showlegend = FALSE,
            margin = list(
              l = 50,  # Left margin
              r = 50,  # Right margin
              b = 50,  # Bottom margin
              t = 50,   # Top margin
              pad = 10
            )
          )
        
        
        # Store the plot and results in the list
        comparison_results[[paste0(group1, "_vs_", group2)]] <- list(
          plot = interactive_plot,
          comparison_df = comparison_df
        )
      }
    }
    
    # Store all comparison results for this group_test
    results_list[[group_col]] <- comparison_results
  }
  
  return(results_list)
}

comparison_group_tests <- automate_group_test_analysis(percentage_group_test_list, num_samples_per_group)
comparison_group_tests[[1]]



####################
binary_data_all <- tmp %>%
  dplyr::select(-group_test, -subgroups, -subgroup_orr, -subgroup_dcr) %>%
  tibble::column_to_rownames("SampleName")


binary_data_orr_dcr <- tmp %>%
  dplyr::filter(!is.na(subgroup_orr))%>%
  dplyr::select(-group_test, -subgroups, -subgroup_orr, -subgroup_dcr) %>%
  tibble::column_to_rownames("SampleName")

dist_matrix_all <- vegan::vegdist(binary_data_all, method = "jaccard") #dist(binary_data_all, method="binary")
dist_matrix_orr_dc <- vegan::vegdist(binary_data_orr_dcr, method = "jaccard")

mds_result_all <- cmdscale(dist_matrix_all, k = 2, eig = T)  # k = 2 for two dimensions
mds_result_orr_dc <- cmdscale(dist_matrix_orr_dc, k = 2, eig = T)  # k = 2 for two dimensions

variance_explained_all <- round(100 * mds_result_all$eig[1:2] / sum(mds_result_all$eig), 2)
variance_explained_orr_dc <- round(100 * mds_result_orr_dc$eig[1:2] / sum(mds_result_orr_dc$eig), 2)

# Convert MDS result to a dataframe for visualization
mds_df_all <- as.data.frame(mds_result_all$points) %>%
  tibble::rownames_to_column("SampleName")
mds_df_orr_dc <- as.data.frame(mds_result_orr_dc$points) %>%
  tibble::rownames_to_column("SampleName")

# Merge MDS result with the group_test column(s)
mds_df_all <- mds_df_all %>%
  left_join(tmp %>%
              dplyr::select(SampleName, group_test, subgroups, subgroup_orr, subgroup_dcr), by = "SampleName")
mds_df_orr_dc <- mds_df_orr_dc %>%
  left_join(tmp %>%
              dplyr::select(SampleName, group_test, subgroups, subgroup_orr, subgroup_dcr), by = "SampleName")


#######
#Permanova test between groups
permanova_result <- vegan::adonis2(dist_matrix_all ~ subgroups, data = mds_df_all, permutations = 999)
p_value <- permanova_result$`Pr(>F)`[1]

centroids <- mds_df_all %>%
  group_by(subgroups) %>%
  summarise(V1 = mean(V1), V2 = mean(V2))

p <- ggplot(mds_df_all, aes(x = V1, y = V2, fill = subgroups)) +  # Replace with desired group_test column
  geom_point(size = 3, alpha = 0.5, shape = 21, color = NA, aes(text = paste("Sample:", SampleName))) +
  geom_point(data = centroids, aes(x = V1, y = V2, fill=subgroups), show.legend=F, size = 5, shape = 21, color = "black") +  # Plot centroids
  #scale_color_manual(values = custom_colors) +  # Assign custom colors
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  labs(
    x = paste("MDS 1 (", variance_explained_all[1], "%)", sep = ""),
    y = paste("MDS 2 (", variance_explained_all[2], "%)", sep = ""),
    #color = "Group Test",
    title = paste("MDS (PCoA) with Jaccard Distance\nPERMANOVA p =", round(p_value, 4),"\npermutations = 999")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "black")
  )

ggplotly(p, tooltip = "text")  %>%
  layout(
    margin = list(t = 100),  # Increase the top margin to make space for the title
    legend = list(title = list(text = "Group Test"))  # Explicitly set legend title to avoid issues
  )



#######
#Permanova test between groups
permanova_result <- vegan::adonis2(dist_matrix_all ~ group_test, data = mds_df_all, permutations = 999)
p_value <- permanova_result$`Pr(>F)`[1]

centroids <- mds_df_all %>%
  group_by(group_test) %>%
  summarise(V1 = mean(V1), V2 = mean(V2))

p <- ggplot(mds_df_all, aes(x = V1, y = V2, fill = group_test)) +  # Replace with desired group_test column
  geom_point(size = 3, alpha = 0.5, shape = 21, color = NA, aes(text = paste("Sample:", SampleName))) +
  geom_point(data = centroids, aes(x = V1, y = V2, fill=group_test), show.legend=F, size = 5, shape = 21, color = "black") +  # Plot centroids
  #scale_color_manual(values = custom_colors) +  # Assign custom colors
  scale_fill_manual(values = custom_colors) +  # Assign custom colors
  labs(
    x = paste("MDS 1 (", variance_explained_all[1], "%)", sep = ""),
    y = paste("MDS 2 (", variance_explained_all[2], "%)", sep = ""),
    #color = "Group Test",
    title = paste("MDS (PCoA) with Jaccard Distance\nPERMANOVA p =", round(p_value, 4),"\npermutations = 999")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "black")
  )

ggplotly(p, tooltip = "text")  %>%
  layout(
    margin = list(t = 100),  # Increase the top margin to make space for the title
    legend = list(title = list(text = "Group Test"))  # Explicitly set legend title to avoid issues
  )



############################
permanova_result <- vegan::adonis2(dist_matrix_orr_dc ~ subgroup_dcr, data = mds_df_orr_dc, permutations = 999)
p_value <- permanova_result$`Pr(>F)`[1]

centroids <- mds_df_orr_dc %>%
  group_by(subgroup_dcr) %>%
  summarise(V1 = mean(V1), V2 = mean(V2))

p <- ggplot(mds_df_orr_dc, aes(x = V1, y = V2, fill = subgroup_dcr)) +  # Replace with desired group_test column
  geom_point(size = 3, alpha = 0.5, shape = 21, color = NA, aes(text = paste("Sample:", SampleName))) +
  geom_point(data = centroids, aes(x = V1, y = V2, fill=subgroup_dcr), show.legend=F, size = 5, shape = 21, color = "black") +  # Plot centroids
  #scale_color_manual(values = custom_colors) +  # Assign custom colors
  #scale_fill_manual(values = custom_colors) +  # Assign custom colors
  labs(
    x = paste("MDS 1 (", variance_explained_orr_dc[1], "%)", sep = ""),
    y = paste("MDS 2 (", variance_explained_orr_dc[2], "%)", sep = ""),
    #color = "Group Test",
    title = paste("MDS (PCoA) with Jaccard Distance\nPERMANOVA p =", round(p_value, 4),"\npermutations = 999")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", color = "black")
  )

ggplotly(p, tooltip = "text")  %>%
  layout(
    margin = list(t = 100),  # Increase the top margin to make space for the title
    legend = list(title = list(text = "Group Test"))  # Explicitly set legend title to avoid issues
  )







################################
# Define subgroups and mapping (converted from Python to R format)
SUBGROUPS_TO_INCLUDE <- c('all', 
                          #'is_ALIGENT', 'is_TWIST', 
                          'is_PNP', 'is_patho', 'is_probio', 'is_IgA',
                          'is_bac_flagella',   'is_infect',
                          #'is_EBV',
                          'is_IEDB_or_cntrl'#, 'is_toxin' 
                          #,'is_phage', 'is_allergens', 'is_influenza',
                          #'is_EM', 'signalp6_slow', 'is_topgraph_new_&_old', 'diamond_mmseqs_intersec_toxin',
                          #'is_IEDB_or_cntrl', 'is_pos_cntrl', 'is_neg_cntrl', 'is_rand_cntrl'
)


SUBGROUPS_TO_NAME <- c(
  'all' = 'Complete library',
  #'is_ALIGENT' = 'Aligent library', 'is_TWIST' = 'Twist library', 
  'is_PNP' = 'Metagenomics\nantigens',  'is_patho' = 'Pathogenic strains', 
  'is_probio' = 'Probiotic strains',  'is_IgA' = 'Antibody-coated\nstrains', 
   'is_bac_flagella' = 'Flagellins', 'is_infect' = 'Infectious\npathogens', 
  #'is_EBV' = 'Epstein-Barr\nVirus', 
  'is_IEDB_or_cntrl' = 'IEDB/controls'#, 'is_toxin' = 'Toxin'
  #,'is_phage' = 'Phages', 'is_allergens' = 'Allergens', 'is_influenza' = 'Influenza',
  #'is_EM' = 'Microbiota\ngenes', 'signalp6_slow' = 'Secreted proteins', 
  #'is_topgraph_new_&_old' = 'Membrane proteins','diamond_mmseqs_intersec_toxin' = 'Predicted toxins',
  #'is_IEDB_or_cntrl' = 'IEDB/controls', 'is_pos_cntrl' = 'Positive control', 'is_neg_cntrl' = 'Negative control', 'is_rand_cntrl' = 'Random control'
)

SUBGROUPS_ORDER <- c('Complete library', 
                     'Metagenomics\nantigens', 'Pathogenic strains', 'Probiotic strains',
                     'Antibody-coated\nstrains',  'Flagellins', 'Infectious\npathogens',
                     'IEDB/controls'#, 'Toxin'
                     #,'Phages', 'Allergens', 'Influenza',
                     #'Microbiota\ngenes', 'Secreted proteins', 'Membrane proteins', 'Predicted toxins',
                     #'IEDB/controls', 'Positive control', 'Negative control', 'Random control'
)


tmp <- as.data.frame(t(exist)) %>%
  tibble::rownames_to_column("SampleName")%>%
  dplyr::left_join(metadata %>%
                     dplyr::select(SampleName,  group_test, subgroups, subgroup_orr, subgroup_dcr)
                   ,by = "SampleName")

actual_subgroup_columns <- setdiff(SUBGROUPS_TO_INCLUDE, "all")
lib_metadata <- as.data.frame(reticulate::py_load_object(LIB_METADATA, pickle = "pickle")) %>%
  tibble::rownames_to_column(var = "Peptide") %>%
  mutate(across(all_of(actual_subgroup_columns), ~ ifelse(is.na(.), FALSE, . == 1 | . == TRUE))) %>%
  mutate(across(all_of(actual_subgroup_columns), ~ case_when(
    . == "True" ~ TRUE,
    . == "False" ~ FALSE,
    is.na(.) ~ FALSE,
    TRUE ~ as.logical(.)
  ))) %>%
  dplyr::select(c('Peptide', all_of(actual_subgroup_columns)))



# Prepare data for boxplots for each subgroup, including the "all" case
#####
df_grouped <- tmp %>%
  dplyr::select(-SampleName) %>% 
  dplyr::group_by(group_test) %>%
  dplyr::summarise(across(where(is.numeric), ~ sum(.x, na.rm = TRUE)))

lib_metadata['all'] <- TRUE
peptides_long <- lib_metadata %>%
  # If needed, first filter to the peptides that are in your presence/absence matrix:
  # filter(Peptide %in% colnames(presence_data))
  tidyr::pivot_longer(cols = -Peptide, names_to = "Category", values_to = "InCategory") %>%
  dplyr::filter(InCategory == TRUE)  # keep only those peptide-category pairs where the peptide is in that category


df_mat <- df_grouped %>% 
  tibble::remove_rownames() %>% 
  tibble::column_to_rownames(var = "group_test") %>% 
  as.matrix()


# Get the class names (e.g., "Controls", "HCC-ICI", "HCC-TKI")
classes <- rownames(df_mat)
# Create all pairwise combinations of classes
# This returns a list of character vectors, each of length 2.
combos <- combn(classes, 2, simplify = FALSE)

# For each pair, compute the ratio for every peptide:
ratio_results <- lapply(combos, function(pair) {
  # Extract counts for the two classes in the pair
  class1_counts <- df_mat[pair[1], ]
  class2_counts <- df_mat[pair[2], ]
  
  # Identify peptides with non-zero counts in both classes
  valid_idx <- (class1_counts > 4) & (class2_counts > 4)
  
  # Calculate the ratio only for those peptides
  ratios <- log(class2_counts[valid_idx] / class1_counts[valid_idx])
  
  tibble(
    Comparison = paste(pair, collapse = "_vs_"),
    Peptide = colnames(df_mat)[valid_idx],
    ratio = as.vector(ratios)
  )
})
ratios_df <- bind_rows(ratio_results)

plot_data <- peptides_long %>%
  dplyr::mutate(Category = factor(Category, levels = names(SUBGROUPS_TO_NAME), labels = SUBGROUPS_TO_NAME)) %>%
  dplyr::mutate(Category = factor(Category, levels = SUBGROUPS_ORDER)) %>% 
  dplyr::left_join(ratios_df, by = "Peptide") %>%
  dplyr::filter(Comparison == "Controls_vs_HCC") 
pairwise_comparisons <- combn(levels(plot_data$Category), 2, simplify = FALSE)

# Use compare_means() to calculate pairwise comparisons and adjust p-values
all_comparisons <- ggpubr::compare_means(
  formula = ratio ~ Category,
  data = plot_data,
  method = "wilcox.test",
  comparisons = pairwise_comparisons,
  p.adjust.method = "BH"  # Adjust using Benjamini-Hochberg
)
# Filter for significant comparisons (e.g., p.adj < 0.05)
sig_comparisons <- all_comparisons %>%
  filter(p.adj < 0.001)
sig_pairs <- lapply(1:nrow(sig_comparisons), function(i) {
  c(sig_comparisons$group1[i], sig_comparisons$group2[i])
})
ggplot(plot_data, aes(x = Category, y = ratio, fill = Category)) +
  geom_boxplot() +#outlier.shape = NA) +             # boxplot without outliers
  geom_jitter(width = 0.2, alpha = 0.05, size = 2) +  # overlay jittered points
  labs(
    x = "Peptide Category",
    y = "log Ratio of antibody responses\nin HCC and Controls") +
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = sig_pairs, 
                             p.adjust.method = "BH",  # Adjust using Benjamini-Hochberg
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

ggsave("plots//subggroup_controlsHCC.png", height = 4.5, width=6, dpi = 300)
 
####
plot_data <- peptides_long %>%
   dplyr::mutate(Category = factor(Category, levels = names(SUBGROUPS_TO_NAME), labels = SUBGROUPS_TO_NAME)) %>%
   dplyr::mutate(Category = factor(Category, levels = SUBGROUPS_ORDER)) %>% 
   dplyr::left_join(ratios_df, by = "Peptide") %>%
   dplyr::filter(Comparison == "Controls_vs_HCC-TKI") 

# Use compare_means() to calculate pairwise comparisons and adjust p-values
all_comparisons <- ggpubr::compare_means(
  formula = ratio ~ Category,
  data = plot_data,
  method = "wilcox.test",
  comparisons = pairwise_comparisons,
  p.adjust.method = "BH"  # Adjust using Benjamini-Hochberg
)
# Filter for significant comparisons (e.g., p.adj < 0.05)
sig_comparisons <- all_comparisons %>%
  filter(p.adj < 0.001)
sig_pairs <- lapply(1:nrow(sig_comparisons), function(i) {
  c(sig_comparisons$group1[i], sig_comparisons$group2[i])
})
ggplot(plot_data, aes(x = Category, y = ratio, fill = Category)) +
  geom_boxplot() +#outlier.shape = NA) +             # boxplot without outliers
  geom_jitter(width = 0.2, alpha = 0.05, size = 2) +  # overlay jittered points
  labs(
    x = "Peptide Category",
    y = "log Ratio of antibody responses\nin HCC-TKI and Healthy Controls") +
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = sig_pairs, 
                             p.adjust.method = "BH",  # Adjust using Benjamini-Hochberg
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")
 


plot_data <- peptides_long %>%
   dplyr::mutate(Category = factor(Category, levels = names(SUBGROUPS_TO_NAME), labels = SUBGROUPS_TO_NAME)) %>%
   dplyr::mutate(Category = factor(Category, levels = SUBGROUPS_ORDER)) %>% 
   dplyr::left_join(ratios_df, by = "Peptide") %>%
   dplyr::filter(Comparison == "HCC-ICI_vs_HCC-TKI") 
pairwise_comparisons <- combn(levels(plot_data$Category), 2, simplify = FALSE)
# Use compare_means() to calculate pairwise comparisons and adjust p-values
all_comparisons <- ggpubr::compare_means(
  formula = ratio ~ Category,
  data = plot_data,
  method = "wilcox.test",
  comparisons = pairwise_comparisons,
  p.adjust.method = "BH"  # Adjust using Benjamini-Hochberg
)
# Filter for significant comparisons (e.g., p.adj < 0.05)
sig_comparisons <- all_comparisons %>%
  filter(p.adj < 0.001)
sig_pairs <- lapply(1:nrow(sig_comparisons), function(i) {
  c(sig_comparisons$group1[i], sig_comparisons$group2[i])
})
ggplot(plot_data, aes(x = Category, y = ratio, fill = Category)) +
   geom_boxplot() +#outlier.shape = NA) +             # boxplot without outliers
   geom_jitter(width = 0.2, alpha = 0.05, size = 2) +  # overlay jittered points
   labs(
     x = "Peptide Category",
     y = "log Ratio of antibody responses\nin HCC TKI and ICI") +
  ggpubr::stat_compare_means(method = "wilcox.test", 
                              comparisons = sig_pairs, 
                              p.adjust.method = "BH",  # Adjust using Benjamini-Hochberg
                              label = "p.signif",  # Display significance level (e.g., * or **)
                              hide.ns = FALSE,      # Hide non-significant comparisons
                              size = 4)  +
   theme_bw() +
   theme(axis.text.x = element_text(angle = 45, hjust = 1),
         legend.position = "none")


###
combined_HCC <- df_mat["HCC-ICI", ] + df_mat["HCC-TKI", ]
df_mat_combined <- rbind(df_mat, HCC = combined_HCC)
# Identify peptides where both Controls and HCC have counts > 1 (or > 0, as needed)
valid_idx <- (df_mat_combined["Controls", ] > 3) & (df_mat_combined["HCC", ] > 3)
ratios <- log((df_mat_combined["HCC", valid_idx]) / (df_mat_combined["Controls", valid_idx]))
# Create a tibble with peptide names and the computed ratio
ratio_df <- tibble(
  Comparison = "Controls_vs_HCC",
  Peptide = colnames(df_mat_combined)[valid_idx],
  ratio = as.vector(ratios)
)
plot_data <- peptides_long %>%
  dplyr::mutate(Category = factor(Category, levels = names(SUBGROUPS_TO_NAME), labels = SUBGROUPS_TO_NAME)) %>%
  dplyr::mutate(Category = factor(Category, levels = SUBGROUPS_ORDER)) %>% 
  dplyr::left_join(ratio_df, by = "Peptide")

pairwise_comparisons <- combn(levels(plot_data$Category), 2, simplify = FALSE)
# Use compare_means() to calculate pairwise comparisons and adjust p-values
all_comparisons <- ggpubr::compare_means(
  formula = ratio ~ Category,
  data = plot_data,
  method = "wilcox.test",
  comparisons = pairwise_comparisons,
  p.adjust.method = "BH"  # Adjust using Benjamini-Hochberg
)
# Filter for significant comparisons (e.g., p.adj < 0.05)
sig_comparisons <- all_comparisons %>%
  filter(p.adj < 0.01)
sig_pairs <- lapply(1:nrow(sig_comparisons), function(i) {
  c(sig_comparisons$group1[i], sig_comparisons$group2[i])
})
ggplot(plot_data, aes(x = Category, y = ratio, fill = Category)) +
  geom_boxplot() +#outlier.shape = NA) +             # boxplot without outliers
  geom_jitter(width = 0.2, alpha = 0.05, size = 2) +  # overlay jittered points
  labs(
    x = "Peptide Category",
    y = "log Ratio of antibody responses\nin HCC and Healthy Controls") +
  ggpubr::stat_compare_means(method = "wilcox.test", 
                             comparisons = sig_pairs, 
                             p.adjust.method = "BH",  # Adjust using Benjamini-Hochberg
                             label = "p.signif",  # Display significance level (e.g., * or **)
                             hide.ns = FALSE,      # Hide non-significant comparisons
                             size = 4)  +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")
