library("stats")  # For chi-squared test
library("multcomp")  # For multiple testing correction
library("dplyr")
library("tidyr")
library("ggplot2")
library("scales")
library("stringr")
library("ggsignif")
library("plotly")
library("nnet")
library("patchwork")
library("ggpubr")
library("rstatix")
library("ggvenn")
library("ggmsa")
library("RColorBrewer")
library("openxlsx")

# Global vars

SUBGROUPS_TO_INCLUDE <- c(#'all', 
  'is_patho', 'is_probio', 'is_MPA', 'is_IgA','is_PNP', 
    'is_infect', 'is_flagellum', "is_allergens" 
  #'is_IEDB_or_cntrl'
  )
SUBGROUPS_ORDER <- c(#'Complete library', 
 'Pathogenic strains', 'Probiotic strains', 'Microbiota strains',
  'Antibody-coated strains',   'Metagenome antigens', 'Infectious pathogens', 'Flagellins', 
 "Allergens"# 'IEDB/controls'
 )
SUBGROUPS_TO_NAME <- c(
  #'all' = 'Complete library',
  'is_patho' = 'Pathogenic strains', 
  'is_probio' = 'Probiotic strains', 'is_MPA' = 'Microbiota strains', 'is_IgA' = 'Antibody-coated strains', 
    'is_PNP' = 'Metagenome antigens', 'is_infect' = 'Infectious pathogens', 'is_flagellum' = 'Flagellins', 
  "is_allergens" = "Allergens"
  #'is_IEDB_or_cntrl' = 'IEDB/controls'
  )

#######################################################
################ Helpers ##############################
#######################################################
# add_flag_by_patterns <- function(df,
#                                  new_flag,
#                                  patterns,
#                                  target_cols = c("Organism_complete_name", "Description")) {
#   # 1) Build a single case‐insensitive regex from all patterns:
#   regex_str <- paste0("(?i)", paste(patterns, collapse = "|"))
#   
#   # 2) Mutate a new logical column.  We use if_any(all_of(target_cols), ~ str_detect(...)):
#   df %>%
#     mutate(
#       !!new_flag := if_any(
#         all_of(target_cols),
#         ~ str_detect(.x, regex(regex_str))
#       )
#     )
# }
# ---- Format p-values nicely ----
format_pval <- function(p, alpha = 0.05) {
  
  # helper to drop trailing zeros (e.g. "1.00" → "1", "0.500" → "0.5")
  drop_zeros <- function(x) sub("\\.?0+$", "", x)
  
  if (is.na(p)) return("NA")
  
  # -------------------------------------------------------------------------
  # 1. Non-significant (p > alpha)
  # -------------------------------------------------------------------------
  if (p > alpha) {
    raw <- formatC(p, digits = 2, format = "f")
    raw <- drop_zeros(raw)
    return(paste0("ns [", raw, "]"))
  }
  
  # -------------------------------------------------------------------------
  # 2. Normal fixed-decimal formatting (0.001 ≤ p ≤ alpha)
  # -------------------------------------------------------------------------
  if (p >= 0.001) {
    raw <- formatC(p, digits = 3, format = "f")
    raw <- drop_zeros(raw)
    return(raw)
  }
  
  # -------------------------------------------------------------------------
  # 3. Scientific notation (< 0.001)
  # -------------------------------------------------------------------------
  raw <- formatC(p, digits = 2, format = "e")
  
  # remove unnecessary zeros: "1.00e-05" → "1e-05"
  raw <- sub("([0-9]+)\\.0+e", "\\1e", raw)
  
  # remove trailing zeros inside "1.10e-04" → "1.1e-04"
  raw <- sub("([0-9]+\\.[0-9]*[1-9])0+e", "\\1e", raw)
  
  return(raw)
}



add_flag_by_patterns <- function(df,
                                 new_flag,
                                 patterns,
                                 target_cols = c("species", "Description")) {
  # build our two regexes:
  regex_sub   <- paste0("(?i)", paste(patterns, collapse = "|"))
  regex_exact <- paste0("(?i)^(", paste(patterns, collapse = "|"), ")$")
  
  # split out any Peptide column
  pep_present  <- "Peptide" %in% target_cols
  nonpep_cols  <- setdiff(target_cols, "Peptide")
  
  df %>%
    mutate(
      # temporary flags
      sub_flag = if (length(nonpep_cols) > 0) {
        if_any(all_of(nonpep_cols), ~ str_detect(.x, regex(regex_sub)))
      } else {
        # no non-Peptide columns requested
        FALSE
      },
      pep_flag = if (pep_present) {
        # exact match only on the Peptide column
        str_detect(.data$Peptide, regex(regex_exact))
      } else {
        FALSE
      },
      # final flag is TRUE if either test passes
      !!new_flag := sub_flag | pep_flag
    ) %>%
    # clean up
    select(-sub_flag, -pep_flag)
}

# add_exact_flag <- function(df,
#                            new_flag,
#                            patterns,
#                            target_cols = c("species", "Description")) {
#   # build an anchored regex: ^(A|B|C)$
#   regex_exact <- paste0("^(", paste(patterns, collapse="|"), ")$")
#   
#   df %>%
#     mutate(
#       # across each target col, test for an exact match
#       !!new_flag := if_any(all_of(target_cols),
#                            ~ stringr::str_detect(.x, regex_exact))
#     )
# }

add_exact_flag <- function(df,
                           new_flag,
                           patterns,
                           target_cols = c("species", "Description")) {
  
  # build an anchored regex: ^(A|B|C)$
  regex_exact <- paste0("^(", paste(patterns, collapse="|"), ")$")
  
  df %>%
    mutate(
      # for each target column, str_detect → NA when x is NA,
      # but replace those NAs with FALSE before the any()
      !!new_flag := if_any(
        all_of(target_cols),
        ~ replace_na(str_detect(.x, regex_exact), FALSE)
      )
    )
}




###############get taxa_vs_rest with pvals adjusted
#' Creates a custom lineage column based on flexible rules.
#'
#' @param comparison_df The data frame containing the lineage columns (e.g., Class, Genus, Species).
#' @param default_lineage_col The lineage column to use as the default for all
#'   taxa not matched by the rules (e.g., "Species").
#' @param custom_rules A data frame or list of lists defining the custom groups.
#'   Each row/list must have:
#'   - 'lineage_col': The column name to check (e.g., "Class", "Genus").
#'   - 'taxa_name': The specific name to match (e.g., "Bacteroidales", "Streptococcus").
#' @param new_col_name The name for the resulting custom column (default: "Custom_Taxa").
#' @return The data frame with the new custom lineage column added.
create_flexible_taxa_column <- function(
    df,
    default_lineage_col,
    custom_rules,
    new_col_name = "custom_taxa"
) {
  # --- 1. VALIDATION ---
  
  # Check if the default column exists
  if (!(default_lineage_col %in% names(df))) {
    stop("Default lineage column '", default_lineage_col, "' not found in data frame.")
  }
  
  # Ensure custom_rules is a data frame for easier iteration
  if (!is.data.frame(custom_rules)) {
    custom_rules <- data.frame(custom_rules)
  }
  
  # Check if all columns specified in custom_rules exist
  required_cols <- unique(custom_rules$lineage_col)
  missing_cols <- required_cols[!(required_cols %in% names(df))]
  if (length(missing_cols) > 0) {
    stop("Lineage columns specified in rules are missing: ", paste(missing_cols, collapse = ", "))
  }
  
  # --- 2. BUILD MUTATE LOGIC (using case_when) ---
  
  # Initialize the case_when statement with the TRUE (default) condition
  # We use !!sym(default_lineage_col) to dynamically select the column
  case_conditions <- list(quo(TRUE ~ !!rlang::sym(default_lineage_col)))
  
  # Build the custom rules (Rule 1, Rule 2, etc. - in order)
  for (i in seq_len(nrow(custom_rules))) {
    rule <- custom_rules[i, ]
    
    lineage_sym <- sym(rule$lineage_col)
    taxa_name_val <- rule$taxa_name
    
    # Create the condition: e.g., Class == "Bacteroidales" ~ "Bacteroidales"
    condition <- quo((!!lineage_sym == !!taxa_name_val) ~ !!taxa_name_val)
    case_conditions <- c(list(condition), case_conditions) # Prepend to apply rules first
  }
  
  # --- 3. APPLY MUTATION ---
  
  # Use !!! to splice the list of quoted expressions into the case_when function
  df <- df %>%
    mutate(
      !!sym(new_col_name) := case_when(!!!case_conditions)
    )
  
  return(df)
}


get_top_significant_taxa_df <- function(comparison_df, lineage_col){#, n = 100) {
  # turn the string into a symbol once
  lineage_sym <- sym(lineage_col)
  
  top_taxa_df <- comparison_df %>%
    # drop rows where that lineage is missing
    filter(!is.na(!!lineage_sym)) %>%
    # make sure we have the same 'ratio' name
    mutate(ratio = log2(ratio)) %>%
    # get one row per lineage
    distinct(!!lineage_sym) %>%
    # for each lineage, pull out x = that group, y = all others
    mutate(
      x = purrr::map(!!lineage_sym, ~ comparison_df$ratio[comparison_df[[lineage_col]] == .x]),
      y = purrr::map(!!lineage_sym, ~ comparison_df$ratio[comparison_df[[lineage_col]] != .x]),
      p = purrr::map2_dbl(x, y, ~ wilcox.test(.x, .y)$p.value)
    ) %>%
    select(!!lineage_sym, p) %>%
    # FDR‐adjust
    adjust_pvalue(method = "BH") %>%
    arrange(p.adj) %>%
    # take the top n
    #slice_head(n = n) %>%
    # ---- CLEAN THE NAMES HERE ----
  # Modify the lineage column in place
  mutate(
    !!lineage_sym := str_remove_all(!!lineage_sym, "\\[|\\]"), # drop any brackets
    !!lineage_sym := str_squish(!!lineage_sym)               # collapse multiple spaces, trim
  ) %>%
    # Select the cleaned lineage and the adjusted p-value
    select(!!lineage_sym, p, p.adj)
  
  return(top_taxa_df)
}



make_flag_lists <- function(
    df,
    n = 10,
    taxa_labels = NULL, # New: Vector of taxa names to select (e.g., c("species1", "species2"))
    colors = NULL,
    brewer_palette = "Set3",
    brewer_n = 12
) {
  # Assuming the taxa column is the first one
  taxa_col_name <- names(df)[1]
  
  # Filter the data frame based on the provided labels
  if (!is.null(taxa_labels)){
    df_filtered <- df %>%
      filter(!!sym(taxa_col_name) %in% taxa_labels)
  } else {
    df_filtered <- head(df, n)
  }
  
  # Check if any of the labels were found
  top_names <- df_filtered[[taxa_col_name]]
  
  if (length(top_names) == 0) {
    stop("No specified taxa found in the first column of the input data frame.")
  }
  
  # derive short keys from first three words
  short_keys <- sapply(top_names, function(x) {
    words <- strsplit(x, "\\s+")[[1]]
    paste(head(words, 3), collapse = " ")
  })
  # ensure unique
  short_keys <- make.unique(short_keys)
  
  # build flags_to_patterns list: key = short_key, value = full name
  flags_to_patterns <- setNames(as.list(top_names), short_keys)
  n_taxa <- length(top_names)
  
  # build highlight_colors vector
  if (!is.null(colors)) {
    if (length(colors) != n_taxa) {
      stop("Length of 'colors' (", length(colors), ") must match number of flags (", n_taxa, ")")
    }
    # Use the provided colors, named by the new short keys
    highlight_colors <- setNames(colors, short_keys)
  } else {
    pal <- colorRampPalette(brewer.pal(brewer_n, brewer_palette))(n_taxa)
    highlight_colors <- setNames(pal, short_keys)
  }
  
  # return
  list(
    flags_to_patterns = flags_to_patterns,
    highlight_colors  = highlight_colors,
    p = df_filtered$p,
    p.adj = df_filtered$p.adj
  )
}


######################################################
############## Count Distribution ####################
######################################################
get_count_percentage_df <- function(features_target, group_col, group_cols, prevalence_threshold = 0,
                                    lib_metadata_df = NULL) {
  df <- features_target %>%
    # only samples that actually have a group
    filter(!is.na(.data[[group_col]])) %>%
    # gather 0/1 peptide calls
    pivot_longer(
      cols     = -c(SampleName, any_of(group_cols)),
      names_to = "Peptide",
      values_to= "Presence"
    ) %>%
    # compute both raw count and percent in one summarise
    group_by(Peptide, Group = .data[[group_col]]) %>%
    summarise(
      count   = sum(Presence, na.rm = TRUE),
      Percent = 100 * count / n(),
      .groups = "drop"
    ) %>%
    # spread into one column per group for Percent and one per group for Count
    pivot_wider(
      names_from  = Group,
      values_from = c(count, Percent),
      names_glue  = "{Group}_{.value}"
    )
  # bring back peptide metadata
  if (!is.null(lib_metadata_df)) {      
    df <- df %>% 
      left_join(
        lib_metadata_df %>%
          tibble::rownames_to_column("Peptide") %>%
          select(Peptide, Description, class, order, family, genus, species),
        by = "Peptide"
      )
  }
  df <- df %>%
    # drop peptides that never hit prevalence threshold in any group
    filter(if_any(ends_with("_Percent"),~ . >= prevalence_threshold)) %>% 
    # strip off the "_Percent" suffix
    rename_with(
      ~ sub("_Percent$", "", .x),
      ends_with("_Percent")
    ) 
  return(df)
}


plot_enrichment_counts <- function(features_target,
                                   group_col, group_cols,
                                   prevalence_threshold = 0,
                                   custom_colors,
                                   binwidth = 1) {
  # get the count/percent table
  percentage_df <- get_count_percentage_df(
    features_target      = features_target,
    group_col            = group_col,
    group_cols           = group_cols, 
    prevalence_threshold = prevalence_threshold
  )
  
  real_order <- features_target %>%
    pull(!!sym(group_col)) %>%
    factor() %>%
    levels()
  
  
  # pivot to long of the *_count columns
  count_df <- percentage_df %>%
    select(ends_with("_count")) %>%
    pivot_longer(
      cols      = everything(),
      names_to  = "Cohort",
      values_to = "n_present"
    ) %>%
    mutate(Cohort = sub("_count$", "", Cohort),
           Cohort = factor(Cohort, levels = real_order)) %>%
    filter(n_present > 0)   # drop zero‐present peptides 
  
  # compute thresholds per cohort
  thresholds <- count_df %>%
    group_by(Cohort) %>%
    summarise(
      n_samples   = max(n_present),
      thresh      = ceiling(n_samples * 0.05),
      n_peptides5 = sum(n_present >= thresh),
      .groups     = "drop"
    )
  
  # build the plot
  p <- ggplot(count_df, aes(x = n_present, fill = Cohort)) +
    geom_histogram(
      binwidth = binwidth,
      position = "identity",
      alpha    = 0.9,
    ) +
    scale_y_log10(
      breaks = 10^(0:6),                                      # 10^0,10^1,…,10^6
      labels = trans_format("log10", math_format(10^.x)),     # render as 10^x
      expand = expansion(mult = c(0, .15))                    # a little space above
    ) +
    annotation_logticks(sides = "l", scaled = TRUE) +
    scale_fill_manual(values = custom_colors) +
    labs(
      x = "# of individuals",
      y = expression("# of significantly bound peptides (" * log[10] * ")")
    )+
    
    # horizontal arrowed line
    geom_segment(
      data        = thresholds,
      aes(
        x    = thresh,
        xend = n_samples,
        y    = n_peptides5,
        yend = n_peptides5
      ),
      inherit.aes = FALSE,
      linetype    = "dashed",
      color       = "black",
      size        = 0.4,
      arrow       = arrow(length = unit(0.1, "cm"), ends = "both")
    ) +
    # centred label above line
    geom_text(
      data        = thresholds,
      aes(
        x     = (thresh + n_samples) / 2,
        y     = n_peptides5,
        label = paste0(n_peptides5, " peptides in ≥5%")
      ),
      inherit.aes = FALSE,
      vjust       = -0.5,
      size        = 4
    ) +
    
    facet_wrap(~ Cohort, ncol = 2, scales = "free_x") +
    theme_bw(base_size = 13) +
    theme(
      legend.position   = "none",
      strip.background  = element_blank(),
      strip.text = element_text(face = "bold", size = 14, colour = "black"),
      panel.grid.major = element_line(color = "grey90", linetype = "solid"),
      panel.grid.minor  = element_blank(),
      axis.text.y.left = element_text(size = 13),
      axis.text.x.bottom = element_text(size = 13),
      axis.title.y = element_text(size = 14),
      axis.title.x = element_text(size = 14, margin = margin(t = 0)),
      plot.margin = margin(0, 3, 0, 0, unit = "pt")    
      )
  
  return(p)
}

######################################################
####### plot  enrichment and diversity################
######################################################
# plot_groups_boxplots <- function(data, group_col, values_col, custom_colors, pairwise_comparisons, 
#                                  label_axis = NA, sig_level   = 0.05,
#                                  #max_sig       = 6,
#                                  label_format  = "p.format"){  # or "p.signif") 
#   # Convert grouping column name to symbol
#   group_sym <- sym(group_col)
#   values_sym <- sym(values_col)
#   
#   # Drop rows where either group or value is NA
#   data <- data %>%
#     filter(!is.na(!!group_sym), !is.na(!!values_sym))
#   
#   # Summarize counts by that group
#   df_counts <- data %>%
#     group_by(!!group_sym) %>%
#     summarize(sample_count = n(), .groups = "drop")
#   
#   # Create x-axis labels using the dynamic variable.
#   # Since group_col is a string, we access it directly on df_counts.
#   x_labels <- setNames(
#     paste0(df_counts[[group_col]], "\n(n = ", df_counts$sample_count, ")"),
#     df_counts[[group_col]]
#   )
#   
#   # — compute pairwise p-values and keep only those < sig_level —
#   sig_comparisons <- purrr::keep(pairwise_comparisons, function(pair) {
#     g1 <- pair[1]
#     g2 <- pair[2]
#     # extract the raw values for each group
#     x <- data %>% filter(!!group_sym == g1) %>% pull(!!values_sym)
#     y <- data %>% filter(!!group_sym == g2) %>% pull(!!values_sym)
#     wt <- wilcox.test(x, y, exact = FALSE)
#     wt$p.value < sig_level
#   })
#   
#   # pairwise_p <-  purrr::map_dfr(pairwise_comparisons, function(pair) {
#   #   g1 <- pair[1]; g2 <- pair[2]
#   #   x  <- data %>% filter(!!group_sym == g1) %>% pull(!!values_sym)
#   #   y  <- data %>% filter(!!group_sym == g2) %>% pull(!!values_sym)
#   #   p  <- wilcox.test(x, y, exact = FALSE)$p.value
#   #   tibble::tibble(group1 = g1, group2 = g2, p.value = p)
#   # })
#   # 
#   # # 2) keep only p < sig_level, sort, take top max_sig
#   # top_pairs <- pairwise_p %>%
#   #   filter(p.value < sig_level) %>%
#   #   arrange(p.value) %>%
#   #   slice_head(n = max_sig)
#   # 
#   # # 3) turn that back into the list-of-pairs format
#   # sig_comparisons <- purrr::pmap(
#   #   list(top_pairs$group1, top_pairs$group2),
#   #   c
#   # )  
#   # Build the plot:
#   p <- ggplot(data, aes(x = !!group_sym, y = !!values_sym)) +
#     geom_boxplot(show.legend = FALSE, outlier.shape = NA, aes(fill = !!group_sym)) +
#     #geom_boxplot(show.legend = FALSE, outlier.shape = NA, fill="gray80") +  # Hide legend if desired
#     #geom_jitter(show.legend = FALSE, aes(color = !!group_sym), width = 0.2, alpha = 0.7, size = 1) +
#     geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = FALSE) +  
#     scale_fill_manual(values = custom_colors) +  # Assign custom colors
#     #scale_colour_manual(values = custom_colors) +  # Assign custom colors
#     
#     scale_x_discrete(labels = x_labels) +         # Use the custom labels
#     theme_bw(base_size = 13) +
#     theme(
#       axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),
#       axis.text.x.bottom = element_text(size = 13),
#       axis.text.y.left = element_text(size = 13),
#       axis.title.y = element_text(size = 13),
#       axis.title.x = element_text(size = 13),
#       plot.margin = margin(0, 1, 0, 1, unit = "pt"),
#       panel.grid = element_blank()
#     ) +
#     scale_y_continuous(expand = expansion(mult = c(0, 0.1))) 
#   # ggpubr::stat_compare_means(method = "wilcox.test", 
#   #                            comparisons = pairwise_comparisons, 
#   #                            label = "p.format",#"p.signif",  # Display significance level (e.g., * or **)
#   #                            hide.ns = T,     # Option to hide non-significant comparisons
#   #                            size = 4)
#   
# 
# 
#   # only add stat_compare_means if there are any significant comparisons
#   if (length(sig_comparisons) > 0) {
#     p <- p +
#       stat_compare_means(
#         method       = "wilcox.test",
#         comparisons  = sig_comparisons,
#         label        = label_format,
#         hide.ns      = F,    # hides any “ns”
#         size         = 4.5, #4.5AA if numbres, 5 if ns, 7 for peptide level -  paper
#         tip.length   = 0.02
#       )
#   }
#   
#   # If label_axis is not NA, add custom axis labels. Assuming label_axis is a vector of length 2:
#   # if (is.na(label_axis[1])) {
#   #   p <- p + labs(
#   #     x = NULL
#   #   )
#   #   if(!is.na(label_axis[2])){
#   #     p <- p + labs(
#   #       y = label_axis[2]
#   #     )
#   #   }
#   # } else if (!is.na(label_axis[1]) && !is.na(label_axis[2])) {
#   #   p <- p + labs(
#   #     x = label_axis[1],
#   #     y = label_axis[2]
#   #   )
#   # }
# 
#   # Build a named list for labs(), turning NA → NULL:
#   labs_args <- list(
#     x = if (is.na(label_axis[1])) NULL else label_axis[1],
#     y = if (is.na(label_axis[2])) NULL else label_axis[2]
#   )
#   # Always call labs() so that x=NULL actually removes the default
#   p <- p + do.call(labs, labs_args)
#   
#   return(p)
# }


######################################################
####### plot  enrichment and diversity################
######################################################
plot_groups_boxplots <- function(data, group_col, values_col, custom_colors,
                                 label_axis = NA, sig_level   = 0.05,
                                 #max_sig       = 6,
                                 label_format  = "p.format"){  # or "p.signif") 
  
  # Convert grouping column name to symbol
  group_sym <- sym(group_col)
  values_sym <- sym(values_col)
  
  # Drop rows where either group or value is NA
  data <- data %>%
    filter(!is.na(!!group_sym), !is.na(!!values_sym))
  
  # Summarize counts by that group
  df_counts <- data %>%
    group_by(!!group_sym) %>%
    summarize(sample_count = n(), .groups = "drop")
  
  # Create x-axis labels using the dynamic variable.
  # Since group_col is a string, we access it directly on df_counts.
  x_labels <- setNames(
    paste0(df_counts[[group_col]], "\n(n = ", df_counts$sample_count, ")"),
    df_counts[[group_col]]
  )
  
  ## ---- SIGNIFICANCE TESTING SECTION ----
  n_groups <- n_distinct(data[[group_col]])
  
  if (n_groups == 2) {
    # --- Wilcoxon test ---
    pvals_df <- data %>%
      rstatix::pairwise_wilcox_test(formula = as.formula(paste0("`", values_col, "` ~ ", group_col)) ) %>%
      rstatix::add_xy_position(x = group_col) %>%
      dplyr::filter(p <= sig_level) %>% 
      dplyr::rename(p.signif = p.adj.signif) %>%
      mutate(p.format = sapply(p.adj, format_pval))
  } else if (n_groups > 2) {
    kw <- kruskal_test(data, formula = as.formula(paste0("`", values_col, "` ~ ", group_col)) )
    #if (kw$p <= sig_level) {
      pvals_df <- data %>%
        rstatix::dunn_test( formula = as.formula(paste0("`", values_col, "` ~ ", group_col)), p.adjust.method = "BH") %>%
        rstatix::add_xy_position(x = group_col) %>%
        dplyr::filter(p.adj <= sig_level) %>% 
        dplyr::rename(p.signif = p.adj.signif) %>%
        mutate(p.format = sapply(p.adj, format_pval))
    #}
  }
  
  
  
  ## ---- BUILD PLOT ----
  p <- ggplot(data, aes(x = !!group_sym, y = !!values_sym)) +
    geom_boxplot(show.legend = FALSE, outlier.shape = NA, aes(fill = !!group_sym)) +
    geom_jitter(color = "black", size = 1, width = 0.2, alpha = 0.3, show.legend = FALSE) +  
    scale_fill_manual(values = custom_colors) + 
    scale_x_discrete(labels = x_labels) +        
    theme_bw(base_size = 13) +
    theme(
      axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),
      axis.text.x.bottom = element_text(size = 13),
      axis.text.y.left = element_text(size = 13),
      axis.title.y = element_text(size = 13),
      axis.title.x = element_text(size = 13),
      plot.margin = margin(0, 1, 0, 1, unit = "pt"),
      panel.grid = element_blank()
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.1))) 
  
  if (nrow(pvals_df) > 0) {
    p <- p + stat_pvalue_manual(
      data          = pvals_df,
      label         = label_format,
      y.position    = "y.position",
      tip.length    = 0.02,
      bracket.size  = 0.25,
      size          = 4.5,
      inherit.aes   = FALSE,
      step.increase = 0.1,
      #bracket.nudge.y = 10
    )
  }
  
  labs_args <- list(
    x = if (is.na(label_axis[1])) NULL else label_axis[1],
    y = if (is.na(label_axis[2])) NULL else label_axis[2]
  )
  
  p <- p + do.call(labs, labs_args)
  
  return(p)
}


#################################
## plot sex/age distribution#####
#################################
test_sex_age_distribution <- function(data, 
                                      group_col, 
                                      age_col = "Age", 
                                      sex_reg = "Sex", 
                                      sex_ctg = "Sex_ctg") {
  # Chi-square test using the grouping variable (group_col) and the categorical sex column (sex_ctg)
  cat("Chi-square test result:\n")
  chisq_result <- chisq.test(table(data[[group_col]], data[[sex_ctg]]))
  print(chisq_result)
  
  # Multinomial regression: Predict the grouping variable based on Age and the numeric Sex variable.
  # The formula is constructed as: `group_col` ~ Age * Sex
  cat("\nMultinomial Regression Summary:\n")
  fmla_multinom <- as.formula(paste0("`", group_col, "` ~ ", age_col, " * ", sex_reg))
  multinom_model <- multinom(fmla_multinom, data = data)
  print(summary(multinom_model))
  
  # Two-way ANOVA: Predict Age based on group and the categorical sex variable.
  # Here the formula is: Age ~ `group_col` * Sex_ctg
  cat("\nANOVA Summary:\n")
  fmla_aov <- as.formula(paste0(age_col, " ~ `", group_col, "` * ", sex_ctg))
  aov_model <- aov(fmla_aov, data = data)
  tidy_aov <- broom::tidy(aov_model)
  tidy_aov$p.value <- round(tidy_aov$p.value, 3)
  
  # Display the ANOVA results in a neat table
  print(knitr::kable(tidy_aov, digits = 3, 
                     caption = "Two-way ANOVA Results for Age by Group and Sex"))
  
  # Optionally, return a list with the results:
  return(list(chisq = chisq_result,
              multinom_summary = summary(multinom_model),
              aov_results = tidy_aov))
}

plot_sex_age_distribution <- function(data, 
                                      group_col, 
                                      age_col = "Age_group", 
                                      sex_col = "Sex_ctg", 
                                      custom_colors) {
  
  # Convert the grouping column name to a symbol for tidy evaluation.
  group_sym <- sym(group_col)
  
  # Step 1: Summarize counts per combination of group, sex and age group.
  # Here we assume that you want to mirror counts so that Male counts become negative.
  data_summary <- data %>%
    # 0) drop any samples with missing Sex or missing Age or missing group
    filter(!is.na(!!group_sym), !is.na(.data[[sex_col]]), !is.na(.data[[age_col]])) %>%
    group_by(!!group_sym, .data[[sex_col]], .data[[age_col]]) %>%
    summarize(count = n(), .groups = "drop") %>%
    mutate(count = ifelse(.data[[sex_col]] == "Male", -count, count))
  
  
  # Step 2: Compute overall counts by group (for generating facet labels).
  df_counts <- data %>%
    filter(!is.na(!!group_sym), !is.na(.data[[sex_col]]), !is.na(.data[[age_col]])) %>%
    group_by(!!group_sym) %>%
    summarize(sample_count = n(), .groups = "drop")
  
  # Create x-axis labels (here, using the group variable value and overall sample count).
  x_labels <- setNames(
    paste0(df_counts[[group_col]], "\n(n = ", df_counts$sample_count, ")"),
    df_counts[[group_col]]
  )
  
  
  p <- ggplot(data_summary, aes(x = ifelse(!!group_sym == levels(factor(.data[[group_sym]]))[1], count, -count), y = .data[[age_col]], fill = .data[[sex_col]])) +
    geom_bar(stat = "identity", position = "identity", width = 0.9) +
    scale_x_continuous(
      labels = function(x) ifelse(x %in%  seq(-300, 300, by = 4), abs(x), ""),
      breaks = seq(-300, 300, by = 4)
    ) +
    scale_fill_manual(values = custom_colors) +
    # Facet by the chosen grouping variable.
    facet_grid(as.formula(paste0("~ `", group_col, "`")), scales = "free_x", space = "free_x", 
               labeller = as_labeller(x_labels)) +
    labs(
      x = "Counts",
      y = "Age Group",
      fill = NULL
    ) +
    theme_light(base_size = 16) +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 16),
      legend.box.margin   = margin(t = 0, r = 0, b = 0, l = 0), 
      legend.margin       = margin(t = 0, r = 0, b = 0, l = 0),
      strip.background.x = element_blank(),
      #strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 16, colour = "black"),
      #panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "grey90", linetype = "solid"),
      panel.grid.major.y = element_blank(),
      axis.text.y.left = element_text(size = 15),
      axis.text.y.right = element_blank(),
      axis.ticks.y.right = element_blank(),
      axis.text.x.bottom = element_text(size = 15),
      axis.title.x = element_text(size = 16),
      axis.title.y = element_text(size = 16),
      legend.box.spacing = unit(0.2, "lines"),
      plot.margin = margin(0, 6, 0, 0, unit = "pt")    
      
    )
  
  return(p)
}


####################################
##########Scatterplot###############
####################################
make_interactive_scatterplot <- function(comparison_df,
                                         group1, group2, N,
                                         reverse_legend = FALSE,
                                         highlight_cols   = NULL,
                                         highlight_colors = NULL,
                                         pvals_adj = NULL,
                                         legend_order =NULL,
                                         default_color    = "gray70",
                                         #multiple_color   = "black",
                                         significant_colors = c(
                                           "not significant"                 = "dodgerblue",
                                           "significant prior correction"    = "forestgreen",
                                           "significant post FDR correction" = "firebrick"),
                                         interactive = F) {
  # sanity-check:
  if (!is.null(highlight_cols)) {
    missing_cols <- setdiff(highlight_cols, names(comparison_df))
    if (length(missing_cols)) {
      stop("These highlight_cols are not in your data frame: ",
           paste(missing_cols, collapse = ", "))
    }
  }
  
  # build the collapsed factor ------------------------------------------------
  if (!is.null(highlight_cols) && length(highlight_cols) > 0) {
    comparison_df <- comparison_df %>%
      rowwise() %>%
      mutate(
        # collect the names of all TRUE flags in this row:
        .trues = list(highlight_cols[ c_across(all_of(highlight_cols)) ]),
        # now assign highlight:
        highlight = if (length(.trues) == 0) {
          "none"
        } else if (length(.trues) >= 1) {
          .trues[[1]]
        } #else {
        #"multiple"}
      ) %>%
      ungroup() %>%
      select(-.trues)
    
    comparison_df <- comparison_df %>% 
      mutate(highlight = replace_na(highlight, "none"))
    # ensure factor has all levels:
    levels_needed <- c("none",  highlight_cols) #multiple
    comparison_df$highlight <- factor(
      comparison_df$highlight,
      levels = levels_needed
    )
    # build tooltip (only for highlighted points)
    comparison_df <- comparison_df %>%
      mutate(
        log2ratio = log2( ratio ),
        tooltip_txt = if_else(
          highlight == "none",
          NA_character_,
          paste0(
            "Peptide: ",  Peptide,               "<br>",
            "Desc: ",     Description,           "<br>",
            "Species: ",  species,"<br>",
            group1, ": ", !!sym(group1), " / ",
            group2, ": ", !!sym(group2),       "<br>",
            "Highlight: ", highlight
          )
        )
      ) %>%
      filter(
        is.finite(.data[[group1]]),
        is.finite(.data[[group2]])
      ) %>%
      arrange(highlight)
    
    if (is.null(pvals_adj)){
      pvals <- sapply(highlight_cols, function(flag) {
        x <- comparison_df %>% filter( !!sym(flag) ) %>% pull(log2ratio)
        #y <- comparison_df$log2ratio
        # better subset y to only non-flag too:
        y <- comparison_df %>% filter(! (!!sym(flag)) ) %>% pull(log2ratio)
        w <- wilcox.test(x, y)
        w$p.value
      })
      pvals_adj <- p.adjust(pvals, method = "BH")
    }
    fmt_p <- sapply(pvals_adj, format_pval)
    legend_labels <- paste0(highlight_cols, " (p=", fmt_p, ")")
    names(legend_labels) <- highlight_cols
    
    if (is.null(highlight_colors)) {
      # pick a palette for the flags
      palette_vals <- setNames(
        RColorBrewer::brewer.pal(
          n = max(length(highlight_cols), 8),
          name = "Set2"
        )[1:length(highlight_cols)],
        highlight_cols
      )
    } else {
      palette_vals <- highlight_colors
    }
    manual_vals <- c(
      none     = default_color,
      #multiple = multiple_color,
      palette_vals
    )
    
    color_aes   <- aes(color = highlight, text = tooltip_txt)
    legend_breaks <- if (!is.null(legend_order)) legend_order else highlight_cols
    color_scale <- scale_color_manual(
      name   = NULL,
      values = manual_vals,
      breaks = legend_breaks,
      labels = legend_labels[legend_breaks] #legend_labels
    )
    legend_theme <- theme(
      legend.position     = c(-0.025, 1.015),   # 50% across, 95% up
      legend.justification = c(0, 1),  
      legend.background = element_blank(),     # remove any grey behind the whole legend
      legend.box.background = element_blank(),  # remove border around the legend box
      legend.key.size      = unit(16, "pt"),
      #legend.text          = element_text(size = 15),
      legend.title         = element_text(size = 15),
      legend.key = element_rect(fill = "transparent", colour = NA),
      legend.spacing.x  = unit(0.3, "cm"), # space between key and label
      legend.text       = element_text(size = 15, margin = margin(l = -2)),
      
      
      axis.text.y.left = element_text(size = 16),
      axis.text.x.bottom = element_text(size = 16),
      axis.title.y = element_text(size = 17),
      axis.title.x = element_text(size = 17, margin = margin(t = 0)),
      plot.margin = margin(0, 1, 0, 0, unit = "pt")    
    )

    show_legend <- TRUE
    #names(legend_labels) <- highlight_cols
    
  } else {
    # no highlights requested → fall back to your old categories logic
    comparison_df <- comparison_df %>%
      mutate(
        tooltip_txt = ifelse(
          categories == "not significant",

          NA_character_,
          paste0(
            "Peptide: ",  Peptide,               "<br>",
            "Desc: ",     Description,           "<br>",
            "Species: ",  species,"<br>",
            group1, ": ", !!sym(group1), " / ",
            group2, ": ", !!sym(group2)
          )
        )
      ) %>%
      filter(
        is.finite(.data[[group1]]),
        is.finite(.data[[group2]])
      )
    
    color_aes   <- aes(color = categories, text = tooltip_txt)
    color_scale <- scale_color_manual(
      values = significant_colors, 
      labels = c("ns", "significant", "significant FDR"),
      name = NULL)
    legend_theme <- theme(legend.position = "none")
    show_legend  <- FALSE
  }
  
  g1 <- as.character(group1)[1]
  g2 <- as.character(group2)[1]
  n1 <- as.integer(N[1])
  n2 <- as.integer(N[2])
  
  axis_lab <- function(g, n) {
    substitute(
      atop(paste("% ", bold(G), " in whom a peptide is"),
           paste("significantly bound (n = ", Nn, ")")),
      list(G = g, Nn = n)
    )
  }

  # build the ggplot + ggplotly -----------------------------------------------
  p <- ggplot(comparison_df,
              aes(x = !!sym(group1), y = !!sym(group2))) +
    geom_point(color_aes, alpha = 0.65, size = 2.8) 
  
  if(reverse_legend){
    p <- p + guides(color = guide_legend(reverse = TRUE))
  }
  
  p <- p + geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray4") +
    color_scale +
    labs(
       x = axis_lab(g1, n1),
       y = axis_lab(g2, n2)
    ) +
    theme_bw(base_size = 16) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border     = element_rect(colour = "black", fill = NA)
    ) +
    legend_theme
  
  
  if (interactive){
    interactive_plot <- ggplotly(p, tooltip = "text",
                                 width   = 550, height  = 550)
    
    if (!is.null(highlight_cols) && length(highlight_cols) > 0) {
      # only then do the trace‐name patching:
      for (i in seq_along(interactive_plot$x$data)) {
        tr    <- interactive_plot$x$data[[i]]
        nm    <- tr$name
        # hide the greys
        if (nm %in% c("none","multiple")) {
          tr$showlegend <- FALSE
        }
        # relabel the real flags
        else if (nm %in% highlight_cols) {
          tr$name <- legend_labels[[nm]]
        }
        interactive_plot$x$data[[i]] <- tr
      }
    }
    
    # --- Reverse legend starting from second element only ---
    if (reverse_legend && length(interactive_plot$x$data) > 1) {
      # Keep first trace as is
      first_trace <- interactive_plot$x$data[[1]]
      rest_traces <- interactive_plot$x$data[-1]
      
      n <- length(rest_traces)
      
      # Assign legendrank to reverse legend order, without touching trace order
      for (i in seq_along(rest_traces)) {
        rest_traces[[i]]$legendrank <- n - i + 1
      }
      
      # Put everything back together
      interactive_plot$x$data <- c(list(first_trace), rest_traces)
    }
    
    return(
      interactive_plot %>% layout(
        showlegend = show_legend,
        
        font = list(size   = 13),
        xaxis = list(
          title = list(                       # the x‐axis title
            text =  paste0("% ", group1, " in whom\na peptide is significantly bound\n(n = ", N[1], ")"),
            font = list(size   = 14)
          )
        ),
        yaxis = list(
          title = list(                       # the y‐axis title
            text = paste0("% ", group2, " in whom\na peptide is significantly bound\n(n = ", N[2], ")"),
            font = list(size   = 14)
          )
        ),
        
        legend = list(
          #orientation = "h",       # horizontal keys
          x       = 0,       # center
          xanchor = "left",
          y       = 1,      # 95% up the plot area
          yanchor = "top",
          font        = list(size = 10)
        ),
        margin     = list(l = 80, r = 80, b = 80, t = 80, pad = 0),
        hoverlabel = list(font = list(size = 10)),
        # xaxis      = list(scaleratio = 1, scaleanchor = "y"),
        # yaxis      = list(scaleratio = 1, scaleanchor = "x")
        xaxis      = list(
          scaleratio = 1, 
          scaleanchor = "y",
          showline = TRUE,         # Add the axis line
          linewidth = 1,           # Set line thickness (optional, default is fine)
          linecolor = 'black',     # Set line color
          mirror = TRUE            # Mirror the line on the top
        ),
        yaxis      = list(
          scaleratio = 1, 
          scaleanchor = "x",
          showline = TRUE,         # Add the axis line
          linewidth = 1,           # Set line thickness
          linecolor = 'black',     # Set line color
          mirror = TRUE            # Mirror the line on the right
        )
      )
    )
  } else {
    return(p)
  }
}



group_test_analysis <- function(percentage_group_test_list, num_samples_per_group,
                                prevalence_threshold = 5, counts_threshold = 2) {
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
        
        N1 <- num_samples_per_group[[group_col]][[group1]]
        N2 <- num_samples_per_group[[group_col]][[group2]]
        
        # ——— apply pairwise prevalence filter ——————————————
        # df_pair <- df %>%
        #   # keep only peptides with >= threshold in at least one of the two
        #   filter(
        #     .data[[ group1 ]] >= prevalence_threshold |
        #       .data[[ group2 ]] >= prevalence_threshold
        #   )
        df_pair <- df %>%
          filter(
            if (!is.null(prevalence_threshold)) {
              .data[[group1]] >= prevalence_threshold |
                .data[[group2]] >= prevalence_threshold
            } else {
              .data[[paste0(group1, "_count")]] >= counts_threshold |
                .data[[paste0(group2, "_count")]] >= counts_threshold
            }
          )
        
        if (nrow(df_pair) == 0) {
          warning("No peptides pass the prevalence filter for ", 
                  group1, " vs ", group2)
          next
        }
        
        # Calculate p-values and ratios for each pair of groups
        delta_ratio_vals <- numeric(nrow(df_pair))
        ratio_vals       <- numeric(nrow(df_pair))
        pvals_chisq      <- numeric(nrow(df_pair))
        epsilon_prop      <- 0.5
        epsilon_delta    <- 1
        
        # Loop over each row in the filtered dataframe
        for (k in 1:nrow(df_pair)) {
          # Extract the values for the two groups being compared
          val1 <- df_pair[[paste0(group1, "_count")]][k]
          val2 <- df_pair[[paste0(group2, "_count")]][k]
          
          # Construct the contingency table using counts from num_samples_per_group
          chitable <- matrix(c(val1 + 1, N1 - val1 + 1, 
                               val2 + 1, N2 - val2 + 1), 
                             nrow = 2, byrow = TRUE)
          
          # Chi-squared test 
          #pvals_chisq[k] < chisq.test(chitable)$p.value
          
          # Fisher test 
          test_result <- fisher.test(chitable)
          pvals_chisq[k] <- test_result$p.value
          
          
          # ratio (with 0.5 pseudocount only for zeros)
          val1_prop <- (val1 + ifelse(val1 == 0, epsilon_prop, 0)) / N1
          val2_prop <- (val2 + ifelse(val2 == 0, epsilon_prop, 0)) / N2
          ratio_vals[k] <- (val1_prop / val2_prop)
          
          
          # Delta-ratio (with 1 pseudocount only for zeros)
          val1_delta <- (val1 + ifelse(val1 == 0, epsilon_delta, 0)) / N1
          val2_delta <- (val2 + ifelse(val2 == 0, epsilon_delta, 0)) / N2
          if (val1_delta >= val2_delta) {
            delta_ratio_vals[k] <- val1_delta / val2_delta - 1
          } else {
            delta_ratio_vals[k] <- -(val2_delta / val1_delta - 1)
          }
        }
        
        # Add p-values and ratios to the dataframe
        comparison_df <- df_pair %>%
          mutate(
            # across(
            #   .cols = all_of(c("class", "order", "family", "genus", "species")),
            #   .fns  = ~ na_if(.x, "")
            # ),
            Delta_ratio = delta_ratio_vals,
            ratio = ratio_vals,
            pvals_not_adj = pvals_chisq,
            passed_not_adj = ifelse(pvals_chisq < 0.05, TRUE, FALSE),
            pvals_bh = p.adjust(pvals_not_adj, method = "BH"),
            pvals_bh_format = sapply(pvals_bh, format_pval),
            
            passed_bh = p.adjust(pvals_not_adj, method = "BH") < 0.05,#,
            categories = case_when(
              passed_bh                     ~ "significant post FDR correction",
              passed_not_adj & !passed_bh   ~ "significant prior correction",
              TRUE                          ~ "not significant"
            )
          )  %>%
          dplyr::arrange(
            #Delta_ratio,            # then by direction of change
            #!passed_bh,            # TRUE (significant) comes first
            pvals_bh,              # then smallest BH-adjusted p-values
            pvals_not_adj,        # then smallest raw p-values
          ) %>%
          dplyr::select(Peptide, Description, everything())
        
        
        # Generate scatter plot for the comparison
        p <- make_interactive_scatterplot(comparison_df = comparison_df, group1 = group1, group2 = group2, N = c(N1,N2))
        
        # Store the plot and results in the list
        comparison_results[[paste0(group1, "_vs_", group2)]] <- list(
          plot = p,
          comparison_df = comparison_df
        )
      }
    }
    
    # Store all comparison results for this group_test
    results_list[[group_col]] <- comparison_results
  }
  
  return(results_list)
}


#########################################
############ Volcano plot ###############
#########################################
make_interactive_volcano <- function(comparison_df, group1, group2,
                                     fc_cut = 1,
                                     p_cut  = 0.05,
                                     highlight_cols   = NULL,
                                     highlight_colors = NULL,
                                     default_color    = "gray70",
                                     pvals_adj = NULL,
                                     reverse_legend = FALSE,
                                     significant_colors = c(
                                       "not significant"                 = "dodgerblue",
                                       "significant prior correction"    = "forestgreen",
                                       "significant post FDR correction" = "firebrick"),
                                     interactive = TRUE){
  
  # sanity-check:
  if (!is.null(highlight_cols)) {
    missing_cols <- setdiff(highlight_cols, names(comparison_df))
    if (length(missing_cols)) {
      stop("These highlight_cols are not in your data frame: ",
           paste(missing_cols, collapse = ", "))
    }
  }
  
  # build the collapsed factor ------------------------------------------------
  if (!is.null(highlight_cols) && length(highlight_cols) > 0) {
    plotly_width <- 650
    margins <- list(l   = 80, r   = 120, t   = 80, b   = 40, pad = 0)
    comparison_df <- comparison_df %>%
      rowwise() %>%
      mutate(
        # collect the names of all TRUE flags in this row:
        .trues = list(highlight_cols[ c_across(all_of(highlight_cols)) ]),
        # now assign highlight:
        highlight = if (length(.trues) == 0) {
          "none"
        } else if (length(.trues) >= 1) {
          .trues[[1]]
        } #else {
        #"multiple"}
      ) %>%
      ungroup() %>%
      select(-.trues) 
    
    # ensure factor has all levels:
    comparison_df <- comparison_df %>% 
      mutate(highlight = replace_na(highlight, "none"))
    
    levels_needed <- c("none", highlight_cols) #multiple
    comparison_df$highlight <- factor(
      comparison_df$highlight,
      levels = levels_needed
    )
    # build tooltip (only for highlighted points)
    comparison_df <- comparison_df %>%
      mutate(
        log2ratio = log2( ratio ),
        tooltip_txt = if_else(
          highlight == "none",
          NA_character_,
          paste0(
            "Peptide: ",  Peptide,               "<br>",
            "Desc: ",     Description,           "<br>",
            "Species: ",  species,"<br>",
            group1, ": ", !!sym(group1), " / ",
            group2, ": ", !!sym(group2),       "<br>",
            "Highlight: ", highlight
          )
        )
      ) %>%
      filter(is.finite(log2ratio), is.finite(-log10(pvals_not_adj))) %>%  # drop any Inf or NaN 
      arrange(highlight)
    
    if (is.null(pvals_adj)){
      pvals <- sapply(highlight_cols, function(flag) {
        x <- comparison_df %>% filter( !!sym(flag) ) %>% pull(log2ratio)
        #y <- comparison_df$log2ratio
        # better subset y to only non-flag too:
        y <- comparison_df %>% filter(! (!!sym(flag)) ) %>% pull(log2ratio)
        w <- wilcox.test(x, y)
        w$p.value
      })
      pvals_adj <- p.adjust(pvals, method = "BH")
    }
    fmt_p <- sapply(pvals_adj, format_pval)
    legend_labels <- paste0(highlight_cols, " (P=", fmt_p, ")")
    
    
    # colors: user‐supplied or a simple default palette
    if (is.null(highlight_colors)) {
      # pick a palette for the flags
      palette_vals <- setNames(
        RColorBrewer::brewer.pal(
          n = max(length(highlight_cols), 8),
          name = "Set2"
        )[1:length(highlight_cols)],
        highlight_cols
      )
    } else {
      palette_vals <- highlight_colors
    }
    manual_vals <- c(
      none     = default_color,
      #multiple = multiple_color,
      palette_vals
    )
    
    color_aes   <- aes(color = highlight, text = tooltip_txt)
    color_scale <- scale_color_manual(
      name   = NULL,
      values = manual_vals,
      #limits = levels_needed,    # ← ensures “none” is the first group drawn
      breaks = highlight_cols,
      labels = legend_labels
      #labels = c("Milk allergens", "Enterovirus", "Bacteriodes")
    )
    legend_theme <- theme(
      #legend.position   = "top",            # place above the plot
      #legend.justification = "center",      # center it
      legend.position     = c(0, 1),   # 50% across, 95% up
      legend.justification = c(0, 1),  
      
      legend.background    = element_rect(fill = NA, color = "gray80"), #element_rect(fill = alpha("white", 0.8), color = "gray80"),
      legend.key.size      = unit(13, "pt"),
      legend.text          = element_text(size = 12),
      legend.title         = element_text(size = 14),
      axis.text.y.left = element_text(size = 13),
      axis.text.x.bottom = element_text(size = 13),
      axis.title.y = element_text(size = 14),
      axis.title.x = element_text(size = 14, margin = margin(t = 0)),
      plot.margin = margin(0, 1, 0, 0, unit = "pt")    
    )
    
    show_legend <- TRUE
    names(legend_labels) <- highlight_cols
    
  } else {
    # no highlights requested → fall back to your old categories logic
    plotly_width <- 500
    margins <- list(l = 0, r = 0, t = 0, b = 0, pad = 2)
    comparison_df <- comparison_df %>%
      mutate(
        tooltip_txt = ifelse(
          #categories == "ns", #"not significant",
          categories == "not significant",
          NA_character_,
          paste0(
            "Peptide: ",  Peptide,               "<br>",
            "Desc: ",     Description,           "<br>",
            "Species: ",  species,"<br>",
            group1, ": ", !!sym(group1), " / ",
            group2, ": ", !!sym(group2)
          )
        )
      ) %>%
      filter(
        is.finite(.data[[group1]]),
        is.finite(.data[[group2]])
      )
    
    color_aes   <- aes(color = categories, text = tooltip_txt)
    color_scale <- scale_color_manual(
      values = significant_colors, 
      labels = c("ns", "significant", "significant FDR"),
      name = NULL)
    legend_theme <- theme(legend.position = "none")
    show_legend  <- FALSE
  }
  
  
  p <- ggplot(comparison_df, aes(x = -log2(ratio), y=-log10(pvals_not_adj))) +
    geom_point(color_aes, alpha = 0.65)
  
  if(reverse_legend){
    p <- p + guides(color = guide_legend(reverse = TRUE))
  }
  
  p <- p +  geom_hline( yintercept = -log10(p_cut), linetype   = "dashed", color = "gray50") +
    geom_vline( xintercept = c(fc_cut,-fc_cut), linetype   = "dashed", color = "gray50") +
    geom_vline( xintercept = 0, linetype   = "dashed", color = "gray50",alpha = 0.5) +
    # scale_color_manual(
    #   values = volcano_colors, 
    #   name   = NULL) +
    color_scale +
    labs(
      x = paste0("log₂-ratio of antibody responses\nin ", group1, " and ", group2),
      y = paste0("-log₁₀(p-value)")
    ) +
    theme_bw(base_size = 13) +  # Use a minimal theme for elegance and set a base font size
    theme(
      panel.grid.major = element_blank(),  # Remove major grid lines
      panel.grid.minor = element_blank(),  # Remove minor grid lines
      panel.border = element_rect(colour = "black", fill = NA)  # Keep border if desired
      #plot.margin = margin(t = 10, r = 15, b = 15, l = 10, unit = "pt"),  # Adjust margins in point
      #axis.text.y.left = element_text(size = 10, face = "italic"),  # Style y-axis labels
      #axis.title.x = element_text(face = "bold"),  # Add margin and bold to x-axis title
      #axis.title.y = element_text(face = "bold")  # Bold y-axis title
    ) +
    legend_theme
  
  if (interactive){
    interactive_plot <- ggplotly(p, tooltip = c("text", "x", "y"),
                                 width   = plotly_width, height  = 500)
    
    
    if (!is.null(highlight_cols) && length(highlight_cols) > 0) {
      # only then do the trace‐name patching:
      for (i in seq_along(interactive_plot$x$data)) {
        tr    <- interactive_plot$x$data[[i]]
        nm    <- tr$name
        # hide the greys
        if (nm %in% c("none","multiple")) {
          tr$showlegend <- FALSE
        }
        # relabel the real flags
        else if (nm %in% highlight_cols) {
          tr$name <- legend_labels[[nm]]
        }
        interactive_plot$x$data[[i]] <- tr
      }
    }
    
    # --- Reverse legend starting from second element only ---
    if (reverse_legend && length(interactive_plot$x$data) > 1) {
      # Keep first trace as is
      first_trace <- interactive_plot$x$data[[1]]
      rest_traces <- interactive_plot$x$data[-1]
      
      n <- length(rest_traces)
      
      # Assign legendrank to reverse legend order, without touching trace order
      for (i in seq_along(rest_traces)) {
        rest_traces[[i]]$legendrank <- n - i + 1
      }
      
      # Put everything back together
      interactive_plot$x$data <- c(list(first_trace), rest_traces)
    }
    
    return(
      interactive_plot %>% layout(
        showlegend = show_legend,
        
        font = list(size   = 13),
        xaxis = list(
          title = list(                       # the x‐axis title
            text =  paste0("log₂-ratio of antibody responses\nin ", group1, " and ", group2),
            font = list(size   = 14)
          )
        ),
        yaxis = list(
          title = list(                       # the y‐axis title
            text = paste0("-log₁₀(p-value)"),
            font = list(size   = 14)
          )
        ),
        
        legend = list(
          #orientation = "h",       # horizontal keys
          x       = 1.02,       # center
          xanchor = "left",
          y       = 1,      # 95% up the plot area
          yanchor = "top",
          font        = list(size = 13)
        ),
        margin = margins, #list(l   = 80, r   = 120, t   = 80, b   = 40, pad = 0),
        hoverlabel  = list(font = list(size = 10)),
        
        xaxis      = list(
          automargin = TRUE,
          showline = TRUE,         #  Add the axis line
          linewidth = 1,           #  Set line thickness (optional, default is fine)
          linecolor = 'black',     #  Set line color
          mirror = TRUE            #  Mirror the line on the top
        ),
        yaxis      = list(
          automargin = TRUE,
          title      = list(standoff = 10),
          showline = TRUE,         #  Add the axis line
          linewidth = 1,           #  Set line thickness
          linecolor = 'black',     #  Set line color
          mirror = TRUE            #  Mirror the line on the right
        ),
        
        # xaxis       = list(automargin = TRUE),
        # yaxis       = list(automargin = TRUE, title      = list(standoff = 10)),
        autosize    = FALSE
      )
    )
  } else {
    return(p)
  }
}

#########################################
###############MDS#######################
############deprecated###################


plot_mds <- function(features_target, group_col, 
                     custom_colors,  group_cols =NA,
                     method = "jaccard", ellipse = T,
                     permutations = 999) {
  
  # Full matrix sames as exist for the corresponding group and transposed
  binary_data_all <- features_target %>%
    filter(!is.na(.data[[group_col]])) %>%       # Only keep rows with non-NA group
    dplyr::select(-any_of(group_cols)) %>%
    tibble::column_to_rownames("SampleName")
  
  # Distance matrices
  dist_all <- vegan::vegdist(binary_data_all, method = method)
  mds_all <- cmdscale(dist_all, k = 2, eig = TRUE)
  var_exp_all <- round(100 * mds_all$eig[1:2] / sum(mds_all$eig), 2)
  
  
  # MDS dataframe
  mds_df_all <- as.data.frame(mds_all$points) %>%
    tibble::rownames_to_column("SampleName") %>%
    left_join(features_target %>% select(any_of(c("SampleName", group_col))), by = "SampleName")
  
  # Permanova test
  permanova <-   vegan::adonis2(dist_all ~ mds_df_all[[group_col]], permutations = permutations)
  p_value <- permanova$`Pr(>F)`[1]
  
  # Dispersion test
  bd  <- vegan::betadisper(dist_all, mds_df_all[[group_col]])
  bt  <- vegan::permutest(bd, permutations=permutations)
  
  # Centroids for plotting
  centroids <- mds_df_all %>%
    group_by(.data[[group_col]]) %>%
    summarise(V1 = mean(V1), V2 = mean(V2), .groups = "drop")
  # hulls <- mds_df_all %>%
  #   group_by(group_test) %>%
  #   slice(chull(V1, V2))
  
  # Main plot
  p <- ggplot(mds_df_all, aes(x = V1, y = V2, fill = .data[[group_col]]))
  
  # Conditionally add the ellipse
  if (ellipse) {
    p <- p + stat_ellipse(aes(colour = .data[[group_col]]), type = "t", level = 0.95, fill = NA, geom = "path", size = 0.8, alpha = 0.5, show.legend = FALSE)
  }
  
  # Add all the remaining layers, making sure to use + at the end of each line
  p <- p +  geom_point(size = 3, alpha = 0.5, shape = 21, color = "black",  stroke = 0, aes(text = paste("Sample:", SampleName))) +
    geom_point(data = centroids, aes(x = V1, y = V2, fill = .data[[group_col]]),
               show.legend = FALSE, size = 5, shape = 21, color = "black") +
    #geom_polygon(data = hulls, aes(x = V1, y = V2, group = .data[[group_col]], color = .data[[group_col]]), fill=NA) +
    scale_fill_manual(values = custom_colors) +
    scale_color_manual(values = custom_colors) +
    # Turn off any stray color legend
    guides(color = "none") +
    labs(
      x = paste0("MDS 1 (", var_exp_all[1], "%)"),
      y = paste0("MDS 2 (", var_exp_all[2], "%)"),
      title = glue::glue(
        "PERMANOVA p={format(permanova$`Pr(>F)`[1], digits=2)}; dispersion p={format(bt$tab$`Pr(>F)`[1],digits=2)}"
      )
    ) +
    #   title = paste0("MDS (PCoA) with Jaccard Distance\nPERMANOVA p = ", round(p_value, 4),
    #                  " | permutations = ", permutations)
    # ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    )
  return(p)
}

####################################
#############PCA###################
####################################
plot_pca <- function(fold_df, group_col, custom_colors,
                     group_cols = NULL, prevalence_cutoff = 0.05,
                     log_transform = TRUE) {
  
  # 1) Identify peptide columns
  peptide_cols <- setdiff(
    names(fold_df),
    c("SampleName", group_col, group_cols)
  )
  
  # 2) Find peptides with ≥ cutoff prevalence in any group
  keep_peptides <- fold_df %>%
    filter(!is.na(.data[[group_col]])) %>%
    tidyr::pivot_longer(all_of(peptide_cols),
                        names_to  = "peptide",
                        values_to = "fc") %>%
    mutate(present = fc != 0) %>%
    group_by(.data[[group_col]], peptide) %>%
    summarise(prevalence = mean(present), .groups = "drop") %>%
    group_by(peptide) %>%
    filter(any(prevalence >= prevalence_cutoff)) %>%
    pull(peptide) %>%
    unique()
  
  # 3) Build filtered matrix (samples × kept peptides)
  mat <- fold_df %>%
    filter(!is.na(.data[[group_col]])) %>%
    select(SampleName, all_of(keep_peptides)) %>%
    tibble::column_to_rownames("SampleName") %>%
    as.matrix()
  
  # 4) Optionally log2-transform fold-changes
  if (log_transform) {
    mat <- log2(mat + 1)  
    # +1 avoids log(0); use another pseudocount if you prefer
  }
  
  # 5) PCA
  pca_res <- prcomp(mat, center = TRUE, scale. = FALSE)
  var_exp  <- round(100 * pca_res$sdev^2 / sum(pca_res$sdev^2), 1)
  
  pca_df <- as.data.frame(pca_res$x[,1:2]) %>%
    tibble::rownames_to_column("SampleName") %>%
    left_join(
      fold_df %>% select(SampleName, any_of(group_col)),
      by = "SampleName"
    )
  
  # 6) Plot PC1 vs PC2
  ggplot(pca_df, aes(x = PC1, y = PC2, fill = .data[[group_col]])) +
    geom_point(shape = 21, color = "black", size = 3, alpha = 0.6) +
    scale_fill_manual(values = custom_colors) +
    labs(
      x = paste0("PC1 (", var_exp[1], "%)"),
      y = paste0("PC2 (", var_exp[2], "%)"),
      fill = group_col
    ) +
    theme_bw() +
    theme(
      plot.title      = element_text(hjust = 0.5, face = "bold"),
      legend.position = "right"
    )
}

####################################
############Similarity pot##########
####################################
compute_patient_correlation <- function(data_matrix, metadata, 
                                        ind_id_col = "ind_id", 
                                        samples_t1_col = "SampleName_t1", 
                                        samples_t2_col = "SampleName_t2",
                                        method = "pearson") {
  
  # Order the metadata by patient_id to align pairs consistently.
  metadata <- metadata[order(metadata[[ind_id_col]]), ]
  
  # Extract baseline and t2 sample names from metadata as character vectors.
  t1_samples <- as.character(metadata[[samples_t1_col]])
  t2_samples       <- as.character(metadata[[samples_t2_col]])
  
  # Create an empty correlation matrix with rows = baseline samples and columns = t2 samples.
  # Note: The dimnames come directly from the metadata values.
  corr_mat <- matrix(NA, nrow = length(t1_samples), ncol = length(t2_samples),
                     dimnames = list(t1_samples, t2_samples))
  
  # Loop over each pair of baseline and t2 sample names.
  for (i in seq_along(t1_samples)) {
    for (j in seq_along(t2_samples)) {
      b_sample <- t1_samples[i]
      t_sample <- t2_samples[j]
      
      # If one of the sample names is missing (or is NA/empty), leave that cell as NA.
      if (is.na(b_sample) || is.na(t_sample) || b_sample == "" || t_sample == "") {
        corr_mat[i, j] <- NA
      } else if (!(b_sample %in% rownames(data_matrix)) || !(t_sample %in% rownames(data_matrix))) {
        # If the sample is not found in the data matrix, set correlation to NA.
        corr_mat[i, j] <- NA
      } else {
        # Otherwise, extract the peptide profiles (each is a numeric vector)
        vec_b <- as.numeric(data_matrix[b_sample, ])
        vec_t <- as.numeric(data_matrix[t_sample, ])
        
        # If either profile has zero variance, correlation is undefined (set to NA)
        if (sd(vec_b, na.rm = TRUE) == 0 || sd(vec_t, na.rm = TRUE) == 0) {
          corr_mat[i, j] <- NA
        } else {
          # Compute the correlation between the two samples.
          corr_mat[i, j] <- cor(vec_b, vec_t, use = "pairwise.complete.obs", method = method)
        }
      }
    }
  }
  
  return(corr_mat)
}

plot_correlation <- function(phiseq_df, metadata,
                             ind_id_col = "ind_id", samples_t1_col = "SampleName_t1", samples_t2_col = "SampleName_t2",
                             label_x = "t1", label_y = "t2",
                             sort_by_status = FALSE, pre_status_col = "status_t1", post_status_col = "status_t2",
                             method = "pearson", require_both_timepoints = FALSE) {
  
  # Optionally filter metadata to only include individuals with both timepoints
  use_fallback <- FALSE
  if (require_both_timepoints) {
    tmp_meta <- metadata %>%
      filter(g1_exists & g2_exists)
    if (nrow(tmp_meta) == 0) {
      print("No individuals with both timepoints. Falling back to those with either timepoint.")
      tmp_meta <- metadata
      use_fallback <- TRUE
    }
    metadata <- tmp_meta
  }
  
  # Compute correlation matrix for this subgroup
  corr_matrix <- compute_patient_correlation(
    data_matrix = t(phiseq_df),
    metadata = metadata,
    ind_id_col = ind_id_col,
    samples_t1_col = samples_t1_col,
    samples_t2_col = samples_t2_col,
    method = method
  )
  
  if (sort_by_status == TRUE){
    metadata <- metadata %>% 
      arrange((!!sym(pre_status_col)), (!!sym(post_status_col)), (!!sym(ind_id_col))) %>% 
      slice(rev(row_number()))
  }
  
  # 2. Extract the sample names in the desired order.
  ordered_ids <- metadata[[ind_id_col]]
  
  
  # Format for plotting
  rownames(corr_matrix) <- as.character(ordered_ids)
  colnames(corr_matrix) <- as.character(ordered_ids)
  
  cor_df <- reshape2::melt(corr_matrix)
  
  if (sort_by_status == TRUE){
    cor_df <- cor_df %>%
      left_join(metadata %>%  select(patient_id = !!sym(ind_id_col), pre_status = !!sym(pre_status_col)),
                by = c("Var1" = ind_id_col)) %>%
      left_join(metadata %>% select(patient_id = !!sym(ind_id_col), post_status = !!sym(post_status_col)),
                by = c("Var2" = ind_id_col))
  }
  
  cor_df$Var1 <- factor(cor_df$Var1, levels = ordered_ids)
  cor_df$Var2 <- factor(cor_df$Var2, levels = ordered_ids)
  
  if (use_fallback) {
    cor_df <- cor_df %>% filter(!is.na(value))
  }
  
  # Plot
  p <- ggplot(cor_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(colour = "gray60", linewidth = 0.25) +
    scale_fill_distiller(palette = "RdYlBu", direction = -1, limits = c(0, 1),
                         na.value = "gray", name = "Pearson\nCorrelation") +
    coord_fixed() +
    theme_bw() +
    theme(
      axis.text.x = element_text(size=6.5, angle = 45, vjust = 1, hjust = 0.5),
      axis.text.y = element_text(size = 6.5),
      panel.grid = element_blank(),
      panel.border = element_blank()
    ) +
    #scale_x_discrete(breaks = col_breaks) +
    #scale_y_discrete(breaks = row_breaks) 
    labs(
      x = paste0(label_x," Ig epitope repertoire ", "\n(individual ID)"),
      y = paste0(label_y," Ig epitope repertoire ", "\n(individual ID)")) 
  
  return(p)
}


plot_correlation_distribution <- function(phiseq_df, metadata,
                                          ind_id_col = "ind_id",
                                          samples_t1_col = "SampleName_t1",
                                          samples_t2_col = "SampleName_t2",
                                          label_x = "t1", label_y = "t2",
                                          method = "pearson",
                                          bins = 20,
                                          require_both_timepoints = FALSE) {
  use_fallback <- FALSE
  if (require_both_timepoints) {
    tmp_meta <- metadata %>%
      filter(g1_exists & g2_exists)
    if (nrow(tmp_meta) == 0) {
      print("No individuals with both timepoints. Falling back to those with either timepoint.")
      tmp_meta <- metadata
      use_fallback <- TRUE
    }
    metadata <- tmp_meta
  }
  
  corr_matrix <- compute_patient_correlation(
    data_matrix = t(phiseq_df),
    metadata = metadata,
    ind_id_col = ind_id_col,
    samples_t1_col = samples_t1_col,
    samples_t2_col = samples_t2_col,
    method = method
  )
  
  ordered_ids <- metadata[[ind_id_col]]
  rownames(corr_matrix) <- as.character(ordered_ids)
  colnames(corr_matrix) <- as.character(ordered_ids)
  
  cor_df <- reshape2::melt(corr_matrix) %>%
    mutate(pair_type = ifelse(Var1 == Var2, "matched", "random"))
  
  
  if (use_fallback) {
    cor_df <- cor_df %>% filter(!is.na(value))
  }
  
  # Separate matched and random
  matched_df <- cor_df %>% filter(pair_type == "matched")
  random_df  <- cor_df %>% filter(pair_type == "random")
  
  # Initialize an empty plot
  p <- ggplot()
  
  has_matched <- nrow(matched_df) > 0
  has_random  <- nrow(random_df) > 0
  
  if (has_random) {
    random_hist <- ggplot(random_df, aes(x = value)) + geom_histogram(bins = bins)
    random_data <- ggplot_build(random_hist)$data[[1]]
    p <- p + geom_col(data = random_data, aes(x = x, y = y),
                      width = random_data$width, fill = "palegreen4", alpha = 0.4)
  }
  
  if (has_matched) {
    matched_hist <- ggplot(matched_df, aes(x = value)) + geom_histogram(bins = bins)
    matched_data <- ggplot_build(matched_hist)$data[[1]]
    
    # Fallback to default if no matched histogram data was produced
    has_matched_hist <- !is.null(matched_data) && nrow(matched_data) > 0 && max(matched_data$y) > 0
    
    if (has_random && has_matched_hist) {
      scale_factor <- max(random_data$y) / max(matched_data$y)
      matched_data <- matched_data %>%
        mutate(y_scaled = y * scale_factor)
      
      p <- p + geom_col(data = matched_data, aes(x = x, y = y_scaled),
                        width = matched_data$width, fill = "dodgerblue3", alpha = 0.4) +
        scale_y_continuous(
          name = "Random Pair Count",
          sec.axis = sec_axis(~ . / scale_factor, name = "Matched Pair Count")
        )
    } else if (has_matched_hist) {
      # Only matched → plot as primary y-axis
      p <- p + geom_col(data = matched_data, aes(x = x, y = y),
                        width = matched_data$width, fill = "dodgerblue3", alpha = 0.4) +
        scale_y_continuous(name = "Matched Pair Count")
    }
  }
  
  p <- p +
    scale_x_continuous(name = paste0("Pearson correlation of\n", label_x, " vs ", label_y)) +
    theme_bw() +
    theme(panel.grid = element_blank())
  
  return(p)
}


###################################################
############ ratios subgroups######################
###################################################

# plot_ratios_by_subgroup <- function(comparison_df, group1, group2, subgroup_lib_df, prevalence_threshold = 5) {
#   # Join peptide subgroup flags
#   long_ratios <- comparison_df %>%
#     mutate(ratio = log2(ratio)) %>% 
#     left_join(subgroup_lib_df, by = "Peptide") %>%
#     tidyr::pivot_longer(cols = all_of(SUBGROUPS_TO_INCLUDE),
#                         names_to = "subgroup_flag",
#                         values_to = "belongs_to_group") %>%
#     dplyr::filter(belongs_to_group) %>%
#     mutate(subgroup = factor(SUBGROUPS_TO_NAME[subgroup_flag],
#                              levels = SUBGROUPS_ORDER)) %>%
#     dplyr::filter(.data[[group1]] >= prevalence_threshold | .data[[group2]] >= prevalence_threshold)
#   
#   
#   # Skip empty plots
#   if (nrow(long_ratios) < 10) {
#     warning("Not enough data for ", group1, " vs ", group2)
#     return(NULL)
#   }
#   
#   # Get pairwise subgroup comparisons
#   subgroup_labels <- levels(long_ratios$subgroup)
#   pairwise_combos <- combn(subgroup_labels, 2, simplify = FALSE)
#   
#   # Get significant pairs
#   sig_comparisons <- ggpubr::compare_means(
#     formula = ratio ~ subgroup,
#     data = long_ratios,
#     method = "wilcox.test",
#     comparisons = pairwise_combos,
#     p.adjust.method = "bonferroni"
#   ) %>% filter(p.adj < 0.01)
#   
#   # Format for stat_compare_means
#   sig_pairs <- lapply(seq_len(nrow(sig_comparisons)), function(i) {
#     c(sig_comparisons$group1[i], sig_comparisons$group2[i])
#   })
#   
#   # Plot
#   p <- ggplot(long_ratios, aes(x = subgroup, y = ratio, fill = subgroup)) +
#     geom_boxplot(outlier.shape = 21, outlier.size = 1, width = 0.6) +
#     scale_fill_brewer(palette = "Paired") +
#     ggpubr::stat_compare_means(
#       method = "wilcox.test",
#       comparisons = sig_pairs,
#       p.adjust.method = "BH",
#       label = "p.signif",
#       hide.ns = TRUE,
#       size = 4
#     ) +
#     theme_bw() +
#     theme(
#       axis.text.x = element_text(angle = 45, hjust = 1),
#       legend.position = "none"
#     ) +
#     labs(
#       x = "Subgroups of the antigen library",
#       y = paste("log-ratio of antibody responses\nin", group1, "and", group2, sept=" ")
#     )
#   
#   return(p)
# }

# Updated plot_ratios_by_subgroup with simplified subgroup logic
# • which_subgroups = "all" (subgroups + any new organism), "default (only subgroups", or "added (only new organisms"
plot_ratios_by_subgroup <- function(
    comparison_df,
    group1,
    group2,
    subgroup_lib_df,
    subgroup_colors      = NULL,
    prevalence_threshold = 0,
    add_subgroups        = NULL,
    which_subgroups      = c("all", "default", "added"),
    x_label = "Subgroups of the antigen library"
) {
  which_subgroups <- match.arg(which_subgroups)
  
  
  # error if no additional subgroups provided but 'added' or 'all' requested
  if (is.null(add_subgroups) && which_subgroups %in% c("added","all")) {
    stop("`add_subgroups` must be provided when which_subgroups is 'added' or 'all'.")
  }
  
  # if add_subgroups given, ensure subgroup_lib_df has those columns
  if (!is.null(add_subgroups)) {
    missing_cols <- setdiff(add_subgroups, names(subgroup_lib_df))
    if (length(missing_cols)) {
      stop("The following 'add_subgroups' are not columns in 'subgroup_lib_df': ",
           paste(missing_cols, collapse = ", "))
    }
  }
  
  # additional subgroups
  added <- if (!is.null(add_subgroups)) add_subgroups else character()
  
  # Build full lists only if needed
  all_incl  <- c(SUBGROUPS_TO_INCLUDE, added)
  all_order <- c(SUBGROUPS_ORDER, added)
  all_names <- SUBGROUPS_TO_NAME
  if (length(added)) {
    for (x in added) all_names[x] <- x
  }
  
  # Determine which flags and ordering based on user choice
  keep_flags <- switch(
    which_subgroups,
    default = SUBGROUPS_TO_INCLUDE,
    added   = added,
    all     = all_incl
  )
  keep_order <- switch(
    which_subgroups,
    default = SUBGROUPS_ORDER,
    added   = added,
    all     = all_order
  )
  
  
  comparison_df <- comparison_df %>% 
    mutate(log2ratio = -log2(ratio)) %>%
    # bring in your logical flags (one column per keep_flag)
    left_join(subgroup_lib_df, by = "Peptide") %>% 
    filter(
      .data[[group1]] >= prevalence_threshold |
        .data[[group2]] >= prevalence_threshold
    )
  
  # compute a named vector of raw p‐values, one flag at a time
  pvals_lib <- sapply(keep_flags, function(flag) {
    in_sub <- comparison_df %>% filter(.data[[flag]]) %>% pull(log2ratio)
    # “rest” is everything _not_ in that subgroup:
    out_sub <- comparison_df %>% filter(! .data[[flag]]) %>% pull(log2ratio)
    wilcox.test(in_sub, out_sub)$p.value
  })
  pvals_lib <- p.adjust(pvals_lib, method = "BH")
  subgroup_vs_rest <- tibble::tibble(
    subgroup_flag = keep_flags,       # temporary holder of the internal names
    p.adj         = pvals_lib
  ) %>%
    mutate(
      group2 = all_names[subgroup_flag],           # map to your “pretty” names
      group1 = "Complete library*"
    ) %>%
    select(group1, group2, p.adj) %>%
    add_significance("p.adj") %>%
    # enforce the plotting order:
    mutate(group2 = factor(group2, levels = keep_order))
  
  
  combos <- combn(keep_flags, 2, simplify = FALSE)
  pair_p <- lapply(combos, function(pair) {
    f1 <- pair[1]; f2 <- pair[2]
    x <- comparison_df %>% filter(.data[[f1]]) %>% pull(log2ratio)
    y <- comparison_df %>% filter(.data[[f2]]) %>% pull(log2ratio)
    data.frame(
      group1 = f1,
      group2 = f2,
      p.raw  = wilcox.test(x, y)$p.value
    )
  })
  
  pairwise_subgroups <- bind_rows(pair_p) %>%
    mutate(p.adj = p.adjust(p.raw, method = "BH"),
           # map internal flag names → display names
           group1 = all_names[group1],
           group2 = all_names[group2]) %>%
    add_significance("p.adj") %>%
    select(group1, group2, p.adj, p.adj.signif) %>% 
    mutate(
      # enforce the desired plotting order
      group1 = factor(group1, levels = keep_order),
      group2 = factor(group2, levels = keep_order)
    ) %>% 
    bind_rows(subgroup_vs_rest)
  
  # Plot 1: boxplot
  base_cols <- RColorBrewer::brewer.pal(9, "Paired")
  names(base_cols) <- SUBGROUPS_ORDER[1:9]
  
  # build palette depending on which_subgroups
  if (which_subgroups == "default") {
    palette_vals <- base_cols[keep_order]
  } else if (which_subgroups == "added") {
    palette_vals <- subgroup_colors[names(subgroup_colors) %in% keep_order]
  } else {
    # all: combine default + added
    extra <- subgroup_colors[names(subgroup_colors) %in% keep_order]
    palette_vals <- c(base_cols, extra)[keep_order]
  }
  
  long_ratios <- comparison_df %>%
    pivot_longer(
      cols      = all_of(keep_flags),
      names_to  = "subgroup",
      values_to = "in_subgroup"
    ) %>%
    filter(in_subgroup) %>%   # now one row per (Peptide, subgroup) that *is* in that subgroup
    mutate(
      subgroup = factor(
        all_names[subgroup], levels = keep_order
      )
    )
  
  p1 <- ggplot(long_ratios, aes(subgroup, log2ratio)) +
    geom_violin(fill = "gray80", trim = T, adjust = 1, linewidth = 1.2)+#, draw_quantiles = c(0.25, 0.5, 0.75)) +    
    geom_jitter(aes(color = subgroup), width = 0.1, alpha = 0.5, size = 1) +
    stat_summary(fun = median, geom = "crossbar", color = "black", size = 0.8, fatten = 1) + #geom = "errorbar", aes(ymin = ..y.., ymax = ..y..) ) +
    stat_summary(fun = mean, geom = "point", color = "black", size = 1.5) +
    
    #geom_boxplot(aes(fill = subgroup), outlier.shape = 21, width = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray3") +
    scale_colour_manual(values = palette_vals, limits = keep_order) +
    scale_fill_manual(values = palette_vals, limits = keep_order) +
    guides(fill = "none", colour = "none") +
    labs(
      x = x_label,
      #y = paste("log₂-ratio of antibody responses\nin", group1, "and", group2, sep =" ")
      #y = paste("log₂-ratio of antibody responses\nin", group1, "and", group2, sep =" ")
      y = bquote(atop("log"[2] * "-ratio of antibody responses", 
                      "in" ~ .(group1) ~ "and" ~ .(group2)))
    ) + 
    theme_bw(base_size = 16) +
    theme(
      axis.text.x = element_text(angle = 35, hjust = 1), 
      #legend.position = "none",
      #plot.margin =  margin(2, 2, 2, 2),
      #axis.text.x = element_text(angle = 45, vjust = 0.6, hjust = 0.5),
      axis.text.x.bottom = element_text(size = 16),
      axis.text.y.left = element_text(size = 16),
      axis.title.y = element_text(size = 17),
      axis.title.x = element_text(size = 17),
      plot.margin = margin(0, 1, 0, 0, unit = "pt")
      )


  
  # Plot 2: heatmap
  p2 <- ggplot(pairwise_subgroups, aes(group1, group2, fill = p.adj.signif)) +
    geom_tile(color = "grey90", size = 0.2, show.legend = T) +
    geom_text(aes(label = sprintf("%.1g", p.adj)), size = 2.5, colour = "black") +
    
    scale_fill_manual(
      values = c(
        "****" = "#67001F",
        "***"  = "#B2182B",
        "**"   = "#D6604D",
        "*"    = "#F4A582",
        "ns"   = "dodgerblue1"
      ),
      limits = c("****","***","**","*","ns"),
      drop = FALSE,
      na.value = "white",
      name     = "Significance"
    ) +
  
  
    scale_x_discrete(limits = c("Complete library*", keep_order), position = "bottom") +
    scale_y_discrete(limits = rev(c("Complete library*", keep_order)), position = "left") +
    labs(y = NULL, x = paste(group1, "and", group2, sep=" ")) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      #plot.margin =  margin(0, 1, 0, 1),
      legend.position  = "right",
      legend.margin    = margin(0, 0, 0, -45),
      legend.box.margin = margin(0, 0, 0, -45),  # pull legend 10px left
      legend.key.size      = unit(15, "pt"),
      legend.text          = element_text(size = 15),
      legend.title         = element_text(size = 15),
      axis.text.x.bottom = element_text(size = 15),
      axis.text.y.left = element_text(size = 15),
      axis.title.y = element_text(size = 16),
      axis.title.x = element_text(size = 16),
      plot.margin = margin(0, 1, 0, 0, unit = "pt")
    )

  
  # Return each plot separately
  return(list(
    boxplot = p1,
    heatmap = p2
  ))
}




# Creates a Venn diagram for the same comparison
plot_ratio_venn <- function(
    comparison_df,
    group1,
    group2,
    custom_colors
) {
  set1 <- comparison_df %>% filter(!!sym(paste0(group1, "_count")) > 0) %>% pull(Peptide)
  set2 <- comparison_df %>% filter(!!sym(paste0(group2, "_count")) > 0) %>% pull(Peptide)
  venn_input <- setNames(list(set1, set2), c(group1, group2))
  p <- ggvenn(
    venn_input,
    fill_color    = c(custom_colors[group1][[1]], custom_colors[group2][[1]]),#custom_colors[c(group1, group2)],
    stroke_size   = 0.1,
    set_name_size = 3,
    text_size     = 3,
    auto_scale    = FALSE,
    show_outside  = "none"
  ) + theme(plot.margin = margin(2,2,2,2))
  return(p)
}



get_top_significant_taxa <- function(comparison_df, lineage_col, n = 8) {
  # turn the string into a symbol once
  lineage_sym <- sym(lineage_col)
  
  comparison_df %>%
    # drop rows where that lineage is missing
    filter(!is.na(!!lineage_sym)) %>%
    # make sure we have the same 'ratio' name
    mutate(ratio = log2(ratio)) %>%
    # get one row per lineage
    distinct(!!lineage_sym) %>%
    # for each lineage, pull out x = that group, y = all others
    mutate(
      x = purrr::map(!!lineage_sym, ~ comparison_df$ratio[comparison_df[[lineage_col]] == .x]),
      y = purrr::map(!!lineage_sym, ~ comparison_df$ratio[comparison_df[[lineage_col]] != .x]),
      p = purrr::map2_dbl(x, y, ~ wilcox.test(.x, .y)$p.value)
    ) %>%
    select(!!lineage_sym, p) %>%
    # FDR‐adjust
    adjust_pvalue(method = "BH") %>%
    arrange(p.adj) %>%
    # take the top n
    slice_head(n = n) %>%
    # pull out just the lineage names
    pull(!!lineage_sym) %>%
    # ---- CLEAN THE NAMES HERE ----
  str_remove_all("\\[|\\]") %>%  # drop any brackets
    str_squish()                  # collapse multiple spaces, trim
}

# Helper: generate flags_to_patterns and highlight_colors from top lineages
# • comparison_df: data.frame with at least column `lineage_col` and `ratio`
# • lineage_col: e.g. "species", "genus", etc.
# • n: how many top lineages to return
# • colors: optional named or unnamed vector of colors matching length n
# • brewer_palette, brewer_n: fallback palette settings
make_flag_lists_from_top_names <- function(
    comparison_df,
    lineage_col,
    n = 10,
    colors = NULL,
    brewer_palette = "Set3",
    brewer_n = 12
) {
  # 1) pull top-n lineages
  top_names <- get_top_significant_taxa(comparison_df, lineage_col, n)
  if (length(top_names) == 0) stop("No lineages found for ", lineage_col)
  
  # 2) derive short keys from first three words
  short_keys <- sapply(top_names, function(x) {
    words <- strsplit(x, "\\s+")[[1]]
    paste(head(words, 3), collapse = " ")
  })
  # ensure unique
  short_keys <- make.unique(short_keys)
  
  # 3) build flags_to_patterns list: key = short_key, value = full name
  flags_to_patterns <- setNames(as.list(top_names), short_keys)
  
  # 4) build highlight_colors vector
  if (!is.null(colors)) {
    if (length(colors) != length(top_names)) {
      stop("Length of 'colors' (", length(colors), ") must match number of flags (", length(top_names), ")")
    }
    highlight_colors <- setNames(colors, short_keys)
  } else {
    # fallback to brewer palette
    pal <- colorRampPalette(brewer.pal(brewer_n, brewer_palette))(length(top_names))
    highlight_colors <- setNames(pal, short_keys)
  }
  
  # return
  list(
    flags_to_patterns = flags_to_patterns,
    highlight_colors  = highlight_colors
  )
}

# 
# 
# plot_ratios_by_subgroup <- function(comparison_df,
#                               group1, group2,
#                               subgroup_lib_df, custom_colors,
#                               subgroup_colors = NULL,
#                               prevalence_threshold = 5,
#                               #min_peptides = 5, 
#                               add_subgroups = NULL) {
#   
# 
#   subgroups_to_include = SUBGROUPS_TO_INCLUDE
#   subgroups_order = SUBGROUPS_ORDER
#   subgroups_to_name = SUBGROUPS_TO_NAME
# 
#   
#   if (!is.null(add_subgroups)) {
#     for (flag in add_subgroups) {
#       subgroups_to_include = c(subgroups_to_include, flag)
#       subgroups_order = c(subgroups_order, flag)
#       subgroups_to_name[flag] = flag 
#     }
#   }
#   
#   # build the long table
#   long_ratios <- comparison_df %>%
#     mutate(ratio = log2(ratio)) %>% 
#     left_join(subgroup_lib_df, by = "Peptide") %>%
#     tidyr::pivot_longer(
#       cols      = all_of(subgroups_to_include),
#       names_to  = "subgroup_flag",
#       values_to = "in_subgroup"
#     ) %>%
#     dplyr::filter(in_subgroup) %>%
#     mutate(
#       subgroup = factor(
#         subgroups_to_name[subgroup_flag],
#         levels = subgroups_order
#       )
#     ) %>%
#     dplyr::filter(.data[[group1]] >= prevalence_threshold |
#                     .data[[group2]] >= prevalence_threshold) %>%
#     select(Peptide, subgroup, ratio)
#   
#   if(nrow(long_ratios) < 10) {
#     warning("Too few points for ", group1, " vs ", group2)
#     return(NULL)
#   }
#   
# 
#   tmp <- comparison_df %>%
#     mutate(ratio = log2(ratio)) %>% 
#     left_join(subgroup_lib_df, by = "Peptide") %>% 
#     rename(setNames(names(subgroups_to_name), subgroups_to_name))
#   
#   pvals <- sapply(subgroups_order, function(flag) {
#     x <- tmp %>% filter( !!sym(flag) ) %>% pull(ratio)
#     y <- tmp %>% filter(! (!!sym(flag)) ) %>% pull(ratio)
#     w <- wilcox.test(x, y)
#     w$p.value
#   })
#   rm(tmp)
#   pvals <- p.adjust(pvals, method = "BH")
#   subgroup_vs_rest <- tibble(
#     group2        = names(pvals),
#     group1        = "Complete library*",
#     p.adj         = pvals
#   ) %>%
#     add_significance("p.adj")
#   
#   pairwise_subgroups <- long_ratios %>%
#     pairwise_wilcox_test(ratio ~ subgroup, p.adjust.method = "BH") %>%  
#     add_significance("p.adj") %>%
#     select(all_of(c("group1", "group2", "p.adj", "p.adj.signif"))) %>% 
#     rbind(subgroup_vs_rest)
#   
#   
#   if (is.null(subgroup_colors)){
#     color_scale <- scale_fill_brewer(palette = "Paired") 
#   } else{
#     color_scale <- scale_color_manual(values = subgroup_colors)
#   }
#   
#   
# 
#   # grab the first up to 9 colours from Paired
#   base_cols <- RColorBrewer::brewer.pal(9, "Paired")
#   names(base_cols) <- subgroups_order[1:9]
#   
#   if (!is.null(subgroup_colors)) {
#     # only keep colours for subgroups we actually plot
#     extra <- subgroup_colors[names(subgroup_colors) %in% subgroups_order]
#     palette_vals <- c(base_cols, extra)[subgroups_order]
#   } else {
#     palette_vals <- base_cols[subgroups_order]
#   }
#   
#   p1 <- ggplot(long_ratios, aes(subgroup, ratio, fill = subgroup)) +
#     geom_boxplot(outlier.shape = 21, width = 0.6) +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "gray1") +
#     scale_fill_manual(values = palette_vals, limits = subgroups_order) +
#     theme_bw(base_size = 11) +
#     theme(
#       axis.text.x     = element_text(angle = 45, hjust = 1),
#       legend.position = "none"
#       ,    plot.margin =  margin(2, 2, 2, 2),
#     ) +
#     labs(
#       x = "Subgroups of the antigen library",
#       y = paste("log₂-ratio of antibody responses\nin", group1, "and", group2, sept=" ")
#     ) 
#   
#   p2 <- ggplot(pairwise_subgroups, aes(x = group1, y = group2, fill = p.adj.signif)) +
#     geom_tile(color = "grey90", linewidth = 0.2) +
#     geom_text(aes(label = sprintf("%.1g", p.adj)),
#               color = "black", size = 1.5) +
#     scale_fill_manual(
#       values = c(
#         "****"  = "#67001F",
#         "***"   = "#B2182B",
#         "**"    = "#D6604D",
#         "*"     = "#F4A582",
#         "ns"    = "dodgerblue1"
#       ),
#       na.value = "white",
#       name     = "Significance"
#     ) +
#     # Put x-axis on the bottom:
#     scale_x_discrete(
#       limits  = c("Complete library*",subgroups_order),
#       position = "bottom"
#     ) +
#     # Reverse y so first factor is at the top:
#     scale_y_discrete(
#       limits   = rev(c("Complete library*",subgroups_order)),
#       position = "left"
#     ) +
#     labs(x = NULL, y = NULL) +
#     #coord_fixed() +
#     theme_minimal(base_size = 11) +
#     theme(
#       aspect.ratio = 1,
#       panel.grid.major = element_blank(),
#       panel.grid.minor = element_blank(),
#       axis.text.x      = element_text(angle = 45, hjust = 1),
#       plot.margin =  margin(2, 2, 2, 2),
#       legend.position  = "right",
#       legend.margin    = margin(0, 0, 0, -25),
#       legend.box.margin = margin(0, 0, 0, -25),  # pull legend 10px left
#       legend.key.size  = unit(10, "pt"),
#       #legend.title     = element_text(size = 8),
#       #legend.text      = element_text(size = 7)
#     )
#   
#   # comparison_df has columns Peptide, <group1>_count, <group2>_count
#   g1_cnt <- paste0(group1, "_count")
#   g2_cnt <- paste0(group2, "_count")
#   
#   comparison_df %>%
#     dplyr::filter(.data[[group1]] >= prevalence_threshold |
#                     .data[[group2]] >= prevalence_threshold) 
#   
#   set1 <- comparison_df %>%
#     filter(!!sym(g1_cnt) >  0) %>%   # present in group1
#     pull(Peptide) %>% 
#     unique()
#   
#   set2 <- comparison_df %>%
#     filter(!!sym(g2_cnt) >  0) %>%   # present in group2
#     pull(Peptide) %>% 
#     unique()
#   
#   
#   # plot Venn with ggvenn
#   venn_input <- setNames(list(set1, set2), c(group1, group2))
#   p3 <- ggvenn(venn_input,
#          fill_color   = c(custom_colors[group1][[1]], custom_colors[group2][[1]]),
#          stroke_size  = 0.1,
#          set_name_size= 3,
#          text_size    = 3, 
#          auto_scale = F,
#          show_outside = "none") +
#     theme(
#       #aspect.ratio = 1,                   # keep the Venn itself square
#       plot.margin  = margin(0,0,0,0)   # top = 15px, right/bottom/left = 2px
#     )
#   
#   #combined <- (p1  | ( p3 / p2 + plot_layout(heights = c(0.95)) ))
#   combined <- (p3 + p1 + p2) +
#     plot_layout(
#       ncol   = 3,
#       widths = c(1.4, 1.7, 1.7),
#       #align  = "v"
#     )
#   return(combined)
#}


####################################################
################## MSA plot ########################
####################################################

make_flagged_msa_plot <- function(df,
                                  flag,
                                  log2_ratio_cut   = 1,
                                  pval_cut         = 0.05,
                                  seq_length       = 64,
                                  coords           = c(NULL, NULL),   # e.g. c(18, 71)
                                  msa_method       = "Muscle",
                                  msa_type         = "protein",
                                  msa_font         = "helvetic",
                                  msa_color_scheme = "Chemistry_AA",
                                  msa_char_width   = 0.5
) {
  
  # Filter
  df_hits <- df %>%
    mutate(log2ratio = log2(ratio)) %>%
    filter(
      !!sym(flag) == TRUE,
      (log2_ratio_cut >= 0 & log2ratio >  log2_ratio_cut) |
        (log2_ratio_cut <  0 & log2ratio <  log2_ratio_cut),
      pvals_not_adj <= pval_cut,
      nchar(aa_seq) >= seq_length
    )
  
  if (nrow(df_hits) < 2) {
    return("Not enough peptides passed the filters.")
  }
  
  # Build AAStringSet
  seqs <- setNames(df_hits$aa_seq, df_hits$Peptide)
  aa_set <- Biostrings::AAStringSet(seqs, use.names = TRUE)
  
  # MSA
  aln <- msa::msa(aa_set,
                  method = msa_method,
                  type   = msa_type,
                  order  = "input")
  class(aln) <- "AAMultipleAlignment"
  
  # Plot
  p <- ggmsa(
    aln,
    start = coords[1], end = coords[2],
    color    = msa_color_scheme,
    font = msa_font,
    char_width = msa_char_width, 
    seq_name = TRUE,
    consensus_views = TRUE,
    disagreement = FALSE,
    ignore_gaps = FALSE
  ) +
    geom_seqlogo(color = msa_color_scheme, font = msa_font, adaptive = TRUE) +
    geom_msaBar() 
  
  return (p)
}



##############################
##########TAXA LEVEL##########
##############################

# Function to generate count tables per taxonomic rank
generate_taxa_count_tables <- function(meta, exist_df) {
  meta <- meta %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "peptide")
  
  exist_df <- exist_df %>%
    as.data.frame() %>%
    tibble::rownames_to_column(var = "peptide")
  
  ranks <- c("domain", "kingdom", "phylum", "class", "order", "family", "genus", "species")
  common_peptides <- intersect(meta$peptide, exist_df$peptide)
  
  meta <- meta %>% filter(peptide %in% common_peptides)
  exist_df <- exist_df %>% filter(peptide %in% common_peptides)
  
  taxa_counts <- list()
  for (rank in ranks) {
    # filter out missing annotations
    df <- meta %>%
      select(peptide, !!sym(rank)) %>%
      filter(!is.na(!!sym(rank))) %>%
      inner_join(exist_df, by = "peptide") %>%
      group_by_at(rank) %>%
      summarise(across(-peptide, sum, .names = "{.col}"), .groups = "drop") %>%
      tibble::column_to_rownames(var = rank)
    
    taxa_counts[[rank]] <- df
  }
  
  return(taxa_counts)
}


normalize_taxa_counts <- function(ct, norm_method = c("relative","hellinger","log","none")){
  norm_method <- match.arg(norm_method)
  if (norm_method == "relative") {
    ct <- sweep(ct, 1, rowSums(ct), "/")
  } else if (norm_method == "hellinger") {
    ct <- decostand(ct, "hellinger")
  } else if (norm_method == "log") {
    ct <- log1p(ct)
  } 
  return(ct)
}


prep_abund_data <- function(ct, sample_meta, group_col = "Group", norm_method = c("relative","hellinger","log","none")) {
  
  # ensure samples × taxa and align to metadata
  mat <- if (all(colnames(ct) %in% sample_meta$SampleName)) t(ct) else ct
  
  sample_meta <- sample_meta %>%
    rename(Group = !!sym(group_col)) %>%
    filter(!is.na(Group))

  common <- intersect(sample_meta$SampleName, rownames(mat))
  mat         <- mat[common, , drop = FALSE]
  
  mat <- normalize_taxa_counts(mat, norm_method)
  
  
  list(mat = mat, sample_meta = sample_meta)
}

compute_alpha_diversity <- function(ct, sample_meta, group_col = "Group") {
  # Ensure samples x taxa
  mat <- if (all(colnames(ct) %in% sample_meta$SampleName)) t(ct) else ct
  # Compute metrics
  metrics <- data.frame(
    SampleName   = rownames(mat),
    Richness = vegan::specnumber(mat),
    "Shannon Diversity"  = vegan::diversity(mat, index = "shannon"),
    "Simpson Diversity"  = vegan::diversity(mat, index = "simpson"),
    check.names       = FALSE
  )
  # Merge metadata
  metrics <- merge(metrics, sample_meta, by.x = "SampleName", by.y = "SampleName",     check.names       = FALSE)
  # Rename grouping column to 'Group' for plotting consistency
  colnames(metrics)[which(names(metrics) == group_col)] <- "Group"
  return(metrics)
}


compute_beta_diversity <- function(ct, sample_meta, group_col = "Group", method = "bray", 
                                   norm_method = c("relative","hellinger","log", "none"), permutations = 1000) {
  prep  <- prep_abund_data(ct, sample_meta, group_col, norm_method)
  mat   <- prep$mat
  sample_meta  <- prep$sample_meta
  
  dist_mat <- vegan::vegdist(mat, method = method)
  pcoa <- cmdscale(dist_mat, eig = TRUE, k = 2)
  var_exp_all <- round(100 * pcoa$eig[1:2] / sum(pcoa$eig), 2)
  
  perm <- vegan::adonis2(dist_mat ~ Group, data = sample_meta, permutations = permutations)
  p_value <- perm$`Pr(>F)`[1]
  
  disp <- vegan::betadisper(dist_mat, sample_meta$Group)
  
  scores <- data.frame(
    SampleName = rownames(pcoa$points),
    PCoA1    = pcoa$points[,1],
    PCoA2    = pcoa$points[,2]
  )
  
  scores <- merge(scores, sample_meta, by.x = "SampleName", by.y = "SampleName")
  
  return(list(dist = dist_mat, permanova = perm, pcoa = scores, beta_dispersion = disp, exp_variance = var_exp_all))
}

# compute_pca <- function(ct, sample_meta,
#                         group_col   = "Group",
#                         norm_method = c("relative","hellinger","log","none")) {
#   
#   prep  <- prep_abund_data(ct, sample_meta, group_col, norm_method)
#   mat   <- prep$mat
#   samp  <- prep$sample_meta
#   
#   pca <- prcomp(mat, center = TRUE, scale. = FALSE)
#   var_exp <- round(100 * (pca$sdev^2 / sum(pca$sdev^2))[1:2], 2)
#   
#   scores <- data.frame(
#     SampleName = rownames(pca$x),
#     PC1 = pca$x[,1],
#     PC2 = pca$x[,2]
#   ) %>% merge(samp, by = "SampleName")
#   
#   
#   list(
#     scores = scores,
#     exp_variance   = var_exp
#   )
# }



## plots
plot_beta_dispersion <- function(disp, custom_colors, sig_level   = 0.05, label_format  = "p.format") {
  
  df <- data.frame(SampleName = names(disp$distances), Distance = disp$distances, Group = disp$group)
  
  p <- plot_groups_boxplots(data = df, 
                            group_col = "Group", 
                            values_col = "Distance",
                            custom_colors = custom_colors, 
                            sig_level   = sig_level,
                            label_format  = label_format,
                            label_axis = c("Group", "Distance to centroid"))
  
  return(p)
}


plot_pcoa <- function(beta_diversity, custom_colors, 
                      ellipse = T, permutations = 999, show_legend = T) {
  
  mds_scores <- beta_diversity$pcoa
  permanova  <- beta_diversity$perm
  disp       <- beta_diversity$beta_dispersion
  var_exp_all <- beta_diversity$exp_variance
  
  disp_perm  <- vegan::permutest(disp, permutations = permutations)
  centroids <- disp$centroids %>%
    as.data.frame() %>% 
    tibble::rownames_to_column(var="Group")
  
  
  # extract the raw p’s
  p_perm_val <- permanova$`Pr(>F)`[1]
  p_disp_val <- disp_perm$tab$`Pr(>F)`[1]
  
  # build your labels
  # lab_perm <- paste0("PERMANOVA p = ", format.pval(p_perm_val, digits = 1, eps = 0.01))
  # lab_disp <- paste0( "Dispersion   p = ", format.pval(p_disp_val, digits = 1, eps = 0.01))
  
  lab_perm <- paste0("PERMANOVA p = ", format_pval(p_perm_val))
  lab_disp <- paste0( "Dispersion   p = ", format_pval(p_disp_val))
  
  p <- ggplot(mds_scores, aes(x = PCoA1, y = PCoA2, fill = Group))
  
  # Conditionally add the ellipse
  if (ellipse) {
    p <- p + stat_ellipse(aes(colour = Group), type = "t", level = 0.95, fill = NA, geom = "path", size = 0.9, alpha = 0.8, show.legend = FALSE)
  }
  
  # Add all the remaining layers, making sure to use + at the end of each line
  p <- p +  geom_point(size = 3.5, alpha = 0.7, shape = 21, color = "black",  stroke = 0, aes(text = paste("Sample:", SampleName))) + #3 for fig1 and 4 for fig2
    geom_point(data = centroids, aes(x = PCoA1, y = PCoA2, fill = Group),
               show.legend = FALSE, size = 6, shape = 21, color = "black") + #5 for fig1 and 6 for fig2
    scale_fill_manual(values = custom_colors) +
    scale_color_manual(values = custom_colors) +
    guides(color = "none") +
    labs(
      x = paste0("PCoA 1 (", var_exp_all[1], "%)"),
      y = paste0("PCoA 2 (", var_exp_all[2], "%)")
      #,title = glue::glue(
      #   "PERMANOVA p={format(permanova$`Pr(>F)`[1], digits=2)}; dispersion p={format(disp_perm$tab$`Pr(>F)`[1],digits=2)}"
      # )
    ) +
    theme_bw(base_size = 13) +
    theme(
      legend.position     = c(0.99, 0.5),        # x=95% from left, y=50% from bottom
      legend.justification = c("right", "center"), 
      legend.background     = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      #legend.background = element_blank(),     # remove any grey behind the whole legend
      #legend.box.background = element_blank(),  # remove border around the legend box
      #legend.position  = "right",
      #legend.margin    = margin(0, 0, 0, -60),
      #legend.box.margin = margin(0, 0, 0, -60),  # pull legend 10px left
      legend.key.size      = unit(14, "pt"),
      legend.text          = element_text(size = 13),
      legend.title         = element_text(size = 13),
      
      axis.text.y.left = element_text(size = 13), #13 in fig1 15 ing fig2
      axis.text.x.bottom = element_text(size = 13),
      axis.title.y = element_text(size = 13), #13 in fig1 16 in fig2
      axis.title.x = element_text(size = 13, margin = margin(t = 1)),
      plot.margin = margin(2, 4, 2, 4, unit = "pt")    
    ) +
    
    annotate(
      "text",
      x = Inf,           # right edge
      y = -Inf,          # bottom edge
      label = paste(lab_perm, lab_disp, sep = "\n"),
      hjust = 1.1,       # nudge left a bit
      vjust = -0.1,      # nudge up a bit
      size = 4           # adjust to taste
    )
  
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  
  return(p)
}



# Given the output of prep_Rank_stats(), makes the ggplot + annotations
plot_taxa_violin <- function(prep_out, ncol = 3, nudge_y = -1) {
  p <- ggplot(prep_out$rel_abundance_filtered, aes(x = Group, y = Abundance)) +
    geom_violin(fill = "gray80") +
    geom_jitter(aes(color = Rank), width = 0.2, alpha = 0.6, size = 1) +
    stat_summary(fun = median, geom = "crossbar", color = "black", size = 0.5, fatten = 1) + #geom = "errorbar", aes(ymin = ..y.., ymax = ..y..) ) +
    stat_summary(fun = mean, geom = "point", color = "black", size = 1.5) +
    scale_color_manual(values = prep_out$rank_colors) +
    #scale_y_log10(expand = expansion(mult = c(0, 0.11))) +
    scale_y_continuous(
      trans  = "log10", 
      #labels = percent_format(accuracy = 1),   # “0.01”→“1%”, “0.1”→“10%”
      expand = expansion(mult = c(0, 0.11))  # 5% below, 20% above
    ) +
    facet_wrap(~ Rank, ncol = ncol)+#, scales = "free_y") +
    theme_bw(base_size = 12) +
    theme(
      legend.position  = "none",
      axis.text.x      = element_text(angle = 35, hjust = 1),
      plot.margin      = margin(t = 0, r = 2, b = 0, l = 0),
      strip.background = element_blank(),#element_rect(fill = "white", colour = "black"),
      strip.text = element_text( size = 11, colour = "black"),
      #panel.grid.major = element_line(color = "grey90", linetype = "solid"),
      #panel.grid.minor  = element_blank(),
      axis.text.y.left = element_text(size = 12),
      axis.text.x.bottom = element_text(size = 12),
      axis.title.y = element_text(size = 13),
      axis.title.x = element_text(size = 12, margin = margin(t = 0)),
    ) +
    

    labs(
      x = NULL,
      y = expression("log"[10]~"(Relative Abundance)")
    ) 
  p + stat_pvalue_manual(
    data          = prep_out$pval_df,
    label         = "p.adj.label",
    #xmin         = "xmin",
    #xmax         = "xmax",
    y.position    = "y.position",
    tip.length    = 0.02,
    bracket.size  = 0.25,
    size          = 3.6,
    inherit.aes   = FALSE,
    step.increase = 0.1,
    step.group.by = "Rank",
    bracket.nudge.y = nudge_y
  )
}

# 
# plot_pca_abundance <- function(pca, custom_colors, 
#                                ellipse = T,  show_legend = T) {
#   
#   scores <- pca$scores
#   var_exp_all <- pca$exp_variance
#   
#   
#   p <- ggplot(scores, aes(x = PC1, y = PC2, fill = Group))
#   
#   # Conditionally add the ellipse
#   if (ellipse) {
#     p <- p + stat_ellipse(aes(colour = Group), type = "t", level = 0.95, fill = NA, geom = "path", size = 0.8, alpha = 0.5, show.legend = FALSE)
#   }
#   
#   p <- p +  geom_point(size = 3, alpha = 0.5, shape = 21, color = "black",  stroke = 0, aes(text = paste("Sample:", SampleName))) +
#     scale_fill_manual(values = custom_colors) +
#     scale_colour_manual(values = custom_colors) +
#     guides(color = "none") +
#     labs(
#       x = paste0("PC 1 (", var_exp_all[1], "%)"),
#       y = paste0("PC 2 (", var_exp_all[2], "%)")
#       #,title = glue::glue(
#       #   "PERMANOVA p={format(permanova$`Pr(>F)`[1], digits=2)}; dispersion p={format(disp_perm$tab$`Pr(>F)`[1],digits=2)}"
#       # )
#     ) +
#     theme_bw() +
#     theme(
#       plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
#     ) 
#   
#   if (!show_legend) {
#     p <- p + theme(legend.position = "none")
#   }
#   
#   return(p)
# }



# Given:
#  • rank            — e.g. "species" (must exist in counts_list)
#  • group_col       — grouping variable name in metadata
#  • counts_list     — named list of count matrices
#  • metadata        — data.frame with SampleName + all group_cols
#  • abundance_cutoff = 0.01
#  • min_samples      = 5
#  • top_n            = how many species to keep
#  • order_by         = which level of Group to use for ordering (or "All" for overall)
# Returns a list with:
#   df_long,            # full long format used for stats/plot
#   pval_df,            # Dunn’s test results with xmin/xmax/y.position
#   species_colors      # named vector for species colors
prep_taxa_stats <- function(
    rank,
    group_col,
    counts_list,
    metadata,
    drop_taxa = T,
    abundance_cutoff = 0.01,
    min_samples      = 5,
    top_n            = 12,
    order_by         = "All"    # "All" or a specific group level
) {
  # pull & clean counts
  counts <- counts_list[[rank]] 
  
  if(drop_taxa & rank  == "species"){
    counts <- counts %>%
      tibble::rownames_to_column("Species") %>%
      dplyr::filter(Species != "Homo sapiens") %>% #let's ignore homo sapiens for now
      tibble::column_to_rownames("Species")
  }
  
  metadata  <- metadata %>% 
    select(SampleName=SampleName, group_col) %>% rename(Group = !!sym(group_col))  
  
  # relative abundance & filter low-prevalence
  rel <- normalize_taxa_counts(t(counts), "relative")
  keep <- colSums(rel > abundance_cutoff) >= min_samples
  rel <- rel[, keep, drop = FALSE]
  
  # pivot into long & join metadata
  df_long <- dplyr::as_tibble(rel, rownames = "SampleName") %>%
    tidyr::pivot_longer(-SampleName, names_to = "Rank", values_to = "Abundance") %>%
    dplyr::left_join(metadata, by = "SampleName") %>%
    dplyr::filter(!is.na(Group)) %>%
    dplyr::mutate(Abundance = tidyr::replace_na(Abundance, 0))
  
  # Kruskal–Wallis per taxa
  # kw_res <- df_long %>%
  #   group_by(Rank) %>%
  #   kruskal_test(Abundance ~ Group) %>%
  #   adjust_pvalue(method = "BH") %>%
  #   filter(p.adj <= 0.1)  # keep only taxa with significant omnibus
  # 
  #  Dunn post-hoc only on those taxa
  
  # Dunn’s test + position
  pval_df <- df_long %>%
    #filter(Species %in% kw_res$Species) %>%
    dplyr::group_by(Rank) %>%
    rstatix::dunn_test(Abundance ~ Group, p.adjust.method = "BH") %>%
    rstatix::add_xy_position(x = "Group") %>%
    dplyr::filter(p.adj < 0.05) %>%
    dplyr::mutate(p.adj.label = signif(p.adj, 2)) %>%
    dplyr::ungroup()
  
  # pick top_n Rank
  unique_sp <- unique(pval_df$Rank)
  n_keep    <- min(top_n, length(unique_sp))
  top_Rank <- unique_sp[seq_len(n_keep)]
  
  # filter df_long to these Rank
  df_long_filtered <- df_long %>%
    dplyr::filter(Rank %in% top_Rank)
  
  # determine ordering
  ordering_data <- if (order_by != "All") {
    df_long_filtered %>% dplyr::filter(Group == order_by)
  } else {
    df_long_filtered
  }
  
  ordering <- ordering_data %>%
    dplyr::group_by(Rank) %>%
    dplyr::summarise(mean_ab = mean(Abundance, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(desc(mean_ab)) %>%
    dplyr::pull(Rank)
  
  df_long_filtered <- df_long_filtered %>%
    dplyr::mutate(Rank = factor(Rank, levels = ordering))
  
  pval_df <- pval_df %>%
    dplyr::filter(Rank %in% ordering) %>% 
    dplyr::mutate(Rank = factor(Rank, levels = ordering))
  
  # assign colors
  
  Rank_colors <- colorRampPalette(brewer.pal(12, "Set3"))(length(unique(df_long_filtered$Rank)))
  names(Rank_colors) <- unique(df_long_filtered$Rank)  # Map names to colors
  
  
  
  list(
    rel_abundance        = df_long, 
    rel_abundance_filtered = df_long_filtered,
    pval_df              = pval_df,
    rank_colors       = Rank_colors
  )
}


