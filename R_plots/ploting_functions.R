# ==============================================================================
# plots.R
#
# Shared helper library for the functions used by:
#   - compare_concurrency_plots.R 
#   - disease_comparison_plots.R: uses metric_compare_multi()

# Packages used by both scripts
library(dplyr)
library(ggplot2)
library(purrr)
library(readr)
library(arrow)

# Constants used by both scripts
PALETTE_SEX <- c("Male" = "#144d85", "Female" = "#681c10")

# Age group, orientation, and sex orders
AGE_GROUP_ORDER <- c("16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75", "Unknown")
ORI_ORDER       <- c("Opposite-sex", "Same-sex", "Bisexual")
SEX_ORDER       <- c("Male", "Female")

# Partner count bins, in PARTNERS (one timestep == one day). Only used by
# compare_concurrency_plots.R's build_counts_with_zeros() 
PARTNER_BINS <- list(
  c(0, 0), c(1, 1), c(2, 2), c(3, 4), c(5, 9), c(10, Inf)
)

# Bin labels for the above PARTNER_BINS, in the same order. Used by both
# compare_concurrency_plots.R and disease_comparison_plots.R.
BIN_LABELS <- c("0", "1", "2", "3\u20134", "5\u20139", "10+")

# Duration bins, in DAYS (one timestep == one day). Only used by
# compare_concurrency_plots.R's build_duration_bins_df() 
DURATION_BINS <- list(
  c(0,    90),    # <3 months  (~90 days)
  c(90,   180),   # 3-6 months (~91-180 days)
  c(180,  365),   # 6-12 months
  c(365,  730),   # 1-2 years
  c(730,  1825),  # 2-5 years
  c(1825, Inf)    # 5+ years
)
DURATION_BIN_LABELS <- c("<3mo", "3\u20136mo", "6\u201312mo", "1\u20132yr", "2\u20135yr", "5yr+")

# Publication-style theme for ggplot2 plots, used by both scripts. Can be called with a custom font family (default is "sans").
apply_publication_style <- function(font = "sans") {
  theme_set(
    theme_minimal(base_size = 14, base_family = font) +
      theme(
        plot.title       = element_text(face = "bold", size = 15, margin = margin(b = 6)),
        plot.subtitle    = element_text(size = 13, margin = margin(b = 10)),
        axis.title       = element_text(size = 14),
        axis.text        = element_text(size = 12),
        axis.line        = element_line(colour = "grey40", linewidth = 0.6),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = "white", colour = NA),
        legend.position  = "top",
        legend.title     = element_text(size = 12, face = "bold"),
        legend.text      = element_text(size = 12),
        legend.background = element_rect(fill = NA),
        strip.text       = element_text(face = "bold", size = 13),
        strip.background = element_rect(fill = "#ecf0f1", colour = NA)
      )
  )
}

# Reads input data from a CSV or Parquet file, returning a data frame. 
read_input_data <- function(filepath) {
  ext <- tolower(tools::file_ext(filepath))
  if (ext == "csv") {
    message(sprintf("[INFO] Reading CSV: %s", filepath))
    return(read_csv(filepath, show_col_types = FALSE))
  } else if (ext == "parquet") {
    message(sprintf("[INFO] Reading Parquet: %s", filepath))
    return(read_parquet(filepath))
  } else {
    stop(sprintf("Unsupported file type: %s. Expected CSV or Parquet.", ext))
  }
}

# Checks if a directory exists, and creates it (including parent directories) if it doesn't. Returns the path invisibly.
check_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

# Saves a ggplot object to a PNG file, and optionally to a PDF file. 
save_plot <- function(p, filepath_no_ext, width = 12, height = 8, save_pdf = FALSE) {
  check_dir(dirname(filepath_no_ext))
  ggsave(paste0(filepath_no_ext, ".png"),
         plot = p, width = width, height = height, dpi = 300, bg = "white")
  if (save_pdf) {
    ggsave(paste0(filepath_no_ext, ".pdf"),
           plot = p, width = width, height = height, dpi = 300, bg = "white")
  }
  invisible(p)
}


# Helpers for preparing data for plotting, used by both scripts. 
# These functions are designed to handle NA values.

# Assign a bin label based on the number of partners (count). 
# Returns the corresponding label from BIN_LABELS. 
assign_bin <- function(count) {
  for (i in seq_along(PARTNER_BINS)) {
    bin <- PARTNER_BINS[[i]]
    if (!is.na(count) && count >= bin[1] && count <= bin[2]) return(BIN_LABELS[i])
  }
  return(BIN_LABELS[length(BIN_LABELS)])
}

# Assign a duration bin label based on the duration in days.
assign_duration_bin <- function(d) {
  for (i in seq_along(DURATION_BINS)) {
    b <- DURATION_BINS[[i]]
    if (!is.na(d) && d >= b[1] && d < b[2]) return(DURATION_BIN_LABELS[i])
  }
  return(DURATION_BIN_LABELS[length(DURATION_BIN_LABELS)])
}

# Add age group to the data frame if it's missing, deriving it from AgentAge if necessary. Returns the modified data frame.
add_age_group_if_missing <- function(df) {
  if ("AgentAgeGroup" %in% names(df) && any(!is.na(df$AgentAgeGroup))) {
    return(df)
  }
  if (!"AgentAge" %in% names(df)) {
    warning("[WARNING] Neither AgentAgeGroup nor AgentAge found.")
    return(df)
  }
  message("[INFO] Deriving AgentAgeGroup from AgentAge.")
  df %>% mutate(
    AgentAgeGroup = cut(
      as.numeric(AgentAge),
      breaks = c(15, 24, 34, 44, 54, 64, 74, Inf),
      labels = c("16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75"),
      right  = TRUE
    ) %>% as.character()
  )
}

# Standardise age group labels to broader categories. Returns the modified age group label.
standardise_agegroup <- function(x) {
  case_when(
    x %in% c("16-20", "21-25") ~ "16-24",
    x %in% c("26-30", "31-35") ~ "25-34",
    x %in% c("36-40", "41-45") ~ "35-44",
    x %in% c("46-50", "51-55") ~ "45-54",
    x %in% c("56-60", "61-65") ~ "55-64",
    x %in% c("66-70", "71-75") ~ "65-74",
    TRUE ~ x
  )
}

# Builds a data frame of partner counts, including agents with zero partners. Returns the modified data frame.
build_counts_with_zeros <- function(df) {
  
  # Derive AgentAgeGroup if not present
  df <- add_age_group_if_missing(df)
  
  if (!"EverPartnered" %in% colnames(df)) {
    df <- df %>% mutate(EverPartnered = RelationshipType != "None")
  }

  has_agent_age <- "AgentAge" %in% colnames(df)

  # Build a data frame of partnered agents with their partner counts and demographics
  partnered <- df %>%
    filter(!is.na(PartnerAgent)) %>%
    group_by(Agent) %>%
    summarise(
      PartnerCount     = n_distinct(PartnerAgent),
      AgentSex         = first(AgentSex),
      AgentOrientation = first(AgentOrientation),
      AgentAgeGroup    = first(AgentAgeGroup),
      AgentAge         = if (has_agent_age) first(AgentAge) else NA_real_,
      .groups = "drop"
    )

  # Build a data frame of agents who have never partnered, with zero partner counts and demographics
  never_cols <- c("Agent", "AgentSex", "AgentOrientation", "AgentAgeGroup")
  if (has_agent_age) never_cols <- c(never_cols, "AgentAge")

  # Never-partnered agents: distinct by demographics, with PartnerCount = 0 
  never <- df %>%
    filter(EverPartnered == FALSE) %>%
    distinct(across(all_of(never_cols))) %>%
    mutate(
      PartnerCount = 0L,
      AgentAge     = if (has_agent_age) AgentAge else NA_real_
    )

  # Combine partnered and never-partnered agents, standardise age groups, and assign bins
  counts <- bind_rows(partnered, never) %>%
    distinct() %>%
    mutate(
      AgentAgeGroup = standardise_agegroup(AgentAgeGroup),
      AgentAgeGroup = factor(AgentAgeGroup, levels = AGE_GROUP_ORDER, ordered = TRUE),
      BinLabel      = factor(sapply(PartnerCount, assign_bin),
                             levels = BIN_LABELS, ordered = TRUE)
    )

  n_unknown <- sum(is.na(counts$AgentAgeGroup))
  if (n_unknown > 0)
    warning(sprintf("[WARNING] %d agents have NA/Unknown age group", n_unknown))

  return(counts)
}

# Builds a data frame of partnership durations, assigning duration bins and standardising age groups.
# Returns the modified data frame.
build_duration_bins_df <- function(df) {
  if (!"Duration" %in% colnames(df)) {
    message("[WARNING] No Duration column"); return(NULL)
  }

  dur <- df %>%
    filter(!is.na(Duration), Duration > 0) %>%
    distinct(Agent, PartnerAgent, Duration,
             AgentSex, AgentOrientation, AgentAgeGroup) %>%
    mutate(
      AgentAgeGroup    = standardise_agegroup(AgentAgeGroup),
      AgentAgeGroup    = factor(AgentAgeGroup, levels = AGE_GROUP_ORDER),
      AgentOrientation = factor(AgentOrientation, levels = ORI_ORDER),
      AgentSex         = factor(AgentSex, levels = SEX_ORDER),
      DurationBin      = factor(sapply(Duration, assign_duration_bin),
                                levels = DURATION_BIN_LABELS)
    ) %>%
    filter(!is.na(AgentAgeGroup), !AgentAgeGroup %in% c("75", "Unknown"))

  if (nrow(dur) == 0) return(NULL)
  dur
}


# Metric engine helpers for disease_comparison_plots.R. These functions compute metrics by age group, sex, and orientation, 
# and are used to generate plots comparing different scenarios.

CELL_COLS <- c("AgentAgeGroup", "AgentSex", "AgentOrientation")

# Orders the cells in a data frame by age group, sex, and orientation. Returns the modified data frame.
order_cells <- function(df) {
  df %>%
    mutate(
      AgentAgeGroup    = factor(AgentAgeGroup,    levels = AGE_GROUP_ORDER),
      AgentOrientation = factor(AgentOrientation, levels = ORI_ORDER),
      AgentSex         = factor(AgentSex,         levels = SEX_ORDER)
    )
}

# Computes a metric for each cell (age group, sex, orientation) using the provided metric function. 
#Returns a data frame with the computed values.
compute_metric_cells <- function(df, metric_fn) {
  out <- metric_fn(df)
  stopifnot(all(c(CELL_COLS, "Value") %in% names(out)))
  out
}

# Computes a metric for each simulation in a list of data frames using the provided metric function.
# Returns a data frame with the computed values and a SimulationID column.
compute_metric_persim <- function(df_list, metric_fn) {
  map_df(seq_along(df_list), function(i) {
    compute_metric_cells(df_list[[i]], metric_fn) %>% mutate(SimulationID = i)
  })
}

# Applies a consistent theme for metric comparison plots, with the legend at the bottom and x-axis text 
metric_theme_bits <- function() {
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1))
}

# Compares a metric across multiple groups (scenarios) and generates bar and box plots.
metric_compare_multi <- function(df_list_by_group, metric_fn, title, y_lab,
                                  group_alpha, output_dir, filename,
                                  group_name = "Scenario", font = "sans", save_pdf = FALSE) {
  apply_publication_style(font); check_dir(output_dir)

  persim <- map_df(names(df_list_by_group), function(g) {
    compute_metric_persim(df_list_by_group[[g]], metric_fn) %>% mutate(Group = g)
  }) %>%
    order_cells() %>%
    mutate(Group = factor(Group, levels = names(df_list_by_group)))

  summary <- persim %>%
    group_by(across(all_of(c(CELL_COLS, "Group")))) %>%
    summarise(Mean = mean(Value, na.rm = TRUE), SD = sd(Value, na.rm = TRUE),
              N_sims = n(), .groups = "drop") %>%
    order_cells() %>%
    mutate(Group = factor(Group, levels = names(df_list_by_group)))

  dodge <- position_dodge(width = 0.9)

  p_bar <- ggplot(summary, aes(x = AgentAgeGroup, y = Mean, fill = AgentSex,
                               alpha = Group, group = interaction(AgentSex, Group))) +
    geom_bar(stat = "identity", position = dodge, colour = "white", linewidth = 0.4) +
    geom_errorbar(aes(ymin = pmax(Mean - SD, 0), ymax = Mean + SD),
                  position = dodge, width = 0.2, colour = "grey40") +
    scale_fill_manual(values = PALETTE_SEX, name = "Sex") +
    scale_alpha_manual(values = group_alpha, name = group_name) +
    facet_wrap(~AgentOrientation, nrow = 1) +
    labs(title = paste0(title, " \u2014 mean \u00b1 SD"), x = "Age group", y = y_lab) +
    guides(fill = guide_legend(order = 1, override.aes = list(alpha = 0.9)),
           alpha = guide_legend(order = 2, override.aes = list(fill = "grey50"))) +
    metric_theme_bits()
  save_plot(p_bar, file.path(output_dir, paste0(filename, "_bar")),
            width = 14, height = 5, save_pdf = save_pdf)

  p_box <- ggplot(persim, aes(x = AgentAgeGroup, y = Value, fill = AgentSex,
                              alpha = Group, group = interaction(AgentAgeGroup, AgentSex, Group))) +
    geom_boxplot(position = dodge, colour = "grey30", linewidth = 0.4,
                 outlier.shape = NA, width = 0.8) +
    geom_point(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.9),
               size = 0.8, alpha = 0.4, colour = "grey20", show.legend = FALSE) +
    scale_fill_manual(values = PALETTE_SEX, name = "Sex") +
    scale_alpha_manual(values = group_alpha, name = group_name) +
    facet_wrap(~AgentOrientation, nrow = 1) +
    labs(title = paste0(title, " \u2014 per-simulation distribution"),
         x = "Age group", y = y_lab,
         caption = "Box: IQR across simulations. Points: individual simulations.") +
    guides(fill = guide_legend(order = 1, override.aes = list(alpha = 0.9)),
           alpha = guide_legend(order = 2, override.aes = list(fill = "grey50"))) +
    metric_theme_bits()
  save_plot(p_box, file.path(output_dir, paste0(filename, "_box")),
            width = 14, height = 5, save_pdf = save_pdf)

  invisible(list(persim = persim, summary = summary, bar = p_bar, box = p_box))
}


# Metrics for disease_comparison_plots.R. 
# Each function takes a data frame and returns a data frame with columns: AgentAgeGroup, AgentSex, AgentOrientation, Value.

# Percentage of agents ever infected, by age group, sex, and  orientation. Returns a data frame with the computed values.
metric_pct_infected <- function(df) {
  df %>%
    group_by(AgentAgeGroup = age_group, AgentSex = sex, AgentOrientation = orientation) %>%
    summarise(Value = mean(ever_infected) * 100, .groups = "drop")
}

# Average number of times infected, by age group, sex, and orientation. Returns a data frame with the computed values.
metric_times_infected <- function(df) {
  df %>%
    group_by(AgentAgeGroup = age_group, AgentSex = sex, AgentOrientation = orientation) %>%
    summarise(Value = mean(times_infected), .groups = "drop")
}

# Average reinfection burden (times infected) among those ever infected, by age group, sex, and orientation. Returns a data frame with the computed values.
metric_reinfection_burden <- function(df) {
  df %>%
    filter(ever_infected) %>%
    group_by(AgentAgeGroup = age_group, AgentSex = sex, AgentOrientation = orientation) %>%
    summarise(Value = mean(times_infected), .groups = "drop")
}