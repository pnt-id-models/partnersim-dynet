# ==============================================================================
# compare_concurrency_plots.R 
# Compares partnership count distributions, durations, and
# single-agent / high-partner proportions between a no-concurrency and a
# 15% concurrency scenario.
#
# Produces four plots (one per orientation), saved as separate files:
#   - Mean partnership count by age group and sex (error bars = SD)
#   - Mean partnership duration by age group and sex
#   - % single agents (PartnerCount == 0)
#   - % agents with 10+ concurrent/lifetime partners
#
# Usage:
#   Rscript R_plots/compare.R \
#       --dir_conc_0   /home/unimelb.edu.au/pillaip/STI-Modelling-Project/devMay2026/partnersim-dynet/examples/output/sweep/full_sweep_0pcconcurrency_15000agents_18Aug2026_#3 \
#       --dir_conc_15  /home/unimelb.edu.au/pillaip/STI-Modelling-Project/devMay2026/partnersim-dynet/examples/output/sweep/full_sweep_15pcconcurrency_15000agents_18Aug2026_#11 \
#       --output       /home/unimelb.edu.au/pillaip/STI-Modelling-Project/devMay2026/partnersim-dynet/examples/output/sweep/full_sweep_0pcconcurrency_15000agents_18Aug2026_#3/comparison_25Aug_final
# ==============================================================================

# Suppress package startup messages for cleaner output
suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(purrr)
  library(arrow)
  library(readr)
  library(gridExtra)
  library(patchwork)
  library(tools)
})

# Shared script directory resolution for sourcing plots.R, even when run via Rscript
.script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile, mustWork = TRUE)),
  error = function(e) getwd()
)
source(file.path(.script_dir, "R_plots/ploting_functions.R"))

# CLI options
option_list <- list(
  make_option("--dir_conc_0",
    type = "character", default = NULL,
    help = "Directory containing parquet files for no-concurrency scenario"),
  make_option("--dir_conc_15",
    type = "character", default = NULL,
    help = "Directory containing parquet files for 15% concurrency scenario"),
  make_option("--output",
    type = "character", default = "comparison_plots",
    help = "Output directory [default: comparison_plots]"),
  make_option("--save_pdf",
    action = "store_true", default = FALSE,
    help = "Also save PDF copies"),
  make_option("--high_partner_threshold",
    type = "integer", default = 10,
    help = "Partner count threshold for 'high partner' proportion [default: 10]")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$dir_conc_0) || is.null(opt$dir_conc_15))
  stop("[ERROR] Both --dir_conc_0 and --dir_conc_15 are required.")

check_dir(opt$output)

# Scenario labels for legend and plot titles. These are used in the
# SCENARIO_ALPHA/SCENARIO_COLOUR mappings below, so changing them will also
# require updating those mappings.
SCENARIO_LABELS <- c(
  "No concurrency"  = "No concurrency",
  "15% concurrency" = "15% concurrency"
)

# Fill alpha, colour, and line width for each scenario. 
SCENARIO_ALPHA   <- c("No concurrency" = 0.85, "15% concurrency" = 0.55)
SCENARIO_COLOUR  <- c("No concurrency" = "white", "15% concurrency" = "#333333")
SCENARIO_LINEW   <- c("No concurrency" = 0.4, "15% concurrency" = 1.0)

SCENARIO_LEVELS <- c("No concurrency", "15% concurrency")

# Load parquet files from a directory
resolve_results_dir <- function(dir_path, pattern) {
  candidates <- c(dir_path)

  repo_root <- tryCatch(
    normalizePath(file.path(.script_dir, ".."), mustWork = TRUE),
    error = function(e) NULL
  )

  if (!is.null(repo_root)) {
    repo_name <- basename(repo_root)
    repo_prefix <- paste0(repo_name, "/")
    if (startsWith(dir_path, repo_prefix))
      candidates <- c(candidates, file.path(repo_root, substr(dir_path, nchar(repo_prefix) + 1, nchar(dir_path))))

    candidates <- c(candidates, file.path(repo_root, dir_path))
  }

  for (candidate in unique(candidates)) {
    if (dir.exists(candidate) && length(list.files(candidate,
                                                 pattern = pattern,
                                                 full.names = TRUE,
                                                 recursive = TRUE)) > 0)
      return(normalizePath(candidate, mustWork = TRUE))
  }

  dir_path
}

load_parquets <- function(dir_path) {
  pattern <- "partnerships.*\\.parquet$"
  dir_path <- resolve_results_dir(dir_path, pattern)
  files <- list.files(dir_path,
                      pattern = pattern,
                      full.names = TRUE, recursive = TRUE)
  if (length(files) == 0)
    stop(sprintf("[ERROR] No partnership parquet files found in: %s", dir_path))

  message(sprintf("[INFO] Loading %d file(s) from %s", length(files), dir_path))
  map(files, ~ read_input_data(.x) %>% add_age_group_if_missing())
}

# Ordering of factor levels for age groups, sex, and orientation. 
# These are used in order_strata() to ensure consistent ordering across all plots.
order_strata <- function(df) {
  df %>%
    mutate(
      AgentAgeGroup    = factor(AgentAgeGroup, levels = AGE_GROUP_ORDER),
      AgentOrientation = factor(AgentOrientation, levels = ORI_ORDER),
      AgentSex         = factor(AgentSex, levels = SEX_ORDER),
      Scenario         = factor(Scenario, levels = SCENARIO_LEVELS)
    )
}

# Collect per-simulation summaries and then collapse to mean ± SD across simulations
summarise_across_sims <- function(df, value_col) {
  df %>%
    group_by(AgentAgeGroup, AgentSex, AgentOrientation, Scenario) %>%
    summarise(
      Mean = mean(.data[[value_col]], na.rm = TRUE),
      SD   = sd(.data[[value_col]],   na.rm = TRUE),
      .groups = "drop"
    ) %>%
    order_strata()
}

# Build per-simulation mean partnership counts by age group, sex, and orientation. 
#This is used for both the bar+errorbar plots and the boxplots.
build_mean_counts <- function(df_list, scenario_label) {
  map_df(seq_along(df_list), function(i) {
    counts <- build_counts_with_zeros(df_list[[i]]) %>%
      filter(!AgentAgeGroup %in% c("75", "Unknown"), !is.na(AgentAgeGroup))

    counts %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
      summarise(MeanCount = mean(PartnerCount, na.rm = TRUE), .groups = "drop") %>%
      mutate(SimulationID = i, Scenario = scenario_label)
  })
}

# Build per-simulation partnership duration summaries, binned by duration.
build_duration_summary <- function(df_list, scenario_label) {
  map_df(seq_along(df_list), function(i) {
    dur <- build_duration_bins_df(df_list[[i]])
    if (is.null(dur)) return(tibble())

    dur %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation, DurationBin) %>%
      summarise(Count = n(), .groups = "drop") %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
      mutate(
        Total      = sum(Count),
        Percentage = if_else(Total > 0, Count / Total * 100, 0)
      ) %>%
      ungroup() %>%
      mutate(SimulationID = i, Scenario = scenario_label)
  })
}

# Build per-simulation mean partnership duration by age group, sex, and orientation.
build_mean_duration_by_age <- function(df_list, scenario_label) {
  map_df(seq_along(df_list), function(i) {
    dur <- build_duration_bins_df(df_list[[i]])
    if (is.null(dur)) return(tibble())

    dur %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
      summarise(MeanDuration = mean(Duration, na.rm = TRUE),
                .groups = "drop") %>%
      mutate(SimulationID = i, Scenario = scenario_label)
  })
}

# # ── Build per-agent waiting time between successive partnerships ──────────────
# # Mirrors the dedup/standardisation pattern used by build_duration_bins_df()
# # and get_concurrent_episodes() elsewhere in this file: the raw partnership
# # rows are not yet one-row-per-partnership, so we distinct() on the relevant
# # columns first. Age groups are standardised the same way as everywhere else.
# #
# # Negative gaps (overlapping partnerships, expected under concurrency) are
# # dropped rather than treated as zero -- this means the 15% concurrency
# # scenario's mean waiting time is computed only over each agent's
# # non-overlapping transitions, which will tend to bias it upward relative to
# # a "true" population-wide gap. Worth flagging in any write-up.
# WAITING_TIME_REQUIRED_COLS <- c("Agent", "PartnerAgent", "StartTime", "EndTime",
#                                 "AgentAgeGroup", "AgentSex", "AgentOrientation")

# build_waiting_times <- function(df) {
#   df <- add_age_group_if_missing(df)

#   if (!all(WAITING_TIME_REQUIRED_COLS %in% names(df))) {
#     warning("[WARN] build_waiting_times: missing required column(s) (",
#             paste(setdiff(WAITING_TIME_REQUIRED_COLS, names(df)), collapse = ", "),
#             ") -- skipping waiting time calculation for this simulation.")
#     return(NULL)
#   }

#   partnerships <- df %>%
#     filter(!is.na(PartnerAgent), !is.na(StartTime)) %>%
#     distinct(Agent, PartnerAgent, StartTime, EndTime,
#              AgentSex, AgentOrientation, AgentAgeGroup) %>%
#     mutate(
#       StartTime     = as.integer(StartTime),
#       EndTime       = as.integer(EndTime),
#       AgentAgeGroup = standardise_agegroup(AgentAgeGroup)
#     ) %>%
#     filter(!AgentAgeGroup %in% c("75", "Unknown"), !is.na(AgentAgeGroup))

#   if (nrow(partnerships) == 0) return(NULL)

#   partnerships %>%
#     arrange(Agent, StartTime) %>%
#     group_by(Agent, AgentAgeGroup, AgentSex, AgentOrientation) %>%
#     mutate(
#       PrevEndTime = lag(EndTime),
#       WaitingTime = StartTime - PrevEndTime
#     ) %>%
#     ungroup() %>%
#     filter(!is.na(WaitingTime), WaitingTime >= 0)
# }

# build_mean_waiting_by_age <- function(df_list, scenario_label) {
#   map_df(seq_along(df_list), function(i) {
#     wt <- build_waiting_times(df_list[[i]])
#     if (is.null(wt) || nrow(wt) == 0) return(tibble())

#     wt %>%
#       group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
#       summarise(MeanWaiting = mean(WaitingTime, na.rm = TRUE),
#                 .groups = "drop") %>%
#       mutate(SimulationID = i, Scenario = scenario_label)
#   })
# }

# Build per-simulation proportion of single agents (PartnerCount == 0) by age group, sex, and orientation.
build_single_proportion <- function(df_list, scenario_label) {
  map_df(seq_along(df_list), function(i) {
    counts <- build_counts_with_zeros(df_list[[i]]) %>%
      filter(!AgentAgeGroup %in% c("75", "Unknown"), !is.na(AgentAgeGroup))

    counts %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
      summarise(
        N           = n(),
        SingleCount = sum(PartnerCount == 0, na.rm = TRUE),
        PctSingle   = if_else(N > 0, SingleCount / N * 100, NA_real_),
        .groups = "drop"
      ) %>%
      mutate(SimulationID = i, Scenario = scenario_label)
  })
}

# Build per-simulation proportion of agents with PartnerCount >= threshold by age group, sex, and orientation
build_high_partner_proportion <- function(df_list, scenario_label, threshold = 10) {
  map_df(seq_along(df_list), function(i) {
    counts <- build_counts_with_zeros(df_list[[i]]) %>%
      filter(!AgentAgeGroup %in% c("75", "Unknown"), !is.na(AgentAgeGroup))

    counts %>%
      group_by(AgentAgeGroup, AgentSex, AgentOrientation) %>%
      summarise(
        N              = n(),
        HighCount      = sum(PartnerCount >= threshold, na.rm = TRUE),
        PctHighPartner = if_else(N > 0, HighCount / N * 100, NA_real_),
        .groups = "drop"
      ) %>%
      mutate(SimulationID = i, Scenario = scenario_label)
  })
}

# Load data, build summaries, and generate plots for both scenarios. Each plot is saved as a separate file per orientation.
message("[INFO] Loading no-concurrency scenario...")
df_list_0  <- load_parquets(opt$dir_conc_0)

message("[INFO] Loading 15% concurrency scenario...")
df_list_15 <- load_parquets(opt$dir_conc_15)

message("[INFO] Building partnership count summaries...")
counts_persim <- bind_rows(
    build_mean_counts(df_list_0,  "No concurrency"),
    build_mean_counts(df_list_15, "15% concurrency")
  ) %>%
  order_strata()
counts_all <- counts_persim %>% summarise_across_sims("MeanCount")

message("[INFO] Building duration bin summaries...")
dur_all <- bind_rows(
    build_duration_summary(df_list_0,  "No concurrency"),
    build_duration_summary(df_list_15, "15% concurrency")
  ) %>%
  group_by(AgentAgeGroup, AgentSex, AgentOrientation, DurationBin, Scenario) %>%
  summarise(
    MeanPct = mean(Percentage, na.rm = TRUE),
    SdPct   = sd(Percentage,   na.rm = TRUE),
    .groups = "drop"
  ) %>%
  order_strata()

message("[INFO] Building mean duration-by-age summaries...")
dur_age_persim <- bind_rows(
    build_mean_duration_by_age(df_list_0,  "No concurrency"),
    build_mean_duration_by_age(df_list_15, "15% concurrency")
  ) %>%
  order_strata()
dur_age_all <- dur_age_persim %>% summarise_across_sims("MeanDuration")

# message("[INFO] Building waiting time summaries...")
# wait_age_persim <- bind_rows(
#     build_mean_waiting_by_age(df_list_0,  "No concurrency"),
#     build_mean_waiting_by_age(df_list_15, "15% concurrency")
#   )

# if (nrow(wait_age_persim) == 0) {
#   message("[WARN] No waiting time data available -- panel will be skipped.")
#   wait_age_all <- wait_age_persim
# } else {
#   wait_age_persim <- wait_age_persim %>% order_strata()
#   wait_age_all    <- wait_age_persim %>% summarise_across_sims("MeanWaiting")
# }

message("[INFO] Building single-agent proportion summaries...")
single_persim <- bind_rows(
    build_single_proportion(df_list_0,  "No concurrency"),
    build_single_proportion(df_list_15, "15% concurrency")
  ) %>%
  order_strata()
single_all <- single_persim %>% summarise_across_sims("PctSingle")

message("[INFO] Building high-partner proportion summaries...")
high_persim <- bind_rows(
    build_high_partner_proportion(df_list_0,  "No concurrency",
                                  threshold = opt$high_partner_threshold),
    build_high_partner_proportion(df_list_15, "15% concurrency",
                                  threshold = opt$high_partner_threshold)
  ) %>%
  order_strata()
high_all <- high_persim %>% summarise_across_sims("PctHighPartner")

# Waiting-time summaries and plots are disabled in this script.
HAS_WAITING_DATA <- FALSE

# Shared theme for all comparison plots, with larger text and bold axis titles 
apply_publication_style()

comparison_theme <- theme(
  legend.position  = "top",
  legend.title     = element_text(size = 16, face = "bold"),
  legend.text      = element_text(size = 16),
  axis.text.x      = element_text(angle = 45, hjust = 1, size = 16),
  axis.text.y      = element_text(hjust = 1, size = 16),
  axis.title.x     = element_text(size = 16, face = "bold"),
  axis.title.y     = element_text(size = 16, face = "bold"),
  axis.label.x     = element_text(size = 16, face = "bold"),
  axis.label.y     = element_text(size = 16, face = "bold"),
  strip.text       = element_blank(),
  plot.title       = element_blank()
)

dodge <- position_dodge(width = 0.85)

# Generic bar+errorbar panel builder for mean ± SD plots. 
make_bar_panel <- function(data, title, y_lab, x_lab = NULL, caption = NULL,
                            show_legend = TRUE) {
  y_max <- suppressWarnings(max(data$Mean + data$SD, na.rm = TRUE)) * 1.15
  if (!is.finite(y_max)) y_max <- 1  # guard against all-NA data

  p <- ggplot(data,
              aes(x     = AgentAgeGroup,
                  y     = Mean,
                  fill  = AgentSex,
                  alpha = Scenario,
                  group = interaction(AgentSex, Scenario))) +
    geom_bar(stat     = "identity",
             position = dodge,
             colour   = "white",
             linewidth = 0.4) +
    geom_errorbar(aes(ymin = pmax(Mean - SD, 0),
                      ymax = Mean + SD),
                  position  = dodge,
                  width     = 0.5,
                  linewidth = 0.75,
                  colour    = "#2d2d2d") +
    scale_fill_manual(values = PALETTE_SEX, name = "Sex") +
    scale_alpha_manual(
      values = c("No concurrency" = 0.40, "15% concurrency" = 0.90),
      name   = "Scenario",
      labels = c(
        "No concurrency"  = SCENARIO_LABELS[["No concurrency"]],
        "15% concurrency" = SCENARIO_LABELS[["15% concurrency"]]
      )
    ) +
    scale_y_continuous(limits = c(0, y_max), expand = c(0, 0)) +
    labs(title = title, x = x_lab, y = y_lab, caption = caption) +
    guides(
      fill  = guide_legend(order = 1, override.aes = list(alpha = 0.9)),
      alpha = guide_legend(order = 2, override.aes = list(fill = "grey50"))
    ) +
    facet_wrap(~AgentSex, nrow = 1) +
    comparison_theme

  if (!show_legend) p <- p + theme(legend.position = "none")
  p
}

# Save a single panel for a given orientation, with optional PDF output.
save_orientation_panel <- function(plot, output_dir, ori, panel_name, save_pdf = FALSE) {
  ori_clean <- tolower(gsub("[-]", "_", ori))
  fname <- file.path(output_dir, paste0("comparison_", ori_clean, "_", panel_name))

  ggsave(paste0(fname, ".png"),
         plot = plot, width = 14, height = 7,
         dpi = 300, bg = "white", limitsize = FALSE)

  if (save_pdf)
    ggsave(paste0(fname, ".pdf"),
           plot = plot, width = 14, height = 7,
           dpi = 300, bg = "white", limitsize = FALSE)

  message(sprintf("  [OK] Saved: %s.png", fname))
}

# Make comparison plot panels for a given orientation, returning a list of ggplot objects.
make_comparison_plot_panels <- function(ori) {

  p_count <- make_bar_panel(
    data        = filter(counts_all, AgentOrientation == ori),
    title       = paste0(ori, " \u2014 Mean partnership count by age group"),
    y_lab       = "Mean partnership count",
    show_legend = TRUE
  )

  p_dur <- make_bar_panel(
    data        = filter(dur_age_all, AgentOrientation == ori),
    title       = paste0(ori, " \u2014 Mean partnership duration by age group"),
    y_lab       = "Mean duration (timesteps)",
    show_legend = FALSE
  )

  p_single <- make_bar_panel(
    data        = filter(single_all, AgentOrientation == ori),
    title       = paste0(ori, " \u2014 % single agents (0 partners) by age group"),
    y_lab       = "% single agents",
    show_legend = FALSE
  )

  p_high <- make_bar_panel(
    data        = filter(high_all, AgentOrientation == ori),
    title       = paste0(ori, " \u2014 % agents with ",
                         opt$high_partner_threshold, "+ partners"),
    y_lab       = paste0("% with ", opt$high_partner_threshold, "+ partners"),
    show_legend = FALSE
  )

  list(
    count  = p_count,
    duration = p_dur,
    single = p_single,
    high   = p_high
  )
}

# Generate and save comparison plots for each orientation, with optional PDF output.
panel_height <- 7

for (ori in ORI_ORDER) {
  message(sprintf("[INFO] Generating comparison plot for: %s", ori))
  panels <- make_comparison_plot_panels(ori)

  save_orientation_panel(panels$count,    opt$output, ori, "count",    save_pdf = opt$save_pdf)
  save_orientation_panel(panels$duration, opt$output, ori, "duration", save_pdf = opt$save_pdf)
  save_orientation_panel(panels$single,   opt$output, ori, "single",   save_pdf = opt$save_pdf)
  save_orientation_panel(panels$high,     opt$output, ori, "high",     save_pdf = opt$save_pdf)
}

message("[DONE] Bar+errorbar comparison plots written to: ", opt$output)

# Build boxplots for per-simulation distributions of each metric, with one panel per metric. 

message("[INFO] Building boxplot panels (per-simulation distributions)...")

box_dodge <- position_dodge(width = 0.85)

# Boxplot panel builder for per-simulation distributions. Each point is one simulation's cell mean.
make_box_panel <- function(data, value_col, title, y_lab, x_lab = NULL,
                            caption = NULL, show_legend = TRUE,
                            show_points = TRUE) {

  p <- ggplot(data,
              aes(x     = AgentAgeGroup,
                  y     = .data[[value_col]],
                  fill  = AgentSex,
                  alpha = Scenario,
                  group = interaction(AgentAgeGroup, AgentSex, Scenario))) +
    geom_boxplot(position  = box_dodge,
                 colour    = "grey30",
                 linewidth = 0.4,
                 outlier.shape = NA,  # avoid double-plotting outliers if points shown
                 width     = 0.75)

  if (show_points) {
    p <- p + geom_point(
      position = position_jitterdodge(jitter.width = 0.12,
                                      dodge.width   = 0.85),
      size = 0.9, alpha = 0.45, colour = "grey20", show.legend = FALSE
    )
  }

  p <- p +
    scale_fill_manual(values = PALETTE_SEX, name = "Sex") +
    scale_alpha_manual(
      values = c("No concurrency" = 0.40, "15% concurrency" = 0.90),
      name   = "Scenario",
      labels = c(
        "No concurrency"  = SCENARIO_LABELS[["No concurrency"]],
        "15% concurrency" = SCENARIO_LABELS[["15% concurrency"]]
      )
    ) +
    labs(title = title, x = x_lab, y = y_lab, caption = caption) +
    guides(
      fill  = guide_legend(order = 1, override.aes = list(alpha = 0.9)),
      alpha = guide_legend(order = 2, override.aes = list(fill = "grey50"))
    ) +
    facet_wrap(~AgentSex, nrow = 1) +
    comparison_theme

  if (!show_legend) p <- p + theme(legend.position = "none")
  p
}

# Make a combined boxplot figure for a given orientation, returning the ggplot object and number of panels. 
# Each panel is one of the four metrics (count, duration, single proportion, high-partner proportion)
make_comparison_boxplot <- function(ori) {

  p_count <- make_box_panel(
    data        = filter(counts_persim, AgentOrientation == ori),
    value_col   = "MeanCount",
    title       = paste0(ori, " \u2014 Partnership count by age group (per-sim distribution)"),
    y_lab       = "Mean partnership count (per sim)",
    show_legend = TRUE
  )

  p_dur <- make_box_panel(
    data        = filter(dur_age_persim, AgentOrientation == ori),
    value_col   = "MeanDuration",
    title       = paste0(ori, " \u2014 Partnership duration by age group (per-sim distribution)"),
    y_lab       = "Mean duration (timesteps, per sim)",
    show_legend = FALSE
  )

  p_single <- make_box_panel(
    data        = filter(single_persim, AgentOrientation == ori),
    value_col   = "PctSingle",
    title       = paste0(ori, " \u2014 % single agents by age group (per-sim distribution)"),
    y_lab       = "% single agents (per sim)",
    show_legend = FALSE
  )

  p_high <- make_box_panel(
    data        = filter(high_persim, AgentOrientation == ori),
    value_col   = "PctHighPartner",
    title       = paste0(ori, " \u2014 % agents with ",
                         opt$high_partner_threshold,
                         "+ partners (per-sim distribution)"),
    y_lab       = paste0("% with ", opt$high_partner_threshold, "+ partners (per sim)"),
    show_legend = FALSE
  )

  panels <- list(p_count, p_dur, p_single, p_high)

  if (HAS_WAITING_DATA) {
    p_wait <- make_box_panel(
      data        = filter(wait_age_persim, AgentOrientation == ori),
      value_col   = "MeanWaiting",
      title       = paste0(ori, " \u2014 Waiting time between partnerships (per-sim distribution)"),
      y_lab       = "Mean waiting time (timesteps, per sim)",
      x_lab       = "Age group",
      caption     = "Each point/box value is one simulation's cell mean. Waiting time excludes overlapping (concurrent) gaps.",
      show_legend = FALSE
    )
    panels <- c(panels, list(p_wait))
  } else {
    panels[[length(panels)]] <- panels[[length(panels)]] +
      labs(caption = paste0("Each point/box value is one simulation's cell mean ",
                            "(n = ", length(df_list_0), " vs ", length(df_list_15), " sims)."))
  }

  combined <- reduce(panels, `/`)

  final_plot <- combined +
    plot_annotation(
      title    = NULL,
      subtitle = paste0(
        "Box = IQR across simulations  |  Points = individual simulations  |  ",
        "Transparent: \u03b8\u2090\u2091\u2099\u2090 = 0  |  Opaque: \u03b8\u2090\u2091\u2099\u2090 = 0.15"
      ),
      theme = theme(
        plot.subtitle = element_text(size = 18, colour = "grey40")
      )
    )

  list(plot = final_plot, n_panels = length(panels))
}

# Generate and save boxplot comparison plots for each orientation, with optional PDF output.
for (ori in ORI_ORDER) {
  message(sprintf("[INFO] Generating boxplot comparison for: %s", ori))
  res <- make_comparison_boxplot(ori)
  p_box <- res$plot
  n_panels <- res$n_panels

  ori_clean <- tolower(gsub("[-]", "_", ori))
  fname_box <- file.path(opt$output,
                         paste0("comparison_", ori_clean, "_boxplot"))

  ggsave(paste0(fname_box, ".png"),
         plot = p_box, width = 14, height = panel_height * n_panels,
         dpi = 300, bg = "white", limitsize = FALSE)

  if (opt$save_pdf)
    ggsave(paste0(fname_box, ".pdf"),
           plot = p_box, width = 14, height = panel_height * n_panels,
           dpi = 300, bg = "white", limitsize = FALSE)

  message(sprintf("  [OK] Saved: %s.png", fname_box))
}

message("[DONE] Boxplot comparison plots written to: ", opt$output)  


