
# ==============================================================================
# disease_comparison_plots.R
#
#   Short version that only produces a subset of plots and csvs
#
# CONCURRENCY STATUS is derived from three raw agent-log categories
# ("Monogamous", "Eligible, not concurrent", "Behaviourally concurrent").
# The plots always use the 2-category EffectiveStatus ("Monogamous" vs.
# "Polygamous"); the CSVs are written for both the 2-category and full
# 3-category breakdowns. Controlled by:
#
#   --eligible_as_monogamous
#       If SET:   "Eligible, not concurrent" agents are folded into
#                 "Monogamous" in every 2-category output.
#       If UNSET (default): they are excluded from every 2-category output
#                 entirely. The 3-category CSVs are unaffected by this flag.
# ──────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(optparse)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  library(purrr)
  library(readr)
  library(patchwork)
  library(forcats)
  library(knitr)
  library(stringr)
})

.script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile, mustWork = TRUE)),
  error = function(e) getwd()
)
source(file.path(.script_dir, "R_plots/ploting_functions.R"))

option_list <- list(
  make_option("--base_dir",
    type = "character", default = NULL,
    help = "Base sweep output directory"),
  make_option("--dir_conc_0",
    type = "character", default = NULL,
    help = "Subdirectory name for no-concurrency replicates"),
  make_option("--dir_conc_15",
    type = "character", default = NULL,
    help = "Subdirectory name for 15% concurrency replicates"),
  make_option("--output",
    type = "character", default = "disease_plots",
    help = "Output directory [default: disease_plots]"),
  make_option("--save_pdf",
    action = "store_true", default = FALSE,
    help = "Also save PDF copies"),
  make_option("--eligible_as_monogamous",
    action = "store_true", default = FALSE,
    help = paste(
      "If set, agents eligible for concurrency but who never actually had",
      "overlapping partnerships are grouped into 'Monogamous'.",
      "If unset (default), they are excluded from all outputs entirely."
    ))
)

opt <- parse_args(OptionParser(option_list = option_list))

message(sprintf(
  "[INFO] Eligible-but-inactive agents will be %s.",
  if (opt$eligible_as_monogamous) "MERGED INTO Monogamous" else "EXCLUDED"
))

AGE_GROUP_ORDER <- c("16-24", "25-34", "35-44", "45-54", "55-64", "65-74")
ORI_ORDER       <- c("Opposite-sex", "Same-sex", "Bisexual")
SEX_ORDER       <- c("Male", "Female")

DISEASE_SCENARIO_ALPHA <- c("No concurrency" = 0.40, "15% concurrency" = 0.90)

# Raw 3-level classification order (used before collapsing to 2 categories).
STATUS_LEVELS_3CAT_RAW <- c("Monogamous", "Eligible, not concurrent", "Behaviourally concurrent")

# Display order/labels for the 3-category CSVs ("Behaviourally concurrent" is
# relabeled to "Polygamous" for terminology consistency with the 2-category outputs
STATUS_LEVELS_3CAT_DISPLAY <- c("Monogamous", "Eligible, not concurrent", "Polygamous")

STATUS_LEVELS_2CAT  <- c("Monogamous", "Polygamous")
STATUS_PALETTE_2CAT <- c("Monogamous" = "#4393c3", "Polygamous" = "#d6604d")

ELIGIBLE_NOTE <- if (opt$eligible_as_monogamous) {
  "Eligible-but-inactive agents are included in Monogamous."
} else {
  "Eligible-but-inactive agents are excluded from this comparison."
}

dir.create(opt$output, showWarnings = FALSE, recursive = TRUE)

# Concurrency status classification (per-agent, per-run) -- ConcurrencyStatus
classify_concurrency_status <- function(ever_concurrent, concurrency_allowed) {
  factor(
    case_when(
      ever_concurrent                          ~ "Behaviourally concurrent",
      concurrency_allowed & !ever_concurrent   ~ "Eligible, not concurrent",
      !concurrency_allowed & !ever_concurrent  ~ "Monogamous"
    ),
    levels = STATUS_LEVELS_3CAT_RAW
  )
}

# Adds EffectiveStatus (2-category) to a data frame that already has ConcurrencyStatus (3-category).
add_effective_status <- function(df, status_col = "ConcurrencyStatus") {
  df %>%
    mutate(
      EffectiveStatus = case_when(
        .data[[status_col]] == "Behaviourally concurrent" ~ "Polygamous",
        .data[[status_col]] == "Monogamous"                ~ "Monogamous",
        .data[[status_col]] == "Eligible, not concurrent" & opt$eligible_as_monogamous ~ "Monogamous",
        TRUE ~ NA_character_
      ),
      EffectiveStatus = factor(EffectiveStatus, levels = STATUS_LEVELS_2CAT)
    )
}

# Load disease replicates from a given scenario directory, resolving the full path based on the base directory 
# and scenario subdirectory. Returns a data frame with all runs and their associated data.
resolve_disease_root <- function(base_dir, scenario_dir) {
  base_tail <- basename(dirname(base_dir))
  sweep_tail <- basename(base_dir)
  tail_marker <- paste0(base_tail, "/", sweep_tail, "/")

  candidates <- unique(c(
    scenario_dir,
    file.path(base_dir, scenario_dir)
  ))

  marker_pos <- regexpr(tail_marker, scenario_dir, fixed = TRUE)
  if (marker_pos[1] > 0) {
    scenario_suffix <- substring(scenario_dir,
                                marker_pos[1] + nchar(tail_marker),
                                nchar(scenario_dir))
    if (nzchar(scenario_suffix))
      candidates <- unique(c(candidates, file.path(base_dir, scenario_suffix)))
  }

  if (grepl("/", scenario_dir, fixed = TRUE)) {
    candidates <- unique(c(
      candidates,
      file.path(base_dir, basename(scenario_dir))
    ))
  }

  # Expected structure: either a replicate directory (replicate_*) or a disease seed directory (disease_seed_*) directly under the scenario root.
  has_expected_structure <- function(path) {
    if (!dir.exists(path))
      return(FALSE)

    child_dirs <- list.dirs(path, recursive = FALSE, full.names = TRUE)
    child_names <- basename(child_dirs)
    any(grepl("^disease_seed_", child_names)) || any(grepl("^replicate_", child_names))
  }

  structured_candidates <- Filter(has_expected_structure, candidates)
  if (length(structured_candidates) > 0)
    return(normalizePath(structured_candidates[[1]], mustWork = TRUE))

  for (candidate in candidates) {
    if (dir.exists(candidate))
      return(normalizePath(candidate, mustWork = TRUE))
  }

  normalizePath(file.path(base_dir, scenario_dir), mustWork = FALSE)
}

# Load disease replicates from a given scenario directory, resolving the full path based on the base directory and scenario subdirectory. 
#Returns a data frame with all runs and their associated data.
load_disease_replicates <- function(base_dir, scenario_dir, scenario_label) {
  full_path <- resolve_disease_root(base_dir, scenario_dir)
  cat("base_dir    :", base_dir, "\n")
  cat("scenario_dir:", scenario_dir, "\n")
  cat("full_path   :", full_path, "\n")
  cat("exists      :", dir.exists(full_path), "\n")

  if (grepl("replicate_", basename(full_path))) {
    rep_dirs <- full_path
  } else {
    rep_dirs <- list.dirs(full_path, recursive = FALSE, full.names = TRUE)
    rep_dirs <- rep_dirs[grepl("replicate_", basename(rep_dirs))]
  }

  all_data <- map_df(seq_along(rep_dirs), function(i) {
    seed_dirs <- list.dirs(rep_dirs[i], recursive = FALSE, full.names = TRUE)
    seed_dirs <- seed_dirs[grepl("disease_seed_", basename(seed_dirs))]
    cat("Replicate:", rep_dirs[i], "\n")
    cat("Found", length(seed_dirs), "seed directories\n")

    if (length(seed_dirs) == 0) {
      message(sprintf("[WARN] No disease seeds in: %s", rep_dirs[i]))
      return(tibble())
    }

    map_df(seq_along(seed_dirs), function(j) {
      f <- file.path(seed_dirs[j], "demographic_summary.csv")
      if (!file.exists(f)) {
        message(sprintf("[WARN] Missing: %s", f))
        return(tibble())
      }
      read_csv(f, show_col_types = FALSE) %>%
        mutate(
          PartnershipRunID = i,
          DiseaseRunID     = j,
          RunID            = paste0("p", i, "_d", j),
          SeedDir          = basename(seed_dirs[j]),
          Scenario         = scenario_label
        )
    })
  })

  dup_check <- all_data %>%
    group_by(RunID, agent_id) %>%
    filter(n() > 1)

  if (nrow(dup_check) > 0)
    warning(sprintf("[WARN] %d duplicate agent rows detected", nrow(dup_check)))

  message(sprintf("  [INFO] Loaded %d unique runs from %s",
                  n_distinct(all_data$RunID), scenario_dir))
  all_data
}

# Store all replicates in a single data frame, with Scenario labels for each scenario.
message("[INFO] Loading no-concurrency replicates...")
df_0  <- load_disease_replicates(opt$base_dir, opt$dir_conc_0,  "No concurrency")

message("[INFO] Loading 15% concurrency replicates...")
df_15 <- load_disease_replicates(opt$base_dir, opt$dir_conc_15, "15% concurrency")

df_all <- bind_rows(df_0, df_15) %>%
  mutate(
    Scenario    = factor(Scenario, levels = c("No concurrency", "15% concurrency")),
    age_group   = factor(age_group,   levels = AGE_GROUP_ORDER),
    orientation = factor(orientation, levels = ORI_ORDER),
    sex         = factor(sex,         levels = SEX_ORDER)
  )

# Build a table of agent counts per run to check for duplicates. 
# If any agent appears more than once in the same run, raise an error.
agent_counts <- df_all %>%
  group_by(Scenario, RunID, agent_id) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n > 1)

if (nrow(agent_counts) > 0) {
  stop(sprintf("[ERROR] %d agents appear more than once in the same run. ", nrow(agent_counts)))
} else {
  message(sprintf(
    "[OK] No duplicate agents. Total runs: %d (%d no-conc, %d 15%%-conc)",
    n_distinct(df_all$RunID),
    n_distinct(df_all$RunID[df_all$Scenario == "No concurrency"]),
    n_distinct(df_all$RunID[df_all$Scenario == "15% concurrency"])))
}

message(sprintf("[INFO] Loaded %d agents total (%d no-conc, %d 15%%-conc)",
                nrow(df_all), nrow(df_0), nrow(df_15)))

df_list_by_run_0  <- df_all %>% filter(Scenario == "No concurrency")   %>% group_split(RunID)
df_list_by_run_15 <- df_all %>% filter(Scenario == "15% concurrency") %>% group_split(RunID)
DISEASE_SCENARIO_GROUPS <- list("No concurrency" = df_list_by_run_0, "15% concurrency" = df_list_by_run_15)

# Shared theme for all comparison plots, with larger text and bold axis titles
base_theme <- theme_minimal(base_size = 18) +
  theme(
    plot.title       = element_text(face = "bold", size = 22),
    plot.subtitle    = element_text(size = 15, colour = "grey40"),
    axis.title       = element_text(size = 17, face = "bold"),
    axis.text        = element_text(size = 15),
    axis.text.x      = element_text(angle = 45, hjust = 1),
    legend.position  = "top",
    legend.title     = element_text(face = "bold", size = 16),
    legend.text      = element_text(size = 15),
    legend.key.size  = unit(1.1, "lines"),
    strip.text       = element_text(face = "bold", size = 16),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background  = element_rect(fill = "white", colour = NA),
    plot.caption     = element_text(size = 12, colour = "grey50")
  )

dodge <- position_dodge(width = 0.9)

save_disease_plot <- function(p, name, width = 14, height = 9) {
  ggsave(file.path(opt$output, paste0(name, ".png")),
         plot = p, width = width, height = height, dpi = 300, bg = "white")
  if (opt$save_pdf)
    ggsave(file.path(opt$output, paste0(name, ".pdf")),
           plot = p, width = width, height = height, bg = "white")
  message(sprintf("  [OK] %s.png", name))
}

# Concurrency status classification (per-agent, per-run) -- ConcurrencyStatus
conc_status_all <- df_all %>%
  mutate(ConcurrencyStatus = classify_concurrency_status(ever_concurrent, concurrency_allowed)) %>%
  add_effective_status("ConcurrencyStatus")

# Concurrency status prevalence breakdown table, at four levels of breakdown
message("[INFO] Building concurrency-status prevalence breakdown table...")

# Concurrency prevalence table for the 15% concurrency scenario only, at four levels of breakdown.
conc_table_base <- conc_status_all %>% filter(Scenario == "15% concurrency")

# Summarise the prevalence of concurrency status (3-category) or effective status (2-category) at a given level of breakdown, averaging across all runs.
summarise_conc_level <- function(data, group_vars, level_label, status_col) {
  out <- data %>%
    filter(!is.na(.data[[status_col]])) %>%
    group_by(across(all_of(c("RunID", status_col, group_vars)))) %>%
    summarise(PctInfected = mean(ever_infected) * 100, .groups = "drop") %>%
    group_by(across(all_of(c(status_col, group_vars)))) %>%
    summarise(Mean_PctInfected = mean(PctInfected), SD_PctInfected = sd(PctInfected),
              N_runs = n(), .groups = "drop")
  names(out)[names(out) == status_col] <- "ConcurrencyStatus"
  out %>%
    mutate(
      ConcurrencyStatus = as.character(ConcurrencyStatus),
      Level = level_label,
      CategorySystem = if (status_col == "ConcurrencyStatus") "3-category" else "2-category",
      .before = 1
    )
}

# Concurrency prevalence table, at four levels of breakdown, for both 3-category and 2-category status systems.
concurrency_prevalence_raw <- bind_rows(
  lapply(c("ConcurrencyStatus", "EffectiveStatus"), function(status_col) {
    bind_rows(
      summarise_conc_level(conc_table_base, character(0), "Overall", status_col),
      summarise_conc_level(conc_table_base, "sex", "By sex", status_col),
      summarise_conc_level(conc_table_base, c("sex", "orientation"), "By sex x orientation", status_col),
      summarise_conc_level(conc_table_base, c("sex", "orientation", "age_group"),
                            "By sex x orientation x age group", status_col)
    )
  })
)

# Finalise the prevalence table for output, filtering by category system and formatting columns.
finalise_prevalence_table <- function(raw, category_system) {
  status_levels <- if (category_system == "3-category") STATUS_LEVELS_3CAT_DISPLAY else STATUS_LEVELS_2CAT

  raw %>%
    filter(CategorySystem == category_system) %>%
    mutate(
      Level = factor(Level, levels = c("Overall", "By sex", "By sex x orientation",
                                        "By sex x orientation x age group")),
      ConcurrencyStatus = if (category_system == "3-category") {
        dplyr::recode(ConcurrencyStatus, "Behaviourally concurrent" = "Polygamous")
      } else {
        ConcurrencyStatus
      },
      ConcurrencyStatus = factor(ConcurrencyStatus, levels = status_levels),
      sex         = if ("sex" %in% names(.))         factor(sex, levels = SEX_ORDER)         else NA,
      orientation = if ("orientation" %in% names(.)) factor(orientation, levels = ORI_ORDER)  else NA,
      age_group   = if ("age_group" %in% names(.))   factor(age_group, levels = AGE_GROUP_ORDER) else NA
    ) %>%
    arrange(Level, sex, orientation, age_group, ConcurrencyStatus) %>%
    transmute(
      Level,
      Sex = sex,
      Orientation = orientation,
      `Age group` = age_group,
      `Concurrency status` = ConcurrencyStatus,
      `Mean % ever infected` = round(Mean_PctInfected, 1),
      `SD % ever infected`   = round(SD_PctInfected, 1)
    )
}

concurrency_prevalence_table_3cat <- finalise_prevalence_table(concurrency_prevalence_raw, "3-category")
concurrency_prevalence_table_2cat <- finalise_prevalence_table(concurrency_prevalence_raw, "2-category")

# Mean partnership count table, at four levels of breakdown, for both 3-category and 2-category status systems.

compute_ever_concurrent <- function(df) {
  intervals <- df %>%
    filter(!is.na(PartnerAgent), !is.na(StartTime)) %>%
    distinct(Agent, PartnerAgent, StartTime, EndTime) %>%
    mutate(StartTime = as.double(StartTime), EndTime = as.double(EndTime))

  if (nrow(intervals) == 0)
    return(tibble(Agent = integer(0), EverConcurrent = logical(0)))

  intervals %>%
    arrange(Agent, StartTime) %>%
    group_by(Agent) %>%
    summarise(
      EverConcurrent = {
        n <- n()
        if (n < 2) {
          FALSE
        } else {
          running_max_end <- cummax(dplyr::lag(EndTime, default = -Inf))
          any(StartTime < running_max_end)
        }
      },
      .groups = "drop"
    )
}

# Build partnership count table for a single replicate, returning a data frame with counts and concurrency status for each agent.
build_partnership_count_table_by_status_one_replicate <- function(replicate_path) {

  pfiles <- list.files(replicate_path, pattern = "partnerships.*\\.parquet$", full.names = TRUE, recursive = TRUE)
  if (length(pfiles) == 0) {
    message(sprintf("[WARN] No partnership parquet file found in: %s -- skipping replicate.", replicate_path))
    return(tibble())
  }
  if (length(pfiles) > 1)
    message(sprintf("[WARN] %d partnership files found -- using the first: %s", length(pfiles), pfiles[1]))

  afiles <- list.files(replicate_path, pattern = "agent_log.*\\.parquet$", full.names = TRUE, recursive = TRUE)
  if (length(afiles) == 0) {
    message(sprintf("[WARN] No agent log parquet file found in: %s -- skipping replicate.", replicate_path))
    return(tibble())
  }
  if (length(afiles) > 1)
    message(sprintf("[WARN] %d agent log files found -- using the first: %s", length(afiles), afiles[1]))

  df        <- read_input_data(pfiles[1]) %>% add_age_group_if_missing()
  agent_log <- read_input_data(afiles[1])

  # Build a data frame of agents and whether they ever had concurrent partnerships, based on the partnership intervals.
  ever_concurrent_df <- compute_ever_concurrent(df)

  # Count the number of unique partners for each agent, filtering out agents with unknown or invalid age groups, and join with the agent log to get concurrency allowed status. Then classify concurrency status and add effective status.
  counts <- build_counts_with_zeros(df) %>%
    filter(!AgentAgeGroup %in% c("75", "Unknown"), !is.na(AgentAgeGroup)) %>%
    left_join(agent_log %>% select(Agent, ConcurrencyAllowed), by = "Agent") %>%
    left_join(ever_concurrent_df, by = "Agent") %>%
    mutate(
      EverConcurrent    = coalesce(EverConcurrent, FALSE),
      ConcurrencyStatus = classify_concurrency_status(EverConcurrent, ConcurrencyAllowed),
      AgentAgeGroup     = factor(AgentAgeGroup,    levels = AGE_GROUP_ORDER),
      AgentOrientation  = factor(AgentOrientation, levels = ORI_ORDER),
      AgentSex          = factor(AgentSex,         levels = SEX_ORDER)
    ) %>%
    add_effective_status("ConcurrencyStatus")

  # For unmatched agents (those not found in the agent log), issue a warning with the count of such agents.
  n_unmatched <- sum(is.na(counts$ConcurrencyStatus))
  if (n_unmatched > 0)
    message(sprintf("[WARN] %d agents could not be matched to the agent log (ConcurrencyStatus is NA).", n_unmatched))

  # Summarise the mean partnership count and number of agents at a given level of breakdown, averaging across all runs.
  summarise_count_level <- function(data, group_vars, level_label, status_col) {
    out <- data %>%
      filter(!is.na(.data[[status_col]])) %>%
      group_by(across(all_of(c(status_col, group_vars)))) %>%
      summarise(Mean_PartnerCount = mean(PartnerCount, na.rm = TRUE), N_agents = n(), .groups = "drop")
    names(out)[names(out) == status_col] <- "ConcurrencyStatus"
    out %>%
      mutate(
        ConcurrencyStatus = as.character(ConcurrencyStatus),
        Level = level_label,
        CategorySystem = if (status_col == "ConcurrencyStatus") "3-category" else "2-category",
        .before = 1
      )
  }

  # Bind rows for both 3-category and 2-category status systems, summarising at four levels of breakdown.
  bind_rows(
    lapply(c("ConcurrencyStatus", "EffectiveStatus"), function(status_col) {
      bind_rows(
        summarise_count_level(counts, character(0), "Overall", status_col),
        summarise_count_level(counts, "AgentSex", "By sex", status_col),
        summarise_count_level(counts, c("AgentSex", "AgentOrientation"), "By sex x orientation", status_col),
        summarise_count_level(counts, c("AgentSex", "AgentOrientation", "AgentAgeGroup"),
                               "By sex x orientation x age group", status_col)
      )
    })
  )
}

# Build partnership count table for a given scenario, averaging across all replicates, and returning a data frame with mean and standard deviation of partnership counts, along with agent counts and replicate counts.
build_scenario_partnership_count_table <- function(dir_conc, scenario_label) {
  scenario_root <- resolve_disease_root(opt$base_dir, dir_conc)
  if (grepl("replicate_", basename(scenario_root))) {
    replicate_dirs <- scenario_root
  } else {
    replicate_dirs <- list.dirs(scenario_root, recursive = FALSE, full.names = TRUE)
    replicate_dirs <- replicate_dirs[grepl("replicate_", basename(replicate_dirs))]
  }

  if (length(replicate_dirs) == 0)
    stop(sprintf("[ERROR] No replicate directories found for %s under: %s", scenario_label, scenario_root))

  message(sprintf("[INFO] Computing partnership count across %d replicate(s) in: %s",
                  length(replicate_dirs), scenario_root))

  persim <- map_df(seq_along(replicate_dirs), function(i) {
    build_partnership_count_table_by_status_one_replicate(replicate_dirs[[i]]) %>%
      mutate(ReplicateID = i)
  })

  if (nrow(persim) == 0)
    stop(sprintf("[ERROR] No partnership-count rows were produced from %d replicate(s) under: %s",
                length(replicate_dirs), scenario_root))

  message(sprintf("[INFO] %s partnership count averaged across %d replicate(s).",
                  scenario_label, n_distinct(persim$ReplicateID)))

  persim %>%
    mutate(Level = factor(Level, levels = c("Overall", "By sex", "By sex x orientation",
                                            "By sex x orientation x age group"))) %>%
    group_by(CategorySystem, Level, AgentSex, AgentOrientation, AgentAgeGroup, ConcurrencyStatus) %>%
    summarise(
      SD_PartnerCount   = sd(Mean_PartnerCount,   na.rm = TRUE),
      Mean_PartnerCount = mean(Mean_PartnerCount, na.rm = TRUE),
      Mean_N_agents     = mean(N_agents, na.rm = TRUE),
      N_replicates      = n(),
      .groups = "drop"
    ) %>%
    arrange(CategorySystem, Level, AgentSex, AgentOrientation, AgentAgeGroup, ConcurrencyStatus) %>%
    transmute(
      CategorySystem, Level,
      Sex = AgentSex, Orientation = AgentOrientation, `Age group` = AgentAgeGroup,
      `Concurrency status` = ConcurrencyStatus,
      `Mean agents (partnership)` = round(Mean_N_agents, 1),
      `Mean partnership count` = round(Mean_PartnerCount, 2),
      `SD partnership count` = round(SD_PartnerCount, 2)
    )
}

# Split the partnership count table into 3-category and 2-category tables, relabeling "Behaviourally concurrent" to "Polygamous" for the 3-category table, and ensuring the concurrency status is a factor with the appropriate levels.
split_partnership_table <- function(tbl, category_system) {
  status_levels <- if (category_system == "3-category") STATUS_LEVELS_3CAT_DISPLAY else STATUS_LEVELS_2CAT
  tbl %>%
    filter(CategorySystem == category_system) %>%
    select(-CategorySystem) %>%
    mutate(
      `Concurrency status` = if (category_system == "3-category") {
        dplyr::recode(`Concurrency status`, "Behaviourally concurrent" = "Polygamous")
      } else {
        `Concurrency status`
      },
      `Concurrency status` = factor(`Concurrency status`, levels = status_levels)
    )
}

# Partnership count tables for the 15% concurrency scenario, split into 3-category and 2-category systems.
partnership_count_status_table_all   <- build_scenario_partnership_count_table(opt$dir_conc_15, "15% concurrency")
partnership_count_status_table_3cat  <- split_partnership_table(partnership_count_status_table_all, "3-category")
partnership_count_status_table_2cat  <- split_partnership_table(partnership_count_status_table_all, "2-category")

# 0% scenario -- used only for p9b's baseline population size.
partnership_count_status_table_all_0  <- build_scenario_partnership_count_table(opt$dir_conc_0, "No concurrency")
partnership_count_status_table_2cat_0 <- split_partnership_table(partnership_count_status_table_all_0, "2-category")

# Combined table of partnership counts and prevalence, for both 3-category and 2-category systems, at four levels of breakdown.
build_combined_table <- function(prevalence_tbl, partnership_tbl) {
  prevalence_tbl %>%
    mutate(across(c(Level, Sex, Orientation, `Age group`, `Concurrency status`), as.character)) %>%
    inner_join(
      partnership_tbl %>% mutate(across(c(Level, Sex, Orientation, `Age group`, `Concurrency status`), as.character)),
      by = c("Level", "Sex", "Orientation", "Age group", "Concurrency status")
    ) %>%
    select(Level, Sex, Orientation, `Age group`, `Concurrency status`,
           `Mean agents (partnership)`, `Mean partnership count`,
           `Mean % ever infected`, `SD % ever infected`)
}

combined_table_3cat <- build_combined_table(concurrency_prevalence_table_3cat, partnership_count_status_table_3cat)
combined_table_2cat <- build_combined_table(concurrency_prevalence_table_2cat, partnership_count_status_table_2cat)

# Scatter plots of partnership count vs. prevalence, split by orientation and faceted by sex, 
# with lines connecting the Monogamous and Polygamous points for each age group. 
# The points are colored by concurrency status, and the axes are scaled appropriately.
message("[INFO] Relationship scatter, split by orientation: partnership count vs. % infected...")

scatter_two_category_relationship_data <- combined_table_2cat %>%
  filter(Level == "By sex x orientation x age group")

STATUS_PALETTE_2CAT_DARK <- c("Monogamous" = "#3E8FBD", "Polygamous" = "#D95F4A")

make_scatter_two_category_orientation_plot <- function(ori_label) {
  d <- scatter_two_category_relationship_data %>% filter(Orientation == ori_label)

  ggplot(d, aes(x = `Mean partnership count`, y = `Mean % ever infected`, colour = `Concurrency status`)) +
    geom_line(aes(group = `Age group`), colour = "grey55", linewidth = 0.5, linetype = "dashed", alpha = 0.8) +
    geom_point(size = 2.6, alpha = 0.80, shape = 16) +
    scale_colour_manual(values = STATUS_PALETTE_2CAT_DARK, name = "Concurrency status") +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25), expand = expansion(mult = c(0, 0.02))) +
    scale_x_continuous(limits = c(0, 25), breaks = seq(0, 25, 5), expand = expansion(mult = c(0.02, 0.03))) +
    facet_wrap(~ Sex, nrow = 1) +
    labs(
      title = paste0("Partnership count and infection prevalence \u2014 ", ori_label),
      x = "Mean number of partnerships",
      y = "Ever infected (%)",
      caption = paste(
        "Each point represents a sex \u00d7 age-group cell within this orientation.",
        "Lines connect the paired Monogamous / Polygamous values for the same age group.",
        ELIGIBLE_NOTE,
        sep = "\n"
      )
    ) +
    base_theme +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      axis.title = element_text(size = 11, face = "bold"),
      axis.text = element_text(size = 9, colour = "grey20"),
      strip.text = element_text(size = 12, face = "bold"),
      legend.position = "top",
      legend.title = element_text(size = 10, face = "bold"),
      legend.text = element_text(size = 9),
      legend.direction = "horizontal",
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.1, colour = "grey88"),
      panel.border = element_blank(),
      plot.caption = element_text(size = 8.5, colour = "grey40", hjust = 0),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )
}

for (ori_label in ORI_ORDER) {
  save_disease_plot(
    make_scatter_two_category_orientation_plot(ori_label),
    paste0("scatter_two_category_partnership_count_vs_prevalence_", janitor::make_clean_names(ori_label)),
    width = 10, height = 5
  )
}

# Prevalence by age group x orientation x scenario (p8) -- one bar per scenario, grouped by age group, faceted by orientation. Error bars are ±1 SD across runs.
message("[INFO] Plot 8: prevalence by age group x orientation x scenario...")

metric_compare_multi(
  df_list_by_group = DISEASE_SCENARIO_GROUPS,
  metric_fn   = metric_pct_infected,
  title       = "Infection prevalence by age group, orientation, and sex",
  y_lab       = "% ever infected",
  group_alpha = DISEASE_SCENARIO_ALPHA,
  output_dir  = opt$output,
  filename    = "p8_prevalence_by_age_orientation_scenario",
  group_name  = "Scenario",
  save_pdf    = opt$save_pdf
)

# Plot 9b/9e: baseline vs. status spillover comparisons -- one bar per scenario or status, grouped by age group, faceted by orientation. Error bars are ±1 SD across runs.
message("[INFO] Plot 9b/9e: baseline vs. status spillover comparisons...")

spillover_base_raw <- bind_rows(
  conc_status_all %>%
    filter(Scenario == "No concurrency", !is.na(EffectiveStatus), EffectiveStatus == "Monogamous") %>%
    mutate(Scenario2 = "No concurrency scenario", Status2 = "Monogamous"),
  conc_status_all %>%
    filter(Scenario == "15% concurrency", !is.na(EffectiveStatus)) %>%
    mutate(Scenario2 = "15% concurrency scenario", Status2 = as.character(EffectiveStatus))
)

# Number of unique runs in the spillover data, used for labeling plots.
N_RUNS_SPILLOVER <- n_distinct(spillover_base_raw$RunID)

# Summarise the spillover data by scenario, status, run, age group, sex, and orientation
# calculating the mean percentage infected, number of agents, and number of infected agents. 
# Then summarise across runs to get the mean and standard deviation for each group.
spillover_summary_all <- spillover_base_raw %>%
  group_by(Scenario2, Status2, RunID, age_group, sex, orientation) %>%
  summarise(PctInfected = mean(ever_infected) * 100, N = n(), N_infected = sum(ever_infected), .groups = "drop") %>%
  group_by(Scenario2, Status2, age_group, sex, orientation) %>%
  summarise(
    Mean = mean(PctInfected), SD = sd(PctInfected),
    N_mean = mean(N), N_total = sum(N),
    N_infected_mean = mean(N_infected), N_infected_SD = sd(N_infected), N_infected_total = sum(N_infected),
    .groups = "drop"
  )

partnership_pop_overall <- partnership_count_status_table_2cat %>%
  filter(Level == "Overall") %>%
  select(`Concurrency status`, `Mean agents (partnership)`) %>%
  tibble::deframe()

partnership_pop_overall_0 <- partnership_count_status_table_2cat_0 %>%
  filter(Level == "Overall", `Concurrency status` == "Monogamous") %>%
  pull(`Mean agents (partnership)`)

# Baseline comparison: No concurrency vs. 15% concurrency, Monogamous only (p9b)
BASELINE_LEVELS  <- c("No concurrency scenario", "15% concurrency scenario")
BASELINE_PALETTE <- c(
  "No concurrency scenario"  = scales::alpha("#4393c3", 0.4),
  "15% concurrency scenario" = "#4393c3"
)

baseline_pop_size <- c(
  "No concurrency scenario"  = partnership_pop_overall_0,
  "15% concurrency scenario" = unname(partnership_pop_overall[["Monogamous"]])
)

BASELINE_LABELS <- setNames(
  sprintf("%s\nMonogamous (avg population size \u2248 %s)", BASELINE_LEVELS,
          scales::comma(baseline_pop_size[BASELINE_LEVELS], accuracy = 1)),
  BASELINE_LEVELS
)

baseline_data <- spillover_summary_all %>%
  filter(Status2 == "Monogamous") %>%
  mutate(Scenario2 = factor(Scenario2, levels = BASELINE_LEVELS))

BASELINE_CAPTION <- paste(
  "Monogamous agents only. No concurrency scenario vs. 15% concurrency scenario.",
  "Error bars: \u00b11 SD across runs.", ELIGIBLE_NOTE,
  sep = "\n"
)

# Make a bar plot comparing the baseline infection rates for monogamous individuals across age groups and orientations, 
# with error bars representing ±1 SD across runs. 
  ggplot(data, aes(x = age_group, y = Mean, fill = Scenario2)) +
    geom_bar(stat = "identity", position = dodge, width = 0.8, colour = "white", linewidth = 0.4) +
    geom_errorbar(aes(ymin = pmax(Mean - SD, 0), ymax = Mean + SD),
                  position = dodge, width = 0.2, linewidth = 0.5, colour = "grey40") +
    scale_fill_manual(values = BASELINE_PALETTE, name = NULL, labels = BASELINE_LABELS, drop = FALSE) +
    scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
    facet_wrap(~orientation, nrow = 1) +
    labs(
      title    = sprintf("Spillover risk to monogamous individuals: baseline vs. 15%% scenario (%s)", sex_label),
      subtitle = sprintf("%% ever infected, averaged across %d runs", N_RUNS_SPILLOVER),
      x = "Age group", y = "% ever infected", caption = BASELINE_CAPTION
    ) +
    base_theme +
    theme(legend.text = element_text(size = 13, lineheight = 1.2), legend.key.size = unit(2, "lines"))
}

save_disease_plot(make_baseline_rate_plot(baseline_data %>% filter(sex == "Male"), "Male"),
                   "p9b_baseline_spillover_by_age_orientation_male", width = 16, height = 8)
save_disease_plot(make_baseline_rate_plot(baseline_data %>% filter(sex == "Female"), "Female"),
                   "p9b_baseline_spillover_by_age_orientation_female", width = 16, height = 8)

# ── STATUS SHARE comparison (p9e): Monogamous vs. Polygamous, within 15% scenario ──
status_data <- spillover_summary_all %>%
  filter(Scenario2 == "15% concurrency scenario") %>%
  mutate(Status2 = factor(Status2, levels = STATUS_LEVELS_2CAT))

# Status share caption text, explaining the plot and the meaning of the bar heights and labels.
STATUS_SHARE_CAPTION <- paste(
  "Within the 15% concurrency scenario only. Bar height = total % ever infected",
  "for that age group (both concurrency statuses combined), split into the",
  "Monogamous and Polygamous contribution. Label above each bar = total mean",
  "infected agents (raw count) behind that bar.", ELIGIBLE_NOTE,
  sep = "\n"
)

# This plot is a stacked bar chart showing the percentage of ever infected individuals by concurrency status 
# (Monogamous vs. Polygamous) within the 15% concurrency scenario, split by age group and orientation. 
# The bars are stacked to show the contribution of each status to the total infection prevalence for that age group, and labels above each bar indicate the total mean number of infected agents.
make_status_share_plot <- function(data, sex_label) {
  make_panel <- function(ori, show_y_title) {
    ori_data <- data %>%
      filter(orientation == ori) %>%
      mutate(N_infected_mean = tidyr::replace_na(N_infected_mean, 0))

    # Total population (both statuses combined) per age group -- this is the
    # denominator for the % ever infected figure, so segments sum to the
    # correct combined rate rather than to an arbitrary 100%.
    age_totals <- ori_data %>%
      group_by(age_group) %>%
      summarise(N_total_pop_age = sum(N_mean, na.rm = TRUE), .groups = "drop")

    ori_data <- ori_data %>%
      left_join(age_totals, by = "age_group") %>%
      mutate(PctContribution = if_else(N_total_pop_age > 0,
                                        N_infected_mean / N_total_pop_age * 100, 0))

    totals <- ori_data %>%
      group_by(age_group) %>%
      summarise(Total = sum(N_infected_mean), TotalPct = sum(PctContribution), .groups = "drop")

    pop_totals <- ori_data %>%
      group_by(Status2) %>%
      summarise(N_total_pop = sum(N_mean), .groups = "drop") %>%
      tidyr::complete(Status2 = factor(STATUS_LEVELS_2CAT, levels = STATUS_LEVELS_2CAT), fill = list(N_total_pop = 0)) %>%
      arrange(Status2)

    caption_text <- paste(sprintf("%s: %s", as.character(pop_totals$Status2),
                                   scales::comma(pop_totals$N_total_pop, accuracy = 1)), collapse = "\n")

    ggplot(ori_data, aes(x = age_group, y = PctContribution, fill = Status2)) +
      geom_bar(stat = "identity", position = "stack", width = 0.7, colour = "white", linewidth = 0.4) +
      geom_text(data = totals, aes(x = age_group, y = TotalPct, label = scales::comma(Total, accuracy = 1)),
                inherit.aes = FALSE, vjust = -0.4, size = 4, colour = "grey20") +
      scale_fill_manual(values = STATUS_PALETTE_2CAT, name = NULL, drop = FALSE) +
      scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 25),
                          labels = function(x) paste0(x, "%"),
                          expand = expansion(mult = c(0, 0.12))) +
      labs(title = ori, x = "Age group",
           y = if (show_y_title) "% ever infected" else NULL,
           caption = caption_text) +
      base_theme +
      theme(plot.title = element_text(size = 17, face = "bold", hjust = 0.5),
            axis.text = element_text(size = 17), axis.title = element_text(size = 17, face = "bold"),
            plot.caption = element_text(size = 13, colour = "grey30", hjust = 0.5, lineheight = 1.3))
  }

  panels <- lapply(seq_along(ORI_ORDER), function(i) make_panel(ORI_ORDER[i], show_y_title = (i == 1)))

  wrap_plots(panels, nrow = 1, guides = "collect") +
    plot_annotation(
      title = sprintf("Infection prevalence by concurrency status \u2014 15%% scenario (%s)", sex_label),
      subtitle = STATUS_SHARE_CAPTION,
      theme = theme(plot.title = element_text(face = "bold", size = 20),
                    plot.subtitle = element_text(size = 12, colour = "grey40", lineheight = 1.2))
    ) &
    theme(legend.position = "top")
}

# Save the status share plots for male and female, showing the percentage of ever infected individuals by concurrency status within the 15% concurrency scenario, split by age group and orientation.
save_disease_plot(make_status_share_plot(status_data %>% filter(sex == "Male"), "Male"),
                   "p9e_status_infected_share_stacked_male", width = 16, height = 8)
save_disease_plot(make_status_share_plot(status_data %>% filter(sex == "Female"), "Female"),
                   "p9e_status_infected_share_stacked_female", width = 16, height = 8)

# Output helpers (CSV + LaTeX) 
# Shortens column names, category values, and the words used inside the
# "Level" column ("By sex x orientation x age group" -> "By S x O x A") in
# both the written CSVs and the written .tex tables
#   columns : Sex -> S, Orientation -> O, Age group -> A
#   Sex          : Male -> M, Female -> F
#   Orientation  : Opposite-sex -> Opp, Same-sex -> Same, Bisexual -> Bi
#   Concurrency status : Monogamous -> Mono, Polygamous -> Poly,
#                         Eligible, not concurrent -> Not_Poly (3-category only)
#   Level        : "sex" -> "S", "orientation" -> "O", "age group" -> "A"
#                  (word-level substring replacement, e.g. "By sex" -> "By S")
abbreviate_for_output <- function(df) {
  df %>%
    mutate(
      across(any_of("Sex"), ~ dplyr::recode(as.character(.x),
                                             "Male" = "M", "Female" = "F")),
      across(any_of("Orientation"), ~ dplyr::recode(as.character(.x),
                                                     "Opposite-sex" = "Opp",
                                                     "Same-sex" = "Same",
                                                     "Bisexual" = "Bi")),
      across(any_of("Concurrency status"), ~ dplyr::recode(as.character(.x),
                                                            "Monogamous" = "Mono",
                                                            "Polygamous" = "Poly",
                                                            "Eligible, not concurrent" = "Not_Poly")),
      across(any_of("Level"), ~ {
        x <- as.character(.x)
        # "age group" must be replaced before "sex"/"orientation" 
        x <- gsub("age group", "A", x, fixed = TRUE)
        x <- gsub("orientation", "O", x, fixed = TRUE)
        x <- gsub("sex", "S", x, fixed = TRUE)
        x
      })
    ) %>%
    rename(any_of(c(S = "Sex", O = "Orientation", A = "Age group")))
}

# Blanks out a repeated value in the "Level" column when it's the same as the
# row directly above it, so each level label appears once per block instead
# of on every row
blank_repeated_level <- function(df, col = "Level") {
  if (!col %in% names(df)) return(df)
  vals <- as.character(df[[col]])
  vals[vals == dplyr::lag(vals, default = "")] <- ""
  df[[col]] <- vals
  df
}

# LaTeX table output helpers

merge_pct_sd_for_latex <- function(df) {
  pct_col <- "\\% ever infected"
  if (all(c("Mean % ever infected", "SD % ever infected") %in% names(df))) {
    df <- df %>%
      mutate(!!pct_col := sprintf("%.1f $\\pm$ %.1f", `Mean % ever infected`, `SD % ever infected`)) %>%
      select(-`Mean % ever infected`, -`SD % ever infected`)
  } else if (all(c("% ever infected (mean)", "% ever infected (SD)") %in% names(df))) {
    df <- df %>%
      mutate(!!pct_col := sprintf("%.1f $\\pm$ %.1f", `% ever infected (mean)`, `% ever infected (SD)`)) %>%
      select(-`% ever infected (mean)`, -`% ever infected (SD)`)
  }
  df
}

make_latex_shortstack <- function(x, align = "c") {
  vapply(x, function(s) {
    lines <- strsplit(s, "\n", fixed = TRUE)[[1]]
    if (length(lines) <= 1) return(s)
    paste0("\\shortstack[", align, "]{", paste(lines, collapse = "\\\\"), "}")
  }, character(1), USE.NAMES = FALSE)
}

write_latex_table <- function(df, filename, caption, label, header_width = 10) {
  df <- df %>% abbreviate_for_output() %>% blank_repeated_level() %>% merge_pct_sd_for_latex()

  # Convert long headers onto multiple lines so wide tables like the
  # "By S x O x A" breakdown stay inside the page margins 
  wrapped_names <- make_latex_shortstack(
    stringr::str_wrap(names(df), width = header_width), align = "c"
  )

  tex <- kable(df, format = "latex", booktabs = TRUE, longtable = TRUE,
               caption = caption, label = label, linesep = "", row.names = FALSE,
               escape = FALSE, col.names = wrapped_names)
  writeLines(as.character(tex), file.path(opt$output, filename))
  message(sprintf("  [OK] %s", filename))
}

# Supplementary table: sample sizes and infected agents, Monogamous vs. baseline scenario comparison
supp_table_monogamous <- spillover_summary_all %>%
  transmute(
    `Age group` = age_group, Sex = sex, Orientation = orientation,
    Scenario = Scenario2, `Concurrency status` = Status2,
    `Mean agents per run (n)`   = round(N_mean, 1),
    `Total agents (all runs)`   = N_total,
    `Mean infected per run`     = round(N_infected_mean, 1),
    `Total infected (all runs)` = N_infected_total,
    `% ever infected (mean)`    = round(Mean, 1),
    `% ever infected (SD)`      = round(SD, 1)
  ) %>%
  arrange(Scenario, `Concurrency status`, Sex, Orientation, `Age group`)

write_csv(abbreviate_for_output(supp_table_monogamous) %>% blank_repeated_level(), file.path(opt$output, "supp_table_monogamous_sample_sizes.csv"))
message("  [OK] supp_table_monogamous_sample_sizes.csv")
write_latex_table(supp_table_monogamous, "supp_table_monogamous_sample_sizes.tex",
                   caption = "Sample sizes and infected agents, Monogamous vs. baseline scenario comparison.",
                   label = "tab:monogamous-sample-sizes")


# SUPPLEMENTARY CSVs + TEX 
message("[INFO] Writing supplementary CSVs and LaTeX tables...")

write_csv(abbreviate_for_output(concurrency_prevalence_table_2cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_prevalence_monogamous_vs_polygamous.csv"))
message("  [OK] supp_table_prevalence_monogamous_vs_polygamous.csv (", nrow(concurrency_prevalence_table_2cat), " rows)")
write_latex_table(concurrency_prevalence_table_2cat, "supp_table_prevalence_monogamous_vs_polygamous.tex",
                   caption = "Infection prevalence by concurrency status: Monogamous vs. Polygamous.",
                   label = "tab:prevalence-mono-vs-poly")

write_csv(abbreviate_for_output(partnership_count_status_table_2cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_partnership_count_monogamous_vs_polygamous.csv"))
message("  [OK] supp_table_partnership_count_monogamous_vs_polygamous.csv (", nrow(partnership_count_status_table_2cat), " rows)")
write_latex_table(partnership_count_status_table_2cat, "supp_table_partnership_count_monogamous_vs_polygamous.tex",
                   caption = "Mean partnership count by concurrency status: Monogamous vs. Polygamous.",
                   label = "tab:partnership-count-mono-vs-poly")

write_csv(abbreviate_for_output(combined_table_2cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_combined_monogamous_vs_polygamous.csv"))
message("  [OK] supp_table_combined_monogamous_vs_polygamous.csv (", nrow(combined_table_2cat), " rows)")
write_latex_table(combined_table_2cat, "supp_table_combined_monogamous_vs_polygamous.tex",
                   caption = "Combined partnership count and infection prevalence: Monogamous vs. Polygamous.",
                   label = "tab:combined-mono-vs-poly")

write_csv(abbreviate_for_output(concurrency_prevalence_table_3cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_prevalence_monogamous_vs_eligible_vs_polygamous.csv"))
message("  [OK] supp_table_prevalence_monogamous_vs_eligible_vs_polygamous.csv (", nrow(concurrency_prevalence_table_3cat), " rows)")
write_latex_table(concurrency_prevalence_table_3cat, "supp_table_prevalence_monogamous_vs_eligible_vs_polygamous.tex",
                   caption = "Infection prevalence by concurrency status: Monogamous vs. Eligible, not concurrent vs. Polygamous.",
                   label = "tab:prevalence-3cat")

write_csv(abbreviate_for_output(partnership_count_status_table_3cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_partnership_count_monogamous_vs_eligible_vs_polygamous.csv"))
message("  [OK] supp_table_partnership_count_monogamous_vs_eligible_vs_polygamous.csv (", nrow(partnership_count_status_table_3cat), " rows)")
write_latex_table(partnership_count_status_table_3cat, "supp_table_partnership_count_monogamous_vs_eligible_vs_polygamous.tex",
                   caption = "Mean partnership count by concurrency status: Monogamous vs. Eligible, not concurrent vs. Polygamous.",
                   label = "tab:partnership-count-3cat")

write_csv(abbreviate_for_output(combined_table_3cat) %>% blank_repeated_level(),
          file.path(opt$output, "supp_table_combined_monogamous_vs_eligible_vs_polygamous.csv"))
message("  [OK] supp_table_combined_monogamous_vs_eligible_vs_polygamous.csv (", nrow(combined_table_3cat), " rows)")
write_latex_table(combined_table_3cat, "supp_table_combined_monogamous_vs_eligible_vs_polygamous.tex",
                   caption = "Combined partnership count and infection prevalence: Monogamous vs. Eligible, not concurrent vs. Polygamous.",
                   label = "tab:combined-3cat")

message("[DONE] All plots, supplementary CSVs, and LaTeX tables written to: ", opt$output)
