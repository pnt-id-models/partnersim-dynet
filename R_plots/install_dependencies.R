
# install_dependencies.R 
# Install all R packages required by plots.R, compare_concurrency_plots.R and disease_comparison_plots.R.
#
#  Usage (from project root or R/):
#   Rscript R_plots/install_dependencies.R
#   Rscript R_plots/install_dependencies.R --upgrade            # reinstall ALL packages, even if present
#   Rscript R_plots/install_dependencies.R --only=arrow  # (re)install just this, skip the rest
#   Rscript R_plots/install_dependencies.R --check-only         # report status without installing
#
# Packages are installed from CRAN using the default mirror.
# Set the CRAN_MIRROR environment variable to override, e.g.:
#   CRAN_MIRROR=https://cloud.r-project.org Rscript R_plots/install_dependencies.R


args <- commandArgs(trailingOnly = TRUE)
upgrade    <- "--upgrade"    %in% args
check_only <- "--check-only" %in% args   # report status without installing

only_arg  <- args[grepl("^--only=", args)]
only_pkgs <- if (length(only_arg) > 0) strsplit(sub("^--only=", "", only_arg[1]), ",")[[1]] else NULL

mirror <- Sys.getenv("CRAN_MIRROR", unset = "https://cloud.r-project.org")

# R version check: plots.R requires R >= 4.2.0 for some of the tidyverse packages.
MIN_R_MAJOR <- 4L
MIN_R_MINOR <- 2L

r_ver <- R.Version()
r_major <- as.integer(r_ver$major)
r_minor <- as.integer(strsplit(r_ver$minor, "\\.")[[1]][1])

message(sprintf("[INFO] R version: %s.%s (%s)", r_ver$major, r_ver$minor, r_ver$version.string))

if (r_major < MIN_R_MAJOR || (r_major == MIN_R_MAJOR && r_minor < MIN_R_MINOR)) {
  stop(sprintf(
    "[ERROR] R >= %d.%d is required (found %s.%s). Please upgrade R before installing packages.",
    MIN_R_MAJOR, MIN_R_MINOR, r_ver$major, r_ver$minor
  ))
}

# Package manifest
# Each entry: name = package, value = human-readable purpose.
# required = TRUE  → plots.R will not run without this
# required = FALSE → optional; improves output quality but not blocking

PACKAGES <- data.frame(stringsAsFactors = FALSE,
  pkg = c(
    # Data wrangling 
    "dplyr",          "tidyr",          "purrr",
    "readr",          "fs",             "forcats",
    # Table output (CSV + LaTeX) 
    "knitr",          "stringr",
    #  Visualisation 
    "ggplot2",        "scales",         "gridExtra",
    "patchwork",      "ggrepel",        "RColorBrewer",
    # Network graphs 
    "igraph",
    # Data cleaning
    "janitor",
    # I/O 
    "arrow",
    #  CLI
    "optparse",
    # Typography (optional) 
    "showtext",       "extrafont",
    # ── Colour accessibility (optional) 
    "colorBlindness"
  ),
  purpose = c(
    "data manipulation",
    "pivoting / reshaping",
    "functional iteration (map, map_df)",
    "read_csv",
    "directory helpers",
    "factor/level helpers (fct_relevel, etc.)",

    "kable() table generation (CSV/LaTeX supplementary tables)",
    "string wrapping (str_wrap) for multi-line LaTeX table headers",

    "base plotting engine",
    "axis formatters, pretty_breaks",
    "grid.arrange for composite figures",
    "modern replacement for gridExtra composites",
    "non-overlapping text labels",
    "ColorBrewer palettes (scale_fill_distiller)",

    "network graph objects (used by transmission-tree plots)",

    "for cleaning column names and other data wrangling tasks",

    "read_parquet / write_parquet",

    "CLI argument parsing in run_partnership_plots.R",

    "load system/Google fonts into ggplot",
    "embed fonts in PDF output",

    "cvdPlot() colour-blindness simulation"
  ),
  required = c(
    TRUE,  TRUE,  TRUE,
    TRUE,  TRUE,  TRUE,

    TRUE,  TRUE,

    TRUE,  TRUE,  TRUE,
    TRUE,  TRUE,  TRUE,

    TRUE,

    TRUE,

    TRUE,

    TRUE,

    FALSE, FALSE,

    FALSE
  )
)

required_pkgs <- PACKAGES$pkg[PACKAGES$required]
optional_pkgs <- PACKAGES$pkg[!PACKAGES$required]
all_pkgs      <- PACKAGES$pkg

# Install logic
installed_pkgs <- rownames(installed.packages())

if (check_only) {
  message("[INFO] --check-only: reporting status without installing\n")
  for (i in seq_len(nrow(PACKAGES))) {
    row    <- PACKAGES[i, ]
    ok     <- row$pkg %in% installed_pkgs
    tag    <- if (row$required) "required" else "optional"
    status <- if (ok) "[OK]  " else "[MISS]"
    message(sprintf("  %s %-20s  (%s)  %s", status, row$pkg, tag, row$purpose))
  }
  missing_req <- required_pkgs[!required_pkgs %in% installed_pkgs]
  missing_opt <- optional_pkgs[!optional_pkgs %in% installed_pkgs]
  message()
  if (length(missing_req) > 0)
    message("[WARN] Missing required:  ", paste(missing_req, collapse = ", "))
  if (length(missing_opt) > 0)
    message("[INFO] Missing optional: ", paste(missing_opt, collapse = ", "))
  if (length(missing_req) == 0 && length(missing_opt) == 0)
    message("[OK] All packages present.")
  quit(status = if (length(missing_req) > 0) 1L else 0L)
}

if (upgrade) {
  to_install <- all_pkgs
  message("[INFO] --upgrade: reinstalling all ", length(to_install), " packages")
} else if (!is.null(only_pkgs)) {
  unknown <- setdiff(only_pkgs, all_pkgs)
  if (length(unknown) > 0)
    message("[WARN] Not in the manifest (installing anyway): ", paste(unknown, collapse = ", "))

  to_install <- only_pkgs[!only_pkgs %in% installed_pkgs]
  if (length(to_install) == 0) {
    message("[OK] All packages in --only are already installed. Use --upgrade to force a reinstall.")
    quit(status = 0)
  }
  message(sprintf("[INFO] --only: targeting %d package(s): %s",
                  length(to_install), paste(to_install, collapse = ", ")))
} else {
  to_install <- all_pkgs[!all_pkgs %in% installed_pkgs]

  if (length(to_install) == 0) {
    message("[OK] All required and optional packages are already installed.")
    quit(status = 0)
  }

  n_req <- sum(to_install %in% required_pkgs)
  n_opt <- sum(to_install %in% optional_pkgs)
  message(sprintf("[INFO] Installing %d package(s): %d required, %d optional",
                  length(to_install), n_req, n_opt))
  message("       ", paste(to_install, collapse = ", "))
}

options(warn = 1)  # print warnings as they happen 

install.packages(
  to_install,
  repos        = mirror,
  dependencies = TRUE,
  quiet        = FALSE
)

# Verification 
message("\n[INFO] Verifying installations...")
failed_req <- character(0)
failed_opt <- character(0)

for (i in seq_len(nrow(PACKAGES))) {
  row <- PACKAGES[i, ]
  ok  <- suppressWarnings(requireNamespace(row$pkg, quietly = TRUE))
  ver <- if (ok) as.character(packageVersion(row$pkg)) else "—"
  tag    <- if (row$required) "required" else "optional"
  status <- if (ok) "[OK]  " else "[FAIL]"
  message(sprintf("  %s %-20s  %-8s  (%s)", status, row$pkg, ver, tag))
  if (!ok) {
    if (row$required) failed_req <- c(failed_req, row$pkg)
    else              failed_opt <- c(failed_opt, row$pkg)
  }
}

#Session info 

tryCatch({
  script_path <- sys.frame(1)$ofile
  script_dir  <- if (!is.null(script_path)) dirname(normalizePath(script_path, mustWork = FALSE)) else getwd()
  session_log <- file.path(script_dir, "session_info.txt")
  capture.output(sessionInfo(), file = session_log)
  message(sprintf("\n[INFO] Session info written to: %s", session_log))
}, error = function(e) {
  message("[WARN] Could not write session_info.txt: ", e$message)
})

# Final status 
message()
if (length(failed_opt) > 0)
  message("[WARN] Optional packages failed (non-blocking): ",
          paste(failed_opt, collapse = ", "))

if (length(failed_req) > 0) {
  message("[ERROR] Required packages failed to install:")
  message("        ", paste(failed_req, collapse = ", "))
  message()
  message("  Troubleshooting:")
  message("  1. System libraries — some packages need system-level deps:")
  sys_deps <- list(
    arrow      = list(pkgs = c("libcurl4-openssl-dev", "libssl-dev", "cmake"), note = NULL),
    extrafont  = list(pkgs = character(0), note = "no system libraries beyond base R"),
    showtext   = list(pkgs = c("libfreetype6-dev", "libfontconfig1-dev", "libpng-dev"), note = NULL)
  )
  for (pkg in failed_req) {
    if (pkg %in% names(sys_deps)) {
      dep <- sys_deps[[pkg]]
      if (length(dep$pkgs) > 0)
        message(sprintf("     %-14s → sudo apt install %s", pkg, paste(dep$pkgs, collapse = " ")))
      if (!is.null(dep$note))
        message(sprintf("     %-14s   (%s)", "", dep$note))
    }
  }
  message()
  message("  2. Manual install:")
  message('     install.packages(c(',
          paste(sprintf('"%s"', failed_req), collapse = ", "), "))")
  quit(status = 1L)
} else {
  message("[SUCCESS] All packages installed and verified.")
}