# posteriorDB Benchmark: nutpieR vs CmdStanR
#
# Runs ~114 posteriorDB models through both nutpieR and CmdStanR,
# comparing wall clock speed, correctness vs reference posteriors,
# and catching any nutpieR errors. Based on the model subset from
# the Fisher Divergence paper (arXiv 2603.18845).
#
# Usage:
#   source("benchmarks/posteriordb_benchmark.R")
#   results <- run_benchmark()                  # all ~114 models
#   results <- run_benchmark(posteriors = "eight_schools-eight_schools_noncentered")

library(posteriordb)
library(posterior)
library(nutpieR)
library(cmdstanr)

RESULTS_DIR <- file.path("benchmarks", "results")
RESULTS_RDS <- file.path(RESULTS_DIR, "benchmark_results.rds")
RESULTS_CSV <- file.path(RESULTS_DIR, "benchmark_summary.csv")

# Models excluded from the Fisher Divergence paper's 114-model subset.
# Reconstructed from paper description (slow / non-convergent / truncated paste).
EXCLUDED_POSTERIORS <- c(

  # COVID imperial models (exceptionally slow)
  "ecdc0401-covid19imperial_v2", "ecdc0401-covid19imperial_v3",
  "ecdc0501-covid19imperial_v2", "ecdc0501-covid19imperial_v3",

  # Election (slow)
  "election88-election88_full",

  # MNIST neural network RBMs (slow)
  "mnist-nn_rbm1bJ100", "mnist_100-nn_rbm1bJ10",

  # LDA topic models (slow / non-convergent)
  "prideprejudice_chapter-ldaK5", "prideprejudice_paragraph-ldaK5",
  "three_docs1200-ldaK2", "three_men1-ldaK2", "three_men2-ldaK2", "three_men3-ldaK2",

  # Mixture models (non-convergent)
  "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse",
  "normal_2-normal_mixture", "normal_5-normal_mixture_k",

  # Not in paper's paste (confidently excluded)
  "GLMM_Poisson_data-GLMM_Poisson_model",
  "dogs-dogs_log",
  "gp_pois_regr-gp_pois_regr",
  "hmm_gaussian_simulated-hmm_gaussian",
  "iohmm_reg_simulated-iohmm_reg",

  # Likely excluded (slow/complex, matching paper's description)
  "mcycle_splines-accel_splines",
  "ovarian-logistic_regression_rhs",
  "pilots-pilots",
  "prostate-logistic_regression_rhs",
  "sir-sir",
  "synthetic_grid_RBF_kernels-kronecker_gp",
  "state_wide_presidential_votes-hierarchical_gp",
  "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
  "mcycle_gp-accel_gp",
  "one_comp_mm_elim_abs-one_comp_mm_elim_abs",
  "sat-hier_2pl",
  "soil_carbon-soil_incubation"
)

# Sampling settings (matched between nutpieR and CmdStanR)
NUM_DRAWS   <- 1000L
NUM_WARMUP  <- 400L
NUM_CHAINS  <- 4L
SEED        <- 12345L
SAMPLE_TIMEOUT_S <- 600  # 10 minutes per posterior


# -- helpers -------------------------------------------------------------------

has_reference_draws <- function(po) {

  tryCatch({
    reference_posterior_draws(po)
    TRUE
  }, error = function(e) FALSE)
}

#' Compute correctness metrics: max |mean_diff|/ref_sd and max sd_ratio
correctness_vs_reference <- function(draws, ref_draws) {
  our  <- summarize_draws(draws, "mean", "sd")
  ref  <- summarize_draws(ref_draws, "mean", "sd")
  shared <- intersect(our$variable, ref$variable)
  if (length(shared) == 0) return(list(max_mean_diff = NA_real_, max_sd_ratio = NA_real_))

  our <- our[our$variable %in% shared, ]
  ref <- ref[ref$variable %in% shared, ]
  our <- our[match(shared, our$variable), ]
  ref <- ref[match(shared, ref$variable), ]

  # Normalized mean difference: |our_mean - ref_mean| / ref_sd
  mean_diffs <- abs(our$mean - ref$mean) / pmax(ref$sd, 1e-10)
  # SD ratio: max(our_sd/ref_sd, ref_sd/our_sd)
  sd_ratios <- pmax(our$sd / pmax(ref$sd, 1e-10), ref$sd / pmax(our$sd, 1e-10))

  list(max_mean_diff = max(mean_diffs), max_sd_ratio = max(sd_ratios))
}

#' Compute correctness between two sets of draws (nutpieR vs CmdStanR)
correctness_vs_draws <- function(draws_a, draws_b) {
  a <- summarize_draws(draws_a, "mean", "sd")
  b <- summarize_draws(draws_b, "mean", "sd")
  shared <- intersect(a$variable, b$variable)
  if (length(shared) == 0) return(list(max_mean_diff = NA_real_, max_sd_ratio = NA_real_))

  a <- a[a$variable %in% shared, ]
  b <- b[b$variable %in% shared, ]
  a <- a[match(shared, a$variable), ]
  b <- b[match(shared, b$variable), ]

  # Use the average SD as normalizer
  avg_sd <- (a$sd + b$sd) / 2
  mean_diffs <- abs(a$mean - b$mean) / pmax(avg_sd, 1e-10)
  sd_ratios <- pmax(a$sd / pmax(b$sd, 1e-10), b$sd / pmax(a$sd, 1e-10))

  list(max_mean_diff = max(mean_diffs), max_sd_ratio = max(sd_ratios))
}

#' Extract summary diagnostics from nutpieR draws
nutpie_diag_summary <- function(draws) {
  diag <- nutpie_diagnostics(draws)
  summ <- summarize_draws(draws, "rhat", "ess_bulk")
  list(
    n_divergent  = sum(diag$diverging),
    max_rhat     = max(summ$rhat, na.rm = TRUE),
    min_ess_bulk = min(summ$ess_bulk, na.rm = TRUE)
  )
}

#' Extract summary diagnostics from CmdStanR fit
cmdstanr_diag_summary <- function(fit) {
  draws <- fit$draws(format = "draws_array")
  summ  <- summarize_draws(draws, "rhat", "ess_bulk")
  np    <- fit$diagnostic_summary(quiet = TRUE)
  list(
    n_divergent  = sum(np$num_divergent),
    max_rhat     = max(summ$rhat, na.rm = TRUE),
    min_ess_bulk = min(summ$ess_bulk, na.rm = TRUE)
  )
}

#' Time an expression, returning list(value, elapsed_s, error)
timed <- function(expr) {
  error <- NULL
  start <- proc.time()[["elapsed"]]
  value <- tryCatch(expr, error = function(e) { error <<- conditionMessage(e); NULL })
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed_s = elapsed, error = error)
}


# -- per-posterior benchmark ---------------------------------------------------

benchmark_one_posterior <- function(posterior_name, pdb,
                                    nutpie_model_cache,
                                    cmdstanr_model_cache) {
  cat(sprintf("[%s] Starting...\n", posterior_name))

  # Initialize result row
  row <- data.frame(
    posterior_name          = posterior_name,
    n_params               = NA_integer_,
    has_reference           = FALSE,
    nutpie_compile_ok       = FALSE,
    nutpie_compile_time_s   = NA_real_,
    nutpie_compile_error    = NA_character_,
    nutpie_sample_ok        = FALSE,
    nutpie_sample_time_s    = NA_real_,
    nutpie_sample_error     = NA_character_,
    nutpie_n_divergent      = NA_integer_,
    nutpie_max_rhat         = NA_real_,
    nutpie_min_ess_bulk     = NA_real_,
    cmdstanr_compile_ok     = FALSE,
    cmdstanr_compile_time_s = NA_real_,
    cmdstanr_compile_error  = NA_character_,
    cmdstanr_sample_ok      = FALSE,
    cmdstanr_sample_time_s  = NA_real_,
    cmdstanr_sample_error   = NA_character_,
    cmdstanr_n_divergent    = NA_integer_,
    cmdstanr_max_rhat       = NA_real_,
    cmdstanr_min_ess_bulk   = NA_real_,
    nutpie_ref_max_mean_diff   = NA_real_,
    nutpie_ref_max_sd_ratio    = NA_real_,
    cmdstanr_ref_max_mean_diff = NA_real_,
    cmdstanr_ref_max_sd_ratio  = NA_real_,
    cross_max_mean_diff     = NA_real_,
    cross_max_sd_ratio      = NA_real_,
    speed_ratio             = NA_real_,
    stringsAsFactors        = FALSE
  )

  # Get posterior, stan code, data
  po <- tryCatch(posterior(posterior_name, pdb), error = function(e) NULL)
  if (is.null(po)) {
    row$nutpie_compile_error <- "Could not load posterior from posteriordb"
    cat(sprintf("[%s] SKIP: could not load posterior\n", posterior_name))
    return(row)
  }

  stan_code_text <- tryCatch(stan_code(po), error = function(e) NULL)
  if (is.null(stan_code_text)) {
    row$nutpie_compile_error <- "No Stan code available"
    cat(sprintf("[%s] SKIP: no Stan code\n", posterior_name))
    return(row)
  }

  data <- tryCatch(pdb_data(po), error = function(e) NULL)

  # Reference draws
  ref_draws <- NULL
  if (has_reference_draws(po)) {
    row$has_reference <- TRUE
    ref_draws <- tryCatch(reference_posterior_draws(po), error = function(e) NULL)
  }

  # Write Stan code to temp file
  mname <- po$model_name
  stan_file <- tempfile(pattern = paste0(mname, "_"), fileext = ".stan")
  writeLines(stan_code_text, stan_file)
  on.exit(unlink(stan_file), add = TRUE)

  # -- Compile nutpieR ---------------------------------------------------------
  nutpie_model <- nutpie_model_cache[[mname]]
  if (is.null(nutpie_model)) {
    res <- timed(nutpie_compile_model(stan_file, verbose = 0L))
    row$nutpie_compile_time_s <- res$elapsed_s
    if (!is.null(res$error)) {
      row$nutpie_compile_error <- res$error
      cat(sprintf("[%s] nutpieR compile FAILED: %s\n", posterior_name, res$error))
    } else {
      row$nutpie_compile_ok <- TRUE
      nutpie_model <- res$value
      nutpie_model_cache[[mname]] <- nutpie_model
    }
  } else {
    row$nutpie_compile_ok <- TRUE
    row$nutpie_compile_time_s <- 0  # cached
  }

  # -- Compile CmdStanR --------------------------------------------------------
  cmdstanr_model <- cmdstanr_model_cache[[mname]]
  if (is.null(cmdstanr_model)) {
    res <- timed(cmdstan_model(stan_file, quiet = TRUE))
    row$cmdstanr_compile_time_s <- res$elapsed_s
    if (!is.null(res$error)) {
      row$cmdstanr_compile_error <- res$error
      cat(sprintf("[%s] CmdStanR compile FAILED: %s\n", posterior_name, res$error))
    } else {
      row$cmdstanr_compile_ok <- TRUE
      cmdstanr_model <- res$value
      cmdstanr_model_cache[[mname]] <- cmdstanr_model
    }
  } else {
    row$cmdstanr_compile_ok <- TRUE
    row$cmdstanr_compile_time_s <- 0  # cached
  }

  # -- Sample nutpieR ----------------------------------------------------------
  nutpie_draws <- NULL
  if (row$nutpie_compile_ok) {
    res <- timed(nutpie_sample(
      nutpie_model, data = data,
      num_draws = NUM_DRAWS, num_warmup = NUM_WARMUP,
      num_chains = NUM_CHAINS, seed = SEED,
      refresh = 0L
    ))
    row$nutpie_sample_time_s <- res$elapsed_s
    if (!is.null(res$error)) {
      row$nutpie_sample_error <- res$error
      cat(sprintf("[%s] nutpieR sample FAILED: %s\n", posterior_name, res$error))
    } else {
      row$nutpie_sample_ok <- TRUE
      nutpie_draws <- res$value
      row$n_params <- dim(nutpie_draws)[3]
      ds <- tryCatch(nutpie_diag_summary(nutpie_draws), error = function(e) NULL)
      if (!is.null(ds)) {
        row$nutpie_n_divergent  <- ds$n_divergent
        row$nutpie_max_rhat     <- ds$max_rhat
        row$nutpie_min_ess_bulk <- ds$min_ess_bulk
      }
    }
  }

  # -- Sample CmdStanR ---------------------------------------------------------
  cmdstanr_fit <- NULL
  cmdstanr_draws <- NULL
  if (row$cmdstanr_compile_ok) {
    res <- timed(cmdstanr_model$sample(
      data = data,
      chains = NUM_CHAINS,
      parallel_chains = NUM_CHAINS,
      iter_warmup = NUM_WARMUP,
      iter_sampling = NUM_DRAWS,
      seed = SEED,
      refresh = 0,
      max_treedepth = 10,
      adapt_delta = 0.8
    ))
    row$cmdstanr_sample_time_s <- res$elapsed_s
    if (!is.null(res$error)) {
      row$cmdstanr_sample_error <- res$error
      cat(sprintf("[%s] CmdStanR sample FAILED: %s\n", posterior_name, res$error))
    } else {
      row$cmdstanr_sample_ok <- TRUE
      cmdstanr_fit <- res$value
      cmdstanr_draws <- tryCatch(cmdstanr_fit$draws(format = "draws_array"),
                                  error = function(e) NULL)
      ds <- tryCatch(cmdstanr_diag_summary(cmdstanr_fit), error = function(e) NULL)
      if (!is.null(ds)) {
        row$cmdstanr_n_divergent  <- ds$n_divergent
        row$cmdstanr_max_rhat     <- ds$max_rhat
        row$cmdstanr_min_ess_bulk <- ds$min_ess_bulk
      }
    }
  }

  # -- Correctness vs reference draws ------------------------------------------
  if (!is.null(ref_draws) && !is.null(nutpie_draws)) {
    cc <- tryCatch(correctness_vs_reference(nutpie_draws, ref_draws),
                   error = function(e) NULL)
    if (!is.null(cc)) {
      row$nutpie_ref_max_mean_diff <- cc$max_mean_diff
      row$nutpie_ref_max_sd_ratio  <- cc$max_sd_ratio
    }
  }
  if (!is.null(ref_draws) && !is.null(cmdstanr_draws)) {
    cc <- tryCatch(correctness_vs_reference(cmdstanr_draws, ref_draws),
                   error = function(e) NULL)
    if (!is.null(cc)) {
      row$cmdstanr_ref_max_mean_diff <- cc$max_mean_diff
      row$cmdstanr_ref_max_sd_ratio  <- cc$max_sd_ratio
    }
  }

  # -- Cross-sampler comparison ------------------------------------------------
  if (!is.null(nutpie_draws) && !is.null(cmdstanr_draws)) {
    cc <- tryCatch(correctness_vs_draws(nutpie_draws, cmdstanr_draws),
                   error = function(e) NULL)
    if (!is.null(cc)) {
      row$cross_max_mean_diff <- cc$max_mean_diff
      row$cross_max_sd_ratio  <- cc$max_sd_ratio
    }
  }

  # -- Speed ratio -------------------------------------------------------------
  if (!is.na(row$nutpie_sample_time_s) && !is.na(row$cmdstanr_sample_time_s) &&
      row$nutpie_sample_time_s > 0) {
    row$speed_ratio <- row$cmdstanr_sample_time_s / row$nutpie_sample_time_s
  }

  # -- Summary line ------------------------------------------------------------
  status_parts <- character()
  if (row$nutpie_sample_ok) {
    status_parts <- c(status_parts,
      sprintf("nutpie=%.1fs", row$nutpie_sample_time_s))
  }
  if (row$cmdstanr_sample_ok) {
    status_parts <- c(status_parts,
      sprintf("cmdstanr=%.1fs", row$cmdstanr_sample_time_s))
  }
  if (!is.na(row$speed_ratio)) {
    status_parts <- c(status_parts, sprintf("ratio=%.1fx", row$speed_ratio))
  }
  if (isTRUE(row$nutpie_n_divergent > 0)) {
    status_parts <- c(status_parts,
      sprintf("div=%d", row$nutpie_n_divergent))
  }
  cat(sprintf("[%s] Done: %s\n", posterior_name, paste(status_parts, collapse = ", ")))

  row
}


# -- main entry point ----------------------------------------------------------

#' Run the posteriorDB benchmark
#'
#' @param posteriors Character vector of posterior names to run, or NULL for
#'   the full paper subset (~114 models).
#' @param resume If TRUE (default), skip posteriors that already have results
#'   in the saved RDS file.
run_benchmark <- function(posteriors = NULL, resume = TRUE) {
  pdb <- pdb_default()

  if (is.null(posteriors)) {
    all_names <- posterior_names(pdb)
    posteriors <- setdiff(all_names, EXCLUDED_POSTERIORS)
    cat(sprintf("Running %d posteriors (%d excluded)\n",
                length(posteriors), length(EXCLUDED_POSTERIORS)))
  }

  # Resume from previous run
  previous <- NULL
  if (resume && file.exists(RESULTS_RDS)) {
    previous <- readRDS(RESULTS_RDS)
    done <- intersect(posteriors, previous$posterior_name)
    if (length(done) > 0) {
      cat(sprintf("Resuming: skipping %d already-completed posteriors\n", length(done)))
      posteriors <- setdiff(posteriors, done)
    }
  }

  if (length(posteriors) == 0) {
    cat("All posteriors already completed.\n")
    return(invisible(previous))
  }

  cat(sprintf("Posteriors to run: %d\n", length(posteriors)))

  # Model caches (environments used as mutable hash maps)
  nutpie_model_cache  <- new.env(parent = emptyenv())
  cmdstanr_model_cache <- new.env(parent = emptyenv())

  dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

  results <- vector("list", length(posteriors))
  for (i in seq_along(posteriors)) {
    cat(sprintf("\n=== [%d/%d] ===\n", i, length(posteriors)))

    results[[i]] <- tryCatch(
      benchmark_one_posterior(posteriors[i], pdb,
                               nutpie_model_cache, cmdstanr_model_cache),
      error = function(e) {
        cat(sprintf("[%s] UNEXPECTED ERROR: %s\n", posteriors[i], conditionMessage(e)))
        data.frame(
          posterior_name = posteriors[i],
          nutpie_sample_error = paste("Unexpected:", conditionMessage(e)),
          stringsAsFactors = FALSE
        )
      }
    )

    # Save incrementally after each posterior
    current <- do.call(rbind, c(
      if (!is.null(previous)) list(previous),
      results[seq_len(i)]
    ))
    saveRDS(current, RESULTS_RDS)
    write.csv(current, RESULTS_CSV, row.names = FALSE)
  }

  all_results <- do.call(rbind, c(
    if (!is.null(previous)) list(previous),
    results
  ))
  saveRDS(all_results, RESULTS_RDS)
  write.csv(all_results, RESULTS_CSV, row.names = FALSE)

  # Print summary
  cat("\n\n========== BENCHMARK SUMMARY ==========\n")
  cat(sprintf("Total posteriors:      %d\n", nrow(all_results)))
  cat(sprintf("nutpieR compile OK:    %d\n", sum(all_results$nutpie_compile_ok, na.rm = TRUE)))
  cat(sprintf("nutpieR sample OK:     %d\n", sum(all_results$nutpie_sample_ok, na.rm = TRUE)))
  cat(sprintf("CmdStanR compile OK:   %d\n", sum(all_results$cmdstanr_compile_ok, na.rm = TRUE)))
  cat(sprintf("CmdStanR sample OK:    %d\n", sum(all_results$cmdstanr_sample_ok, na.rm = TRUE)))

  both_ok <- all_results$nutpie_sample_ok & all_results$cmdstanr_sample_ok
  if (any(both_ok, na.rm = TRUE)) {
    ratios <- all_results$speed_ratio[both_ok & !is.na(all_results$speed_ratio)]
    if (length(ratios) > 0) {
      cat(sprintf("\nSpeed ratio (CmdStanR/nutpieR):\n"))
      cat(sprintf("  Median: %.2fx\n", median(ratios)))
      cat(sprintf("  Mean:   %.2fx\n", mean(ratios)))
      cat(sprintf("  Range:  %.2fx - %.2fx\n", min(ratios), max(ratios)))
    }
  }

  # Flag problems
  nutpie_errors <- all_results[!is.na(all_results$nutpie_sample_error), ]
  if (nrow(nutpie_errors) > 0) {
    cat(sprintf("\nnutpieR errors (%d):\n", nrow(nutpie_errors)))
    for (j in seq_len(nrow(nutpie_errors))) {
      cat(sprintf("  %s: %s\n",
          nutpie_errors$posterior_name[j],
          nutpie_errors$nutpie_sample_error[j]))
    }
  }

  ref_flagged <- all_results[!is.na(all_results$nutpie_ref_max_mean_diff) &
                              all_results$nutpie_ref_max_mean_diff > 0.5, ]
  if (nrow(ref_flagged) > 0) {
    cat(sprintf("\nPossible correctness issues vs reference (%d):\n", nrow(ref_flagged)))
    for (j in seq_len(nrow(ref_flagged))) {
      cat(sprintf("  %s: max_mean_diff=%.2f, max_sd_ratio=%.2f\n",
          ref_flagged$posterior_name[j],
          ref_flagged$nutpie_ref_max_mean_diff[j],
          ref_flagged$nutpie_ref_max_sd_ratio[j]))
    }
  }

  cat(sprintf("\nResults saved to:\n  %s\n  %s\n", RESULTS_RDS, RESULTS_CSV))

  invisible(all_results)
}
