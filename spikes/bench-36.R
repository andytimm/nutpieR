#!/usr/bin/env Rscript
# Benchmark #36: Compare progress paths on the current install (proxy active).
#
# Configurations (all with tbbmalloc_proxy linked):
#   A) rprintf  — progress="auto" (Rprintf bar, no SEXP allocation)
#   B) callback — force R callback path via NUTPIER_FORCE_R_CALLBACK=1
#   C) none     — progress="none" (no progress at all, lower bound)

suppressPackageStartupMessages({
  library(nutpieR)
  library(posteriordb)
})

pdb <- pdb_local("/tmp/posteriordb")

models <- list(
  list(name = "large_params_200k", pdb_name = NULL, draws = 50, warmup = 50, chains = 4),
  list(name = "large_params_200k_8c", pdb_name = NULL, draws = 50, warmup = 50, chains = 8),
  list(name = "eight_schools_nc", pdb_name = "eight_schools-eight_schools_noncentered", draws = 1000, warmup = 1000, chains = 4),
  list(name = "dogs", pdb_name = "dogs-dogs", draws = 500, warmup = 500, chains = 4),
  list(name = "diamonds", pdb_name = "diamonds-diamonds", draws = 500, warmup = 500, chains = 4),
  list(name = "arma11", pdb_name = "arma-arma11", draws = 500, warmup = 500, chains = 4),
  list(name = "mesquite", pdb_name = "mesquite-mesquite", draws = 500, warmup = 500, chains = 4),
  list(name = "radon_hier_nc", pdb_name = "radon_all-radon_hierarchical_intercept_noncentered", draws = 500, warmup = 500, chains = 4)
)

configs <- c("rprintf", "callback", "none")

# Build interleaved run plan
run_plan <- expand.grid(model_idx = seq_along(models), config = configs)
set.seed(42)
run_plan <- do.call(rbind, lapply(split(run_plan, run_plan$model_idx), function(d) {
  d[sample(nrow(d)), ]
}))

# Compile all models upfront
cat("Compiling models...\n")
compiled <- list()
for (m in models) {
  if (is.null(m$pdb_name)) {
    compiled[[m$name]] <- list(
      model = nutpie_compile_model("spikes/large_params.stan", verbose = 0L),
      data = NULL
    )
  } else {
    po <- posterior(m$pdb_name, pdb)
    stan_code <- stan_code(po)
    stan_file <- tempfile(fileext = ".stan")
    writeLines(stan_code, stan_file)
    compiled[[m$name]] <- list(
      model = nutpie_compile_model(stan_file, verbose = 0L),
      data = pdb_data(po)
    )
    unlink(stan_file)
  }
  cat(sprintf("  %s compiled\n", m$name))
}
cat("\n")

results <- data.frame(
  config = character(), model = character(), seconds = numeric(),
  divergences = integer(), rhat_max = numeric(), ess_bulk_min = numeric(),
  stringsAsFactors = FALSE
)

seed <- 42

for (i in seq_len(nrow(run_plan))) {
  m <- models[[run_plan$model_idx[i]]]
  cfg <- run_plan$config[i]

  if (cfg == "callback") {
    Sys.setenv(NUTPIER_FORCE_R_CALLBACK = "1")
  } else {
    Sys.unsetenv("NUTPIER_FORCE_R_CALLBACK")
  }

  progress_mode <- if (cfg == "none") "none" else "auto"

  cat(sprintf("[%s] %s (%d draws, %d warmup, %d chains)... ",
              cfg, m$name, m$draws, m$warmup, m$chains))
  flush.console()

  t0 <- proc.time()[["elapsed"]]

  draws <- tryCatch({
    nutpie_sample(
      compiled[[m$name]]$model,
      data = compiled[[m$name]]$data,
      num_draws = m$draws,
      num_warmup = m$warmup,
      num_chains = m$chains,
      seed = seed,
      progress = progress_mode,
      cores = min(m$chains, 4)
    )
  }, error = function(e) {
    cat(sprintf("ERROR: %s\n", conditionMessage(e)))
    return(NULL)
  })

  t1 <- proc.time()[["elapsed"]]
  elapsed <- round(t1 - t0, 2)

  if (is.null(draws)) {
    results <- rbind(results, data.frame(
      config = cfg, model = m$name, seconds = elapsed,
      divergences = NA_integer_, rhat_max = NA_real_,
      ess_bulk_min = NA_real_, stringsAsFactors = FALSE
    ))
    next
  }

  diags <- nutpie_diagnostics(draws)
  n_div <- if ("divergent" %in% names(diags)) sum(diags$divergent == 1, na.rm = TRUE) else 0L

  summ <- tryCatch(
    posterior::summarize_draws(draws, posterior::rhat, posterior::ess_bulk),
    error = function(e) NULL
  )
  rhat_max <- if (!is.null(summ) && "rhat" %in% names(summ)) max(summ$rhat, na.rm = TRUE) else NA
  ess_min <- if (!is.null(summ) && "ess_bulk" %in% names(summ)) min(summ$ess_bulk, na.rm = TRUE) else NA

  cat(sprintf("%.1fs | div:%d | rhat:%.3f | ess:%.0f\n",
              elapsed, n_div, rhat_max, ess_min))

  results <- rbind(results, data.frame(
    config = cfg, model = m$name, seconds = elapsed,
    divergences = n_div, rhat_max = round(rhat_max, 4),
    ess_bulk_min = round(ess_min, 1), stringsAsFactors = FALSE
  ))

  write.csv(results, "spikes/bench-36-results.csv", row.names = FALSE)
}

cat("\n=== Summary: Sampling time (seconds) ===\n\n")

wide <- reshape(results[, c("config", "model", "seconds")],
                idvar = "model", timevar = "config", direction = "wide")
colnames(wide) <- gsub("seconds.", "", colnames(wide))
wide <- wide[, c("model", "rprintf", "callback", "none")]
wide$rp_vs_cb_pct <- round((wide$callback / wide$rprintf - 1) * 100, 1)
wide$rp_vs_none_pct <- round((wide$rprintf / wide$none - 1) * 100, 1)
print(wide, row.names = FALSE)

cat("\n--- Correctness (rprintf config) ---\n")
corr <- results[results$config == "rprintf", c("model", "divergences", "rhat_max", "ess_bulk_min")]
print(corr, row.names = FALSE)

cat("\nResults saved to spikes/bench-36-results.csv\n")
