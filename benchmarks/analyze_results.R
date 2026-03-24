d <- readRDS("benchmarks/results/benchmark_results.rds")

cat("========== OVERVIEW ==========\n")
cat("Total posteriors:", nrow(d), "\n")
cat("nutpieR compile OK:", sum(d$nutpie_compile_ok, na.rm = TRUE), "\n")
cat("nutpieR sample OK:", sum(d$nutpie_sample_ok, na.rm = TRUE), "\n")
cat("CmdStanR compile OK:", sum(d$cmdstanr_compile_ok, na.rm = TRUE), "\n")
cat("CmdStanR sample OK:", sum(d$cmdstanr_sample_ok, na.rm = TRUE), "\n")

# nutpieR errors
cat("\n========== nutpieR ERRORS ==========\n")
compile_errs <- d[!is.na(d$nutpie_compile_error), ]
sample_errs <- d[!is.na(d$nutpie_sample_error), ]
if (nrow(compile_errs) > 0) {
  cat("Compile errors:\n")
  for (i in seq_len(nrow(compile_errs)))
    cat("  ", compile_errs$posterior_name[i], ":", compile_errs$nutpie_compile_error[i], "\n")
}
if (nrow(sample_errs) > 0) {
  cat("Sample errors:\n")
  for (i in seq_len(nrow(sample_errs)))
    cat("  ", sample_errs$posterior_name[i], ":", sample_errs$nutpie_sample_error[i], "\n")
}
if (nrow(compile_errs) == 0 && nrow(sample_errs) == 0) cat("None!\n")

# CmdStanR errors
cat("\n========== CmdStanR ERRORS ==========\n")
compile_errs2 <- d[!is.na(d$cmdstanr_compile_error), ]
sample_errs2 <- d[!is.na(d$cmdstanr_sample_error), ]
if (nrow(compile_errs2) > 0) {
  cat("Compile errors:\n")
  for (i in seq_len(nrow(compile_errs2)))
    cat("  ", compile_errs2$posterior_name[i], ":", compile_errs2$cmdstanr_compile_error[i], "\n")
}
if (nrow(sample_errs2) > 0) {
  cat("Sample errors:\n")
  for (i in seq_len(nrow(sample_errs2)))
    cat("  ", sample_errs2$posterior_name[i], ":", sample_errs2$cmdstanr_sample_error[i], "\n")
}
if (nrow(compile_errs2) == 0 && nrow(sample_errs2) == 0) cat("None!\n")

# Speed
cat("\n========== SPEED (CmdStanR / nutpieR) ==========\n")
both <- d[d$nutpie_sample_ok & d$cmdstanr_sample_ok & !is.na(d$speed_ratio), ]
cat("Models compared:", nrow(both), "\n")
cat("Median:", round(median(both$speed_ratio), 2), "x\n")
cat("Mean:", round(mean(both$speed_ratio), 2), "x\n")
cat("Min:", round(min(both$speed_ratio), 2), "x\n")
cat("Max:", round(max(both$speed_ratio), 2), "x\n")
cat("nutpieR faster:", sum(both$speed_ratio > 1), "/", nrow(both), "\n")
cat("CmdStanR faster:", sum(both$speed_ratio < 1), "/", nrow(both), "\n")

cat("\nTop 5 nutpieR wins:\n")
top5 <- both[order(-both$speed_ratio), ][1:min(5, nrow(both)), ]
for (i in seq_len(nrow(top5)))
  cat(sprintf("  %s: %.1fx (nutpie=%.1fs, cmdstanr=%.1fs)\n",
      top5$posterior_name[i], top5$speed_ratio[i],
      top5$nutpie_sample_time_s[i], top5$cmdstanr_sample_time_s[i]))

cat("\nTop 5 CmdStanR wins (or closest):\n")
bot5 <- both[order(both$speed_ratio), ][1:min(5, nrow(both)), ]
for (i in seq_len(nrow(bot5)))
  cat(sprintf("  %s: %.2fx (nutpie=%.1fs, cmdstanr=%.1fs)\n",
      bot5$posterior_name[i], bot5$speed_ratio[i],
      bot5$nutpie_sample_time_s[i], bot5$cmdstanr_sample_time_s[i]))

# Correctness vs reference
cat("\n========== CORRECTNESS vs REFERENCE ==========\n")
ref <- d[!is.na(d$nutpie_ref_max_mean_diff), ]
cat("Models with reference draws:", nrow(ref), "\n")
cat("nutpieR max_mean_diff > 0.5:", sum(ref$nutpie_ref_max_mean_diff > 0.5), "\n")
cat("CmdStanR max_mean_diff > 0.5:", sum(ref$cmdstanr_ref_max_mean_diff > 0.5, na.rm = TRUE), "\n")

flagged <- ref[ref$nutpie_ref_max_mean_diff > 0.5, ]
if (nrow(flagged) > 0) {
  cat("\nFlagged nutpieR models (mean_diff > 0.5 SDs from reference):\n")
  for (i in seq_len(nrow(flagged)))
    cat(sprintf("  %s: mean_diff=%.2f, sd_ratio=%.2f, div=%s, rhat=%.3f\n",
        flagged$posterior_name[i], flagged$nutpie_ref_max_mean_diff[i],
        flagged$nutpie_ref_max_sd_ratio[i],
        flagged$nutpie_n_divergent[i], flagged$nutpie_max_rhat[i]))
} else {
  cat("All nutpieR models within 0.5 SDs of reference! Great.\n")
}

# Cross-sampler
cat("\n========== CROSS-SAMPLER AGREEMENT ==========\n")
cross <- d[!is.na(d$cross_max_mean_diff), ]
cat("Models compared:", nrow(cross), "\n")
cat("Disagreements (max_mean_diff > 0.5):", sum(cross$cross_max_mean_diff > 0.5), "\n")
bad_cross <- cross[cross$cross_max_mean_diff > 0.5, ]
if (nrow(bad_cross) > 0) {
  cat("\nDisagreeing models:\n")
  for (i in seq_len(nrow(bad_cross)))
    cat(sprintf("  %s: mean_diff=%.2f, sd_ratio=%.2f\n",
        bad_cross$posterior_name[i], bad_cross$cross_max_mean_diff[i],
        bad_cross$cross_max_sd_ratio[i]))
}

# Divergences
cat("\n========== DIVERGENCES ==========\n")
divs <- d[!is.na(d$nutpie_n_divergent) & d$nutpie_n_divergent > 0, ]
cat("nutpieR models with divergences:", nrow(divs), "\n")
if (nrow(divs) > 0) {
  divs <- divs[order(-divs$nutpie_n_divergent), ]
  for (i in seq_len(nrow(divs)))
    cat(sprintf("  %s: nutpie=%d, cmdstanr=%s\n",
        divs$posterior_name[i], divs$nutpie_n_divergent[i],
        ifelse(is.na(divs$cmdstanr_n_divergent[i]), "NA", divs$cmdstanr_n_divergent[i])))
}

# Rhat concerns
cat("\n========== CONVERGENCE (Rhat > 1.05) ==========\n")
rhat_bad <- d[!is.na(d$nutpie_max_rhat) & d$nutpie_max_rhat > 1.05, ]
cat("nutpieR:", nrow(rhat_bad), "models\n")
if (nrow(rhat_bad) > 0) {
  for (i in seq_len(nrow(rhat_bad)))
    cat(sprintf("  %s: rhat=%.3f, ess=%.0f\n",
        rhat_bad$posterior_name[i], rhat_bad$nutpie_max_rhat[i],
        rhat_bad$nutpie_min_ess_bulk[i]))
}
rhat_bad2 <- d[!is.na(d$cmdstanr_max_rhat) & d$cmdstanr_max_rhat > 1.05, ]
cat("CmdStanR:", nrow(rhat_bad2), "models\n")
if (nrow(rhat_bad2) > 0) {
  for (i in seq_len(nrow(rhat_bad2)))
    cat(sprintf("  %s: rhat=%.3f, ess=%.0f\n",
        rhat_bad2$posterior_name[i], rhat_bad2$cmdstanr_max_rhat[i],
        rhat_bad2$cmdstanr_min_ess_bulk[i]))
}
