#' @export
print.nutpie_density_evaluator_check <- function(x, ...) {
  number <- function(value) format(value, digits = 3, trim = TRUE)
  maximum <- function(values) {
    values <- values[!is.na(values)]
    if (length(values)) number(max(abs(values))) else "not available"
  }
  short <- function(text) {
    text <- gsub("[[:cntrl:][:space:]]+", " ", text)
    if (nchar(text) > 160L) paste0(substr(text, 1L, 157L), "...") else text
  }
  cat("Density evaluator check: ", x$status, " (advisory); points: ",
      x$counts[["pass"]], " pass, ", x$counts[["fail"]], " fail, ",
      x$counts[["inconclusive"]], " inconclusive\n", sep = "")
  if (!is.null(x$groups)) for (i in seq_len(nrow(x$groups))) {
    group <- x$groups[i, ]
    cat("  ", group$point_source, ": ", group$n, " points; pointwise ", group$status,
        " (", group$pass, " pass, ", group$fail, " fail, ",
        group$inconclusive, " inconclusive)\n", sep = "")
  }
  if (!is.null(x$pilot)) {
    s <- x$pilot$settings
    cat("Reference pilot: 1 chain, ", s$num_warmup, " warmup + ", s$num_draws,
        " draws; ", number(x$pilot$elapsed_seconds), " s (BridgeStan only)\n", sep = "")
    cat("Pilot diagnostics: $pilot$diagnostics; short pilot is not a convergence guarantee.\n")
  }
  cat("Max absolute difference: logp ", maximum(x$comparisons$logp_error),
      "; gradient ", maximum(x$gradients$error), "\n", sep = "")
  reason <- switch(x$repeatability$reason,
    need_two_distinct_points = "need two distinct points",
    invalid_repeated_point = "invalid repeated point",
    interleaved_q1_q2_q1 = "q1, q2, q1",
    x$repeatability$reason)
  cat("Repeatability: ", x$repeatability$status, " (", reason, ")\n", sep = "")

  failures <- which(x$comparisons$status == "fail")
  if (length(failures)) {
    row <- x$comparisons[failures[1L], ]
    bad_gradient <- which(x$gradients$point == row$point & !x$gradients$pass)
    coordinate <- if (length(bad_gradient))
      paste0(", coordinate ", x$gradients$coordinate[bad_gradient[1L]]) else ""
    message <- if (nzchar(row$evaluator_message)) row$evaluator_message else row$reference_message
    cat("First failure: point ", row$point, coordinate, " (", row$reason, ")",
        if (nzchar(message)) paste0(": ", short(message)) else "", "\n", sep = "")
  }
  if (identical(x$constant_offset$status, "possible_constant_offset")) {
    within <- all(x$comparisons$logp_pass %in% TRUE)
    cat("Possible constant offset: ", number(x$constant_offset$offset),
        if (within) " (within ordinary logp tolerances; not corrected)" else
          " (no tolerance exemption; not corrected)", "\n", sep = "")
  }
  cat("Details: $comparisons, $gradients, $repeatability, $tolerances, $untested\n")
  invisible(x)
}
