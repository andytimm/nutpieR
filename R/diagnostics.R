# Bring the upstream nuts-rs `chain` (0-indexed) and `draw` (cumulative
# 1-indexed) onto `posterior::draws_array` conventions: `chain` 1-indexed in
# 1:num_chains, `draw` 1-indexed within phase (1..num_draws or 1..num_warmup).
# Each mutation is guarded because the field set is upstream-controlled and a
# future schema change could drop one of these columns.
reindex_diagnostics <- function(diag, num_warmup, phase = c("sample", "warmup")) {
  if (is.null(diag)) return(diag)
  phase <- match.arg(phase)
  if (!is.null(diag$chain)) {
    diag$chain <- diag$chain + 1L
  }
  if (!is.null(diag$draw) && identical(phase, "sample")) {
    diag$draw <- diag$draw - as.integer(num_warmup)
  }
  diag
}

# Rename `num_tune` -> `num_warmup` in the sampler_config JSON so the field
# matches the `nutpie_sample()` argument name. Upstream nuts-rs uses
# "tune" terminology; we align it with R / Stan / cmdstanr conventions on
# the way out. Defensive: a parse failure (e.g. a future schema bump that
# breaks fromJSON) returns the original string unchanged.
rename_sampler_config <- function(json_str) {
  if (is.null(json_str) || !nzchar(json_str)) return(json_str)
  tryCatch({
    cfg <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
    if (!is.null(cfg$num_tune)) {
      cfg$num_warmup <- cfg$num_tune
      cfg$num_tune <- NULL
    }
    jsonlite::toJSON(cfg, auto_unbox = TRUE, null = "null")
  }, error = function(e) json_str)
}

#' Extract sampler diagnostics from nutpie draws
#'
#' Diagnostics are extracted directly from the nuts-rs sample-stats schema, so
#' the exact set of fields depends on the installed nuts-rs version and the
#' sampling options used. Count fields (`depth`, `n_steps`, `chain`, `draw`,
#' `index_in_trajectory`) are returned as R `integer` when every value fits
#' in `i32`, else as `numeric`; floating-point fields (`logp`, `energy`,
#' `step_size`, etc.) are always `numeric`. `NA`s use the matching sentinel
#' (`NA_integer_` / `NA_real_`).
#'
#' @section Indexing conventions:
#' `chain` is 1-indexed in `1:num_chains`. `draw` is 1-indexed in
#' `1:num_draws` for post-warmup diagnostics and `1:num_warmup` for warmup
#' diagnostics (returned via [nutpie_warmup_diagnostics()]). Matches
#' `posterior::draws_array` conventions, so a `data.frame` of diagnostics
#' joins cleanly against draws indexed by `(chain, iteration)`.
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `nutpie_diagnostics` object (a named list with a print method).
#'   Commonly available scalar fields: `diverging`, `tuning`, `maxdepth_reached`
#'   (logical); `depth`, `n_steps`, `chain`, `draw`, `index_in_trajectory`
#'   (integer when fits in `i32`, else numeric); `logp`, `energy`,
#'   `energy_error`, `step_size`, `step_size_bar`, `mean_tree_accept`,
#'   `mean_tree_accept_sym` (numeric).
#'   Wide fields (one row per draw): when `store_unconstrained = TRUE`,
#'   `unconstrained_draw` (`NA` rows where unrecorded); when
#'   `store_gradient = TRUE`, `gradient` (`NA` rows where unrecorded); when
#'   `store_mass_matrix = TRUE`, `mass_matrix_inv` (and `mass_matrix_eigvals`
#'   / `mass_matrix_stds` when reported), with the most recently recorded
#'   value carried forward into draws between updates — the inverse mass
#'   matrix is piecewise-constant between adapter steps, not undefined.
#'   These surface as `(n_draws * n_chains, ndim_unc)` numeric matrices when
#'   every recorded row has the same width; mixed-width columns fall back
#'   to a list of numeric vectors (one per draw, `NULL` when not recorded).
#'   With `store_divergences = TRUE`: `divergence_start`, `divergence_end`,
#'   `divergence_momentum`, `divergence_start_gradient` (lists, only
#'   present when at least one draw diverged).
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, num_draws = 1000, num_chains = 4)
#' diag <- nutpie_diagnostics(draws)
#' diag                               # printed summary
#' sum(diag$diverging)                # divergence count
#' max(diag$depth)                    # peak treedepth across all draws
#' }
#' @export
nutpie_diagnostics <- function(draws) {
  diag <- attr(draws, "diagnostics")
  if (is.null(diag)) {
    stop("No diagnostics found. Was this object created by nutpie_sample()?",
         call. = FALSE)
  }
  num_chains <- attr(draws, "num_chains") %||% dim(draws)[[2]] %||% 1L
  structure(diag, class = "nutpie_diagnostics", num_chains = num_chains)
}

#' @export
print.nutpie_diagnostics <- function(x, ...) {
  n <- length(x$diverging %||% x[[1]])
  num_chains <- attr(x, "num_chains") %||% 1L
  n_per_chain <- n %/% num_chains

  cat("Sampler diagnostics\n")
  cat(sprintf("  Draws:         %d (%d per chain, %d chains)\n",
              n, n_per_chain, num_chains))

  if (!is.null(x$diverging)) {
    n_div <- sum(x$diverging, na.rm = TRUE)
    cat(sprintf("  Divergences:   %d", n_div))
    if (n_div > 0) cat(sprintf(" (%.1f%%)", 100 * n_div / n))
    cat("\n")
  }
  if (!is.null(x$maxdepth_reached)) {
    n_md <- sum(x$maxdepth_reached, na.rm = TRUE)
    cat(sprintf("  Max-treedepth hits: %d", n_md))
    if (n_md > 0) cat(sprintf(" (%.1f%%)", 100 * n_md / n))
    cat("\n")
  } else if (!is.null(x$depth)) {
    cat(sprintf("  Max treedepth: %d\n", max(x$depth, na.rm = TRUE)))
  }
  if (!is.null(x$mean_tree_accept)) {
    cat(sprintf("  Mean accept:   %.3f\n", mean(x$mean_tree_accept, na.rm = TRUE)))
  }
  if (!is.null(x$step_size_bar)) {
    step_sizes <- x$step_size_bar[seq(n_per_chain, n, by = n_per_chain)]
    cat(sprintf("  Step size:     %s\n",
                paste(sprintf("%.4f", step_sizes), collapse = ", ")))
  }

  if (!is.null(x$diverging) && sum(x$diverging, na.rm = TRUE) > 0) {
    cat("\n")
    cat("  Warning: divergent transitions detected. Consider increasing\n")
    cat("  target_accept or reparameterizing the model.\n")
  }

  cat(sprintf("\nAvailable fields: %s\n",
              paste(names(x), collapse = ", ")))
  cat("Use `str(nutpie_diagnostics(draws))` for raw diagnostic vectors.\n")
  invisible(x)
}

#' Extract warmup draws from nutpie output
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()] with
#'   `save_warmup = TRUE`.
#' @return A `posterior::draws_array` containing the warmup draws, or `NULL`
#'   if warmup draws were not saved.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, save_warmup = TRUE)
#' warmup <- nutpie_warmup_draws(draws)
#' posterior::summarize_draws(warmup)
#' }
#' @export
nutpie_warmup_draws <- function(draws) {
  warmup <- attr(draws, "warmup_draws")
  if (is.null(warmup)) {
    stop("No warmup draws found. Did you use save_warmup = TRUE?",
         call. = FALSE)
  }
  warmup
}

#' Extract warmup diagnostics from nutpie output
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()] with
#'   `save_warmup = TRUE`.
#' @return A named list of diagnostic vectors for the warmup phase. `chain`
#'   is 1-indexed and `draw` is 1-indexed in `1:num_warmup`; see
#'   [nutpie_diagnostics()] for the full indexing convention.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, save_warmup = TRUE)
#' wd <- nutpie_warmup_diagnostics(draws)
#' max(wd$depth)                       # warmup treedepth peak
#' }
#' @export
nutpie_warmup_diagnostics <- function(draws) {
  diag <- attr(draws, "warmup_diagnostics")
  if (is.null(diag)) {
    stop("No warmup diagnostics found. Did you use save_warmup = TRUE?",
         call. = FALSE)
  }
  diag
}

#' NUTS sampler parameters in bayesplot's long format
#'
#' Reshapes the diagnostics from [nutpie_diagnostics()] into the four-column
#' long-format `data.frame` that bayesplot's NUTS plotting helpers (e.g.
#' `bayesplot::mcmc_pairs(np = ...)`, `bayesplot::mcmc_nuts_divergence()`)
#' expect. Names match Stan's CSV convention (`accept_stat__`,
#' `divergent__`, `treedepth__`, `n_leapfrog__`, `stepsize__`,
#' `energy__`).
#'
#' @param draws A `posterior::draws_array` returned by [nutpie_sample()].
#' @return A `data.frame` with columns:
#'   \describe{
#'     \item{`Chain`}{Integer chain index (1-indexed).}
#'     \item{`Iteration`}{Integer post-warmup draw index (1-indexed
#'       within chain, in `1:num_draws`).}
#'     \item{`Parameter`}{Character; one of `"accept_stat__"`,
#'       `"divergent__"`, `"treedepth__"`, `"n_leapfrog__"`,
#'       `"stepsize__"`, `"energy__"`.}
#'     \item{`Value`}{Numeric value of the corresponding diagnostic.}
#'   }
#'   The data frame has `num_draws * num_chains * 6` rows.
#' @seealso [bayesplot::mcmc_pairs()] for the most common consumer of this
#'   format. [nutpie_diagnostics()] for the raw diagnostics.
#' @examples
#' \dontrun{
#' draws <- nutpie_sample(model, data = dat, num_chains = 4)
#' np <- nutpie_nuts_params(draws)
#' bayesplot::mcmc_pairs(draws, np = np, pars = c("mu", "tau"))
#' }
#' @export
nutpie_nuts_params <- function(draws) {
  diag <- nutpie_diagnostics(draws)
  num_chains <- attr(diag, "num_chains") %||% dim(draws)[[2]] %||% 1L
  n <- length(diag$diverging %||% diag[[1]])
  n_per_chain <- n %/% num_chains

  chain <- as.integer(diag$chain %||% rep(seq_len(num_chains), each = n_per_chain))
  iter  <- as.integer(diag$draw  %||% rep(seq_len(n_per_chain), times = num_chains))

  mk <- function(name, values) {
    data.frame(
      Chain = chain,
      Iteration = iter,
      Parameter = name,
      Value = as.numeric(values),
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, list(
    mk("accept_stat__", diag$mean_tree_accept),
    mk("divergent__",   as.integer(diag$diverging)),
    mk("treedepth__",   diag$depth),
    mk("n_leapfrog__",  diag$n_steps),
    mk("stepsize__",    diag$step_size),
    mk("energy__",      diag$energy)
  ))
}
