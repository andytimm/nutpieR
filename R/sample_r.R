#' Sample an arbitrary R log-density with nutpie's NUTS
#'
#' Run nuts-rs NUTS over a log-posterior supplied as plain R functions, instead
#' of a compiled Stan model. This lets you put models that already expose an R
#' density + gradient — e.g. a preconditioned `RTMB`/`TMB` objective — through
#' nutpie's adaptation, in the spirit of
#' [StanEstimators](https://github.com/andrjohns/StanEstimators).
#'
#' **Single chain only.** R is single-threaded, so the density callbacks must run
#' on the main R thread; the parallel, multi-chain machinery used by
#' [nutpie_sample()] is not available here. Run this several times (varying
#' `seed`) if you want multiple chains.
#'
#' @param fn Optional `function(y)` returning the log-posterior density at the
#'   unconstrained vector `y` (higher = more probable). For a TMB objective
#'   `obj$fn` (a *negative* log-likelihood), pass `function(y) -obj$fn(y)`.
#'   Required unless `value_grad` is supplied.
#' @param grad Optional `function(y)` returning the gradient of `fn` as a numeric
#'   vector of length `ndim`. Required unless `value_grad` is supplied.
#' @param value_grad Optional combined callback `function(y)` returning a numeric
#'   vector `c(logp, gradient)`, length `ndim + 1`. This is the preferred fast
#'   path for expensive transformed densities because it crosses into R once per
#'   leapfrog step and can share intermediate work between value and gradient.
#' @param ndim Integer dimension of `y`. If `init` is supplied as a numeric
#'   vector, `ndim` defaults to its length.
#' @param init Starting position. Either a numeric vector of length `ndim`, or a
#'   `function(chain_id)` returning one (called once with `1L`). Defaults to
#'   `rnorm(ndim)`.
#' @param num_draws Post-warmup draws to keep. Default 1000.
#' @param num_warmup Warmup (tuning) draws. Must be at least 1. Default 1000.
#' @param seed Integer RNG seed. Default random.
#' @param save_warmup Whether to retain warmup draws + diagnostics
#'   (see [nutpie_warmup_draws()]). Default `FALSE`.
#' @param max_treedepth Optional NUTS maximum tree depth.
#' @param target_accept Optional target acceptance probability in `(0, 1)`.
#' @param expand A `function(y)` mapping an unconstrained draw to the values you
#'   want reported — e.g. the back-transform from a preconditioned space, or
#'   derived quantities. Called once per kept draw (never in the leapfrog hot
#'   loop). Should return a numeric vector; if named, the names become the
#'   variable names. Default: report `y` as `y1..y{ndim}`.
#' @param progress Whether to print a periodic one-line status to the console.
#'   Defaults to `TRUE` in interactive sessions.
#'
#' @return A [posterior::draws_array] carrying the same diagnostics attributes
#'   that [nutpie_diagnostics()] and the posterior/bayesplot tooling consume, so
#'   they work as usual. Diagnostics cover divergences, leapfrog count
#'   (`n_steps`), step size, tree depth, energy, `logp`, and mean acceptance.
#'   (Unlike [nutpie_sample()] output, no `sampler_config` attribute is attached.)
#'   The `callback_stats` attribute reports R-callback evaluation counts and
#'   timing, useful for separating sampler geometry from implementation overhead.
#'
#' @seealso [nutpie_sample()] for Stan models.
#' @export
nutpie_sample_r <- function(fn = NULL, grad = NULL, value_grad = NULL,
                            ndim = NULL, init = NULL,
                            num_draws = 1000L, num_warmup = 1000L,
                            seed = NULL, save_warmup = FALSE,
                            max_treedepth = NULL, target_accept = NULL,
                            expand = NULL, progress = interactive()) {
  if (is.null(value_grad)) {
    if (!is.function(fn)) stop("`fn` must be a function.", call. = FALSE)
    if (!is.function(grad)) stop("`grad` must be a function.", call. = FALSE)
  } else {
    if (!is.function(value_grad)) stop("`value_grad` must be a function.", call. = FALSE)
    if (!is.null(fn) && !is.function(fn)) stop("`fn` must be NULL or a function.", call. = FALSE)
    if (!is.null(grad) && !is.function(grad)) stop("`grad` must be NULL or a function.", call. = FALSE)
  }
  if (!is.null(expand) && !is.function(expand)) {
    stop("`expand` must be NULL or a function.", call. = FALSE)
  }

  num_draws <- check_count(num_draws, "num_draws", min = 1L)
  num_warmup <- check_count(num_warmup, "num_warmup", min = 1L)
  max_treedepth <- check_optional_count(max_treedepth, "max_treedepth", min = 1L)
  target_accept <- check_optional_probability(target_accept, "target_accept")
  save_warmup <- check_flag(save_warmup, "save_warmup")
  progress <- check_flag(progress, "progress")
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }
  seed <- check_count(seed, "seed", min = 0L, max = .Machine$integer.max)

  init_pos <- resolve_r_init(init, ndim, seed)
  ndim <- length(init_pos)

  raw <- sample_r_density(
    logp_fn = fn,
    grad_fn = grad,
    value_grad_fn = value_grad,
    ndim = ndim,
    init = init_pos,
    num_draws = num_draws,
    num_warmup = num_warmup,
    seed = seed,
    save_warmup = save_warmup,
    max_treedepth = max_treedepth,
    target_accept = target_accept,
    progress = progress
  )

  assemble_r_sample_result(raw, num_draws, num_warmup, save_warmup, expand)
}

#' Resolve the `init`/`ndim` arguments to a starting position vector.
#'
#' A `NULL` init defaults to `rnorm(ndim)`, drawn deterministically from `seed`
#' (so a given seed reproduces the whole run) without disturbing the caller's
#' global RNG stream.
#' @noRd
resolve_r_init <- function(init, ndim, seed) {
  if (is.function(init)) {
    init <- init(1L)
  }
  if (is.null(init)) {
    if (is.null(ndim)) {
      stop("Supply either `ndim` or an `init` vector so the dimension is known.",
           call. = FALSE)
    }
    ndim <- check_count(ndim, "ndim", min = 1L)
    return(with_local_seed(seed, stats::rnorm(ndim)))
  }
  if (!is.numeric(init)) {
    stop("`init` must be a numeric vector or a function returning one.",
         call. = FALSE)
  }
  init <- as.numeric(init)
  if (!all(is.finite(init))) {
    stop("`init` contains non-finite values.", call. = FALSE)
  }
  if (!is.null(ndim) && length(init) != ndim) {
    stop(sprintf("`init` has length %d but `ndim` is %d.", length(init), ndim),
         call. = FALSE)
  }
  init
}

#' Turn the raw Rust list into a posterior::draws_array with diagnostics.
#' @noRd
assemble_r_sample_result <- function(raw, num_draws, num_warmup, save_warmup,
                                     expand) {
  draws <- r_draws_array(raw$draws, num_draws, raw$ndim, expand)

  diag <- raw$diagnostics
  diag$chain <- rep(1L, num_draws)
  diag$draw <- seq_len(num_draws)
  attr(draws, "diagnostics") <- diag
  attr(draws, "num_chains") <- 1L
  attr(draws, "num_warmup") <- num_warmup
  attr(draws, "num_draws") <- num_draws
  attr(draws, "callback_stats") <- raw$callback_stats

  if (save_warmup && !is.null(raw$warmup_draws)) {
    attr(draws, "warmup_draws") <-
      r_draws_array(raw$warmup_draws, num_warmup, raw$ndim, expand)
    wdiag <- raw$warmup_diagnostics
    wdiag$chain <- rep(1L, num_warmup)
    wdiag$draw <- seq_len(num_warmup)
    attr(draws, "warmup_diagnostics") <- wdiag
  }

  draws
}

#' Build a single-chain draws_array from the flat (draw-major) Rust buffer,
#' applying `expand` per draw if supplied.
#' @noRd
r_draws_array <- function(flat, n_draws, ndim, expand) {
  if (n_draws == 0L) return(NULL)
  m <- matrix(flat, nrow = n_draws, ncol = ndim, byrow = TRUE)

  if (is.null(expand)) {
    var_names <- paste0("y", seq_len(ndim))
  } else {
    first <- expand(m[1L, ])
    nm <- names(first)
    first <- as.numeric(first)
    width <- length(first)
    out <- matrix(0, nrow = n_draws, ncol = width)
    out[1L, ] <- first
    for (i in seq_len(n_draws)[-1L]) {
      v <- as.numeric(expand(m[i, ]))
      # Guard against a varying-length return: `out[i, ] <- v` would silently
      # recycle a too-short `v` (e.g. length 1) across the row.
      if (length(v) != width) {
        stop(sprintf(
          "`expand` returned length %d on draw %d but length %d on draw 1; it must return a fixed-length vector.",
          length(v), i, width
        ), call. = FALSE)
      }
      out[i, ] <- v
    }
    m <- out
    var_names <- if (!is.null(nm)) nm else paste0("v", seq_len(ncol(m)))
  }

  # posterior requires unique variable names; RTMB/TMB vector parameters share a
  # base name (e.g. `eta`, `eta`, ...), so disambiguate rather than error deep in
  # as_draws_array.
  if (anyDuplicated(var_names)) {
    var_names <- make.unique(var_names, sep = "")
  }

  arr <- array(m, dim = c(n_draws, 1L, ncol(m)),
               dimnames = list(iteration = seq_len(n_draws),
                               chain = 1L, variable = var_names))
  posterior::as_draws_array(arr)
}
