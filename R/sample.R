#' Sample from a Stan model using the NUTS sampler
#'
#' Runs the nuts-rs NUTS sampler on a compiled Stan model.
#'
#' @param model A `"nutpie_model"` object from [nutpie_compile_model()],
#'   or a path to a compiled shared library.
#' @param data Model data. Can be:
#'   - A named list (will be converted to JSON via [jsonlite::toJSON()])
#'   - A JSON string
#'   - A path to a `.json` file
#'   - `NULL` for models with no data block
#' @param num_draws Number of post-warmup draws per chain.
#' @param num_warmup Number of warmup (tuning) draws per chain.
#' @param num_chains Number of parallel chains.
#' @param seed Random seed for reproducibility.
#' @param max_treedepth Maximum tree depth for NUTS. The number of leapfrog
#'   steps per draw is at most `2^max_treedepth`.
#' @param target_accept Target acceptance probability for step size adaptation.
#' @param refresh How often to print progress updates, in draws per chain.
#'   Set to `0` to suppress progress output. Default is `100`.
#' @param init Initial values for each chain. Single entry point that
#'   dispatches on the input shape:
#'   \describe{
#'     \item{`NULL` (default)}{Each chain starts from a Uniform(-2, 2) draw on
#'       the unconstrained scale (nuts-rs default).}
#'     \item{Scalar numeric `x`}{Each chain starts from a Uniform(-x, x) draw
#'       on the unconstrained scale. `x = 0` starts every chain at the origin.
#'       Must be non-negative.}
#'     \item{Named list, e.g. `list(mu = 0, sigma = 1)`}{Constrained values
#'       used to start each chain. If fully specified, every chain starts at
#'       the same point. If partial (some parameters missing), each chain
#'       gets its own random fill for the missing parameters (per-chain seeds
#'       derived from `seed`).}
#'     \item{List of `num_chains` named lists}{One constrained start per
#'       chain. Each element may be partial.}
#'     \item{Function `function(chain_id) ...`}{Called once per chain with
#'       `chain_id` in `1:num_chains`; must return a (possibly partial)
#'       named list of constrained parameter values.}
#'     \item{Character path(s)}{A JSON file path, or a character vector of
#'       `num_chains` JSON file paths (CmdStan-style). Parsed values are
#'       treated as constrained named lists.}
#'   }
#'   No jitter is applied; starting points are used exactly (after any
#'   random fill for partial constrained inits). To work on the unconstrained
#'   scale, see [nutpie_unconstrain()] to convert constrained values first.
#'
#'   **Chain assignment is unspecified.** When supplying per-chain starts
#'   (list-of-lists or `function(chain_id)`), the N positions are guaranteed
#'   to be distributed one-per-chain, but the mapping from list index /
#'   `chain_id` to the output chain dimension is not currently guaranteed.
#'   Use this to provide N dispersed starts; do not rely on "chain 1 starts
#'   at element 1" for downstream identifiability. (Threading the true
#'   chain id requires an upstream change in nuts-rs; tracked for a future
#'   release.)
#' @param init_mean Deprecated. Scalar or numeric vector on the unconstrained
#'   scale, with ±0.5 uniform jitter per chain. Use
#'   `init = function(chain_id) ...`, `init = <scalar>` for Uniform(-x, x), or
#'   `init = <named list>` instead. Will be removed in a future version.
#' @param save_warmup If `TRUE`, warmup draws and diagnostics are attached as
#'   attributes. Retrieve them with [nutpie_warmup_draws()].
#' @param cores Number of CPU cores to use for parallel sampling. Defaults to
#'   `min(num_chains, parallel::detectCores())`.
#' @param store_divergences If `TRUE`, store detailed information about each
#'   divergent transition (start/end positions, momentum, gradient). Adds
#'   list columns to diagnostics.
#' @param store_mass_matrix If `TRUE`, store the inverse mass matrix diagonal
#'   at each draw. Adds a list column to diagnostics.
#' @param store_unconstrained If `TRUE`, store the unconstrained position at
#'   each draw as a list column on diagnostics (`unconstrained_draw`). Adds
#'   one `ndim_unc`-length numeric vector per draw — for high-dimensional
#'   models this can rival the draws matrix in size. Default `FALSE`.
#' @param store_gradient If `TRUE`, store the log-density gradient at each
#'   draw as a list column on diagnostics (`gradient`). Same size profile as
#'   `store_unconstrained`. Default `FALSE`.
#' @param pars An optional character vector of block-level parameter names
#'   (e.g. `"beta"`, `"sigma"`). When supplied, only these parameters (or
#'   all parameters *except* these, depending on `include`) are returned in the
#'   output draws. By default (`NULL`), all parameters are returned. Parameter
#'   names should be the Stan block names, not indexed names — e.g. `"beta"`
#'   will match `beta`, `beta[1]`, `beta[2]`, etc.
#'
#'   When the kept set excludes the entire transformed-parameter and/or
#'   generated-quantities blocks, bridgestan is told to skip materializing
#'   those slices per draw. On wide models this is a meaningful speedup and
#'   memory reduction (the GQ block's `*_rng` calls don't run, and the
#'   per-draw constrained buffer shrinks accordingly). One conservative rule:
#'   keeping any GQ name forces TP to also be materialized, since GQ is
#'   permitted to reference TP and we don't want a name lookup to fail
#'   silently.
#' @param include Logical (default `TRUE`). If `TRUE`, `pars` specifies the
#'   parameters to *keep* (whitelist). If `FALSE`, `pars` specifies parameters
#'   to *exclude* (blacklist). Ignored when `pars` is `NULL`.
#' @param low_rank_modified_mass_matrix If `TRUE`, use low-rank modified mass
#'   matrix adaptation. This can improve sampling efficiency for models with
#'   correlated parameters by capturing posterior correlations in the mass
#'   matrix. Default is `FALSE` (diagonal mass matrix).
#' @param mass_matrix_gamma Regularisation parameter for low-rank mass matrix
#'   adaptation. Only used when `low_rank_modified_mass_matrix = TRUE`.
#'   Default is `1e-5`.
#' @param mass_matrix_eigval_cutoff Eigenvalue cutoff for low-rank mass matrix.
#'   Eigenvalues outside `(1/cutoff, cutoff)` are ignored. Only used when
#'   `low_rank_modified_mass_matrix = TRUE`. Default is `2.0`.
#' @return A `posterior::draws_array` with dimensions
#'   `(num_draws, num_chains, n_params)`. Sampler diagnostics are attached
#'   as an attribute and can be retrieved with [nutpie_diagnostics()].
#'   The attributes `"num_warmup"` and `"num_draws"` record the sampling
#'   configuration (accessible via `attr(draws, "num_warmup")` etc.).
#' @export
nutpie_sample <- function(model, data = NULL, num_draws = 1000L,
                          num_warmup = 400L, num_chains = 4L, seed = NULL,
                          max_treedepth = 10L, target_accept = 0.8,
                          refresh = 100L,
                          init = NULL,
                          init_mean = NULL,
                          save_warmup = FALSE, cores = NULL,
                          pars = NULL, include = TRUE,
                          store_divergences = FALSE,
                          store_mass_matrix = FALSE,
                          store_unconstrained = FALSE,
                          store_gradient = FALSE,
                          low_rank_modified_mass_matrix = FALSE,
                          mass_matrix_gamma = 1e-5,
                          mass_matrix_eigval_cutoff = 2.0) {
  lib_path <- resolve_model(model)
  data_json <- resolve_data(data)
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }
  num_draws <- check_count(num_draws, "num_draws", min = 1L)
  num_warmup <- check_count(num_warmup, "num_warmup", min = 0L)
  num_chains <- check_count(num_chains, "num_chains", min = 1L)
  max_treedepth <- check_count(max_treedepth, "max_treedepth", min = 1L)
  if (!is.numeric(target_accept) || length(target_accept) != 1L ||
      !is.finite(target_accept) || target_accept <= 0 || target_accept >= 1) {
    stop("`target_accept` must be a single finite value in (0, 1).",
         call. = FALSE)
  }

  if (is.null(cores)) {
    cores <- min(num_chains, parallel::detectCores())
  }
  cores <- check_count(cores, "cores", min = 1L)

  handle <- bs_open(lib_path, data_json, as.integer(seed))
  init_resolved <- resolve_init(init, init_mean, handle, num_chains,
                                seed = seed)

  flags <- resolve_constrain_flags(handle, pars, include)
  constrain_names <- constrain_names_for_flags(handle, flags$include_tp,
                                               flags$include_gq)
  pars_in_scope <- pars
  if (!is.null(pars_in_scope)) {
    constrain_prefixes <- unique(sub("\\..*", "", constrain_names))
    pars_in_scope <- intersect(pars_in_scope, constrain_prefixes)
    if (length(pars_in_scope) == 0) pars_in_scope <- NULL
  }
  keep_indices <- resolve_keep_indices(constrain_names, pars_in_scope, include)

  raw <- sample_stan(
    handle,
    num_draws,
    num_warmup,
    num_chains,
    as.integer(seed),
    as.integer(max_treedepth),
    as.double(target_accept),
    as.integer(refresh),
    init_resolved$positions,
    isTRUE(init_resolved$jitter),
    isTRUE(save_warmup),
    cores,
    isTRUE(store_divergences),
    isTRUE(store_mass_matrix),
    isTRUE(store_unconstrained),
    isTRUE(store_gradient),
    isTRUE(low_rank_modified_mass_matrix),
    as.double(mass_matrix_gamma),
    as.double(mass_matrix_eigval_cutoff),
    keep_indices,
    isTRUE(flags$include_tp),
    isTRUE(flags$include_gq)
  )
  draws <- matrix_to_draws_array(raw$draws, num_draws, num_chains)
  attr(draws, "diagnostics") <- raw$diagnostics
  attr(draws, "num_chains") <- num_chains
  attr(draws, "num_warmup") <- num_warmup
  attr(draws, "num_draws") <- num_draws

  if (isTRUE(save_warmup) && !is.null(raw$warmup_draws)) {
    warmup <- matrix_to_draws_array(raw$warmup_draws, num_warmup, num_chains)
    attr(draws, "warmup_draws") <- warmup
    attr(draws, "warmup_diagnostics") <- raw$warmup_diagnostics
  }

  n_expand_errors <- raw$expand_errors %||% 0L
  if (n_expand_errors > 0L) {
    warning(
      n_expand_errors, " draw(s) had generated quantities that could not be ",
      "computed (filled with NaN). This typically happens when the sampler ",
      "explores extreme unconstrained values where parameter constraints ",
      "(e.g. bounds) are violated during transformation.",
      call. = FALSE
    )
  }

  draws
}

check_count <- function(x, name, min = 0L) {
  if (length(x) != 1L) {
    stop(sprintf("`%s` must be a single integer; got length %d.",
                 name, length(x)), call. = FALSE)
  }
  if (!is.numeric(x) || !is.finite(x)) {
    stop(sprintf("`%s` must be a finite integer.", name), call. = FALSE)
  }
  if (x != as.integer(x)) {
    stop(sprintf("`%s` must be a whole number; got %s.", name, format(x)),
         call. = FALSE)
  }
  x <- as.integer(x)
  if (x < min) {
    stop(sprintf("`%s` must be >= %d; got %d.", name, min, x),
         call. = FALSE)
  }
  x
}

resolve_model <- function(model) {
  if (inherits(model, "nutpie_model")) {
    return(model$lib_path)
  }
  if (is.character(model) && length(model) == 1L) {
    if (!file.exists(model)) {
      stop("Model library not found: ", model, call. = FALSE)
    }
    return(normalizePath(model))
  }
  stop(
    "`model` must be a nutpie_model object or path to a compiled library.",
    call. = FALSE
  )
}

resolve_data <- function(data) {
  if (is.null(data)) {
    return("")
  }
  if (is.character(data) && length(data) == 1L) {
    # Check if it's a file path
    if (file.exists(data) && grepl("\\.json$", data, ignore.case = TRUE)) {
      return(paste(readLines(data, warn = FALSE), collapse = "\n"))
    }
    # Assume it's a JSON string
    return(data)
  }
  if (is.list(data)) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("Package 'jsonlite' is required to convert list data to JSON.",
           call. = FALSE)
    }
    return(jsonlite::toJSON(data, auto_unbox = TRUE))
  }
  stop("`data` must be NULL, a JSON string, a .json file path, or a named list.",
       call. = FALSE)
}
