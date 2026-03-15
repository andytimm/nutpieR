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
#' @param progress Whether to show progress bars during sampling.
#' @param init_mean Optional numeric vector of initial values in unconstrained
#'   parameter space. Each chain starts at `init_mean + small jitter`. Length
#'   must match the number of unconstrained parameters. If `NULL` (default),
#'   chains are initialized with `Uniform(-2, 2)`.
#' @param save_warmup If `TRUE`, warmup draws and diagnostics are attached as
#'   attributes. Retrieve them with [nutpie_warmup_draws()].
#' @param cores Number of CPU cores to use for parallel sampling. Defaults to
#'   `min(num_chains, parallel::detectCores())`.
#' @param store_divergences If `TRUE`, store detailed information about each
#'   divergent transition (start/end positions, momentum, gradient). Adds
#'   list columns to diagnostics.
#' @param store_mass_matrix If `TRUE`, store the inverse mass matrix diagonal
#'   at each draw. Adds a list column to diagnostics.
#' @return A `posterior::draws_array` with dimensions
#'   `(num_draws, num_chains, n_params)`. Sampler diagnostics are attached
#'   as an attribute and can be retrieved with [nutpie_diagnostics()].
#' @export
nutpie_sample <- function(model, data = NULL, num_draws = 1000L,
                          num_warmup = 400L, num_chains = 4L, seed = NULL,
                          max_treedepth = 10L, target_accept = 0.8,
                          progress = TRUE, init_mean = NULL,
                          save_warmup = FALSE, cores = NULL,
                          store_divergences = FALSE,
                          store_mass_matrix = FALSE) {
  lib_path <- resolve_model(model)
  data_json <- resolve_data(data)
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }
  num_draws <- as.integer(num_draws)
  num_warmup <- as.integer(num_warmup)
  num_chains <- as.integer(num_chains)

  if (is.null(cores)) {
    cores <- min(num_chains, parallel::detectCores())
  }
  cores <- as.integer(cores)

  raw <- sample_stan(
    lib_path,
    data_json,
    num_draws,
    num_warmup,
    num_chains,
    as.integer(seed),
    as.integer(max_treedepth),
    as.double(target_accept),
    isTRUE(progress),
    init_mean,
    isTRUE(save_warmup),
    cores,
    isTRUE(store_divergences),
    isTRUE(store_mass_matrix)
  )
  draws <- matrix_to_draws_array(raw$draws, num_draws, num_chains)
  attr(draws, "diagnostics") <- raw$diagnostics
  attr(draws, "num_chains") <- num_chains

  if (isTRUE(save_warmup) && !is.null(raw$warmup_draws)) {
    warmup <- matrix_to_draws_array(raw$warmup_draws, num_warmup, num_chains)
    attr(draws, "warmup_draws") <- warmup
    attr(draws, "warmup_diagnostics") <- raw$warmup_diagnostics
  }

  draws
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
