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
#' @param num_chains Number of parallel chains.
#' @param seed Random seed for reproducibility.
#' @return A matrix of draws with dimensions `(num_draws * num_chains)` by
#'   number of parameters. Column names are the constrained parameter names
#'   (including transformed parameters and generated quantities).
#' @export
nutpie_sample <- function(model, data = NULL, num_draws = 1000L,
                          num_chains = 4L, seed = NULL) {
  lib_path <- resolve_model(model)
  data_json <- resolve_data(data)
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }
  sample_stan(
    lib_path,
    data_json,
    as.integer(num_draws),
    as.integer(num_chains),
    as.integer(seed)
  )
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
