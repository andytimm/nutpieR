#' Compile a Stan model
#'
#' Compiles a Stan model to a shared library using BridgeStan.
#' Downloads BridgeStan sources on first use (this is slow).
#'
#' @section Caching:
#'
#' By default the compiled `.so` is cached so repeat calls are near-instant:
#'
#' * `stan_file = ...` --- the artifact lives next to the `.stan` as
#'   `<basename>_model.so` (matching cmdstanr's convention). A subsequent
#'   call returns the cached artifact when its mtime is at least as new as
#'   the `.stan`. If the user's `.stan` directory is read-only, the call
#'   falls back transparently to the inline cache below with a warning.
#' * `code = "..."` --- the artifact lives under
#'   [`nutpie_cache_dir()`][nutpie_cache_dir], keyed by a hash of the source
#'   plus BridgeStan version and compile flags. Identical inputs hit the
#'   cache; different inputs get separate entries.
#'
#' Cache controls:
#'
#' * `cache = FALSE` on a single call --- compile fresh, drop artifact in
#'   a per-call tempdir. Result is valid for the R session.
#' * `Sys.setenv(NUTPIER_DISABLE_COMPILE_CACHE = "1")` --- same effect
#'   process-wide, without changing call sites.
#' * [`nutpie_clear_cache()`][nutpie_clear_cache] wipes the inline cache.
#'
#' @param stan_file Path to a `.stan` file. Exactly one of `stan_file` or
#'   `code` must be provided.
#' @param code A string containing Stan model code. If provided, the code is
#'   written to a temporary `.stan` file and compiled.
#' @param stanc_args Character vector of extra arguments passed to the
#'   `stanc` compiler (e.g., `"--O1"` for optimization).
#' @param compile_args Character vector of extra arguments passed to `make`
#'   during compilation.
#' @param verbose Integer controlling compilation output. `0` = silent,
#'   `1` (default) = print status messages.
#'   Note: full make/stanc output (verbose=2) is not yet supported because
#'   bridgestan captures subprocess output internally rather than streaming it.
#' @param cache Logical, default `TRUE`. When `TRUE`, reuse a previously
#'   compiled artifact when the source is unchanged. When `FALSE`, force a
#'   fresh compile and place the artifact in a per-call tempdir.
#' @return An object of class `"nutpie_model"` containing the path to the
#'   compiled shared library.
#' @export
nutpie_compile_model <- function(stan_file = NULL, code = NULL,
                                 stanc_args = character(),
                                 compile_args = character(),
                                 verbose = 1L,
                                 cache = TRUE) {
  if (!is.null(stan_file) && !is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`, not both.", call. = FALSE)
  }
  if (is.null(stan_file) && is.null(code)) {
    stop("Provide exactly one of `stan_file` or `code`.", call. = FALSE)
  }

  if (Sys.which("make") == "") {
    platform_hint <- if (.Platform$OS.type == "windows") {
      "Install Rtools (https://cran.r-project.org/bin/windows/Rtools/) and ensure it is on your PATH."
    } else if (Sys.info()[["sysname"]] == "Darwin") {
      "Install Xcode Command Line Tools: xcode-select --install"
    } else {
      "Install build-essential: sudo apt install build-essential (Debian/Ubuntu) or sudo dnf install make gcc-c++ (Fedora)"
    }
    stop(
      "`make` is required to compile Stan models but was not found on PATH.\n",
      platform_hint,
      call. = FALSE
    )
  }

  verbose <- as.integer(verbose)
  use_cache <- isTRUE(cache) &&
    !identical(Sys.getenv("NUTPIER_DISABLE_COMPILE_CACHE"), "1")

  if (!is.null(code)) {
    return(compile_inline(code, stanc_args, compile_args, verbose, use_cache))
  }

  stan_file <- normalizePath(stan_file, mustWork = TRUE)

  if (use_cache && cache_hit_in_place(stan_file)) {
    if (verbose >= 1L) message("Using cached compiled model.")
    return(structure(
      list(
        lib_path = normalizePath(expected_artifact_path(stan_file),
                                 mustWork = TRUE),
        stan_file = stan_file
      ),
      class = "nutpie_model"
    ))
  }

  if (use_cache) {
    if (!dir_writable(dirname(stan_file))) {
      warning(
        "Stan file directory ", dirname(stan_file), " is not writable; ",
        "falling back to the inline cache. Pass cache = FALSE to suppress.",
        call. = FALSE
      )
      code <- paste(readLines(stan_file, warn = FALSE), collapse = "\n")
      return(compile_inline(code, stanc_args, compile_args, verbose, TRUE))
    }
    lib_path <- compile_stan_file_to_inplace(stan_file, stanc_args,
                                             compile_args, verbose)
  } else {
    lib_path <- compile_stan_file_to_tempdir(stan_file, stanc_args,
                                             compile_args, verbose)
  }

  structure(
    list(lib_path = lib_path, stan_file = stan_file),
    class = "nutpie_model"
  )
}

#' @export
print.nutpie_model <- function(x, ...) {
  cat("nutpie Stan model\n")
  cat("  Stan file:", x$stan_file, "\n")
  cat("  Library:  ", x$lib_path, "\n")
  invisible(x)
}
