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
#'   `<basename>_model.so` (matching cmdstanr's convention), with a small
#'   `<basename>_model.cache_meta` sidecar tracking BridgeStan version and
#'   compile flags. A subsequent call hits the cache when the artifact
#'   mtime is at least as new as the `.stan` *and* the sidecar matches.
#'   If the user's `.stan` directory is read-only, the call falls back
#'   transparently to the inline cache below with a warning.
#' * `code = "..."` --- the artifact lives under
#'   [`nutpie_cache_dir()`][nutpie_cache_dir], keyed by a hash of the source
#'   plus BridgeStan version and compile flags (argument order preserved).
#'
#' Cache controls:
#'
#' * `cache = FALSE` on a single call --- force a fresh compile that
#'   overwrites the cached artifact and updates the sidecar.
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
#'   compiled artifact when the source, BridgeStan version, and compile
#'   flags all match. When `FALSE`, force a fresh compile.
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

  if (!dir_writable(dirname(stan_file))) {
    if (use_cache) {
      warning(
        "Stan file directory ", dirname(stan_file), " is not writable; ",
        "falling back to the inline cache. Pass cache = FALSE to suppress.",
        call. = FALSE
      )
    }
    code <- paste(readLines(stan_file, warn = FALSE), collapse = "\n")
    return(compile_inline(code, stanc_args, compile_args, verbose, use_cache))
  }

  bs <- bs_version()
  if (use_cache && in_place_hit(stan_file, bs, stanc_args, compile_args)) {
    if (verbose >= 1L) message("Using cached compiled model.")
    return(nutpie_model(
      normalizePath(expected_artifact_path(stan_file), mustWork = TRUE),
      stan_file
    ))
  }

  lib_path <- compile_in_place(stan_file, stanc_args, compile_args, verbose)
  write_cache_meta(stan_file, bs, stanc_args, compile_args)
  nutpie_model(lib_path, stan_file)
}

#' @export
print.nutpie_model <- function(x, ...) {
  cat("nutpie Stan model\n")
  cat("  Stan file:", x$stan_file, "\n")
  cat("  Library:  ", x$lib_path, "\n")
  invisible(x)
}
