# Compile-time caching for nutpie_compile_model().
#
# Two cache flavours:
#
#   stan_file = ...   -- in-place: <stan_dir>/<basename>_model.so lives next
#                        to the user's .stan, mtime-guarded. Matches cmdstanr's
#                        convention so users with mixed-backend workflows get
#                        the same artifact layout.
#
#   code = "..."      -- content-hashed cache under R_user_dir("nutpieR",
#                        "cache")/models/<hash16>/. Hash key folds in
#                        bridgestan_version() and the sorted compile flags so
#                        a BridgeStan upgrade invalidates entries automatically.
#
# Both flavours route their physical compile through compile_in_place(), which
# uses utils::shortPathName() on Windows to feed `make` a no-space path.

# bridgestan emits "<basename>_model.so" on every platform (Stan's makefile
# does not branch on OS extension). Treat the artifact name as a constant.
expected_artifact_path <- function(stan_file) {
  base <- tools::file_path_sans_ext(basename(stan_file))
  file.path(dirname(stan_file), paste0(base, "_model.so"))
}

cache_hit_in_place <- function(stan_file) {
  artifact <- expected_artifact_path(stan_file)
  if (!file.exists(artifact)) return(FALSE)
  file.info(artifact)$mtime >= file.info(stan_file)$mtime
}

# Probe whether a directory accepts a fresh file. We need this for two
# failure modes the move-back step would otherwise hit late: the user's
# .stan is in a read-only location, or the cache_dir parent is on a
# locked-down volume. Both are recoverable -- we fall back to the inline
# cache -- but only if we detect them before invoking make.
dir_writable <- function(path) {
  if (!dir.exists(path)) return(FALSE)
  probe <- tempfile(tmpdir = path)
  ok <- tryCatch(file.create(probe),
                 error = function(e) FALSE,
                 warning = function(w) FALSE)
  if (isTRUE(ok)) unlink(probe)
  isTRUE(ok)
}

# Run `make` on the .stan file at `stan_path`. `stan_path` must be the path
# we want the artifact to land next to -- bridgestan writes "<basename>_model.so"
# alongside the input. Returns the canonicalized artifact path.
compile_in_place <- function(stan_path, stanc_args, compile_args, verbose) {
  build_path <- stan_path
  if (.Platform$OS.type == "windows") {
    # Apply shortPathName to the directory only. 8.3 short names truncate
    # the file extension to 3 chars (".stan" -> ".STA"), which bridgestan
    # rejects with "File must be a .stan file". The directory pieces are
    # the source of any spaces; the basename we control upstream.
    short_dir <- utils::shortPathName(dirname(build_path))
    build_path <- file.path(short_dir, basename(build_path))
  }
  if (grepl(" ", build_path)) {
    stop(
      "Could not resolve a no-space build path for ", stan_path, ". ",
      "On Windows, ensure 8.3 short names are enabled on the volume, or ",
      "set R_USER_CACHE_DIR to a no-space path.",
      call. = FALSE
    )
  }

  if (verbose >= 1L) {
    message("Compiling Stan model...")
    start_time <- proc.time()[["elapsed"]]
  }
  built_so <- compile_stan_model(
    build_path,
    as.character(stanc_args),
    as.character(compile_args)
  )
  if (verbose >= 1L) {
    elapsed <- proc.time()[["elapsed"]] - start_time
    message(sprintf("Compiled in %.1fs", elapsed))
  }

  normalizePath(built_so, mustWork = TRUE)
}

# Pick a staging basename derived from the user's stan file. Falls back to
# a safe constant if the user's basename has spaces, which would survive
# directory-only shortPathName resolution and break `make`.
staging_basename <- function(stan_file) {
  bn <- basename(stan_file)
  if (grepl(" ", bn)) "model.stan" else bn
}

# stan_file + cache: build in a clean tempdir, move the .so to the in-place
# slot. Building in a tempdir keeps `*.o`/`*.d`/`*.hpp` intermediates out of
# the user's project tree -- only `<basename>_model.so` lands next to .stan.
compile_stan_file_to_inplace <- function(stan_file, stanc_args,
                                         compile_args, verbose) {
  build_dir <- tempfile("nutpieR-build-")
  dir.create(build_dir, recursive = TRUE)
  on.exit(unlink(build_dir, recursive = TRUE), add = TRUE)

  build_stan <- file.path(build_dir, staging_basename(stan_file))
  if (!file.copy(stan_file, build_stan, overwrite = TRUE)) {
    stop("Could not copy ", stan_file, " into staging dir ", build_dir,
         call. = FALSE)
  }

  built_so <- compile_in_place(build_stan, stanc_args, compile_args, verbose)

  target <- expected_artifact_path(stan_file)
  if (!file.copy(built_so, target, overwrite = TRUE)) {
    stop("Could not move compiled artifact to ", target,
         ". Is the directory writable?", call. = FALSE)
  }
  normalizePath(target, mustWork = TRUE)
}

# stan_file + no cache: build in a per-call tempdir and leave the artifact
# there. R cleans up tempdir at session end; the path stays valid for the
# session, which is all we promise when the user has opted out of caching.
compile_stan_file_to_tempdir <- function(stan_file, stanc_args,
                                         compile_args, verbose) {
  build_dir <- tempfile("nutpieR-build-")
  dir.create(build_dir, recursive = TRUE)

  build_stan <- file.path(build_dir, staging_basename(stan_file))
  if (!file.copy(stan_file, build_stan, overwrite = TRUE)) {
    stop("Could not copy ", stan_file, " into staging dir ", build_dir,
         call. = FALSE)
  }
  compile_in_place(build_stan, stanc_args, compile_args, verbose)
}

# Resolve (and lazily create) the inline-code cache root. R_user_dir is the
# preferred location; if it isn't writable -- locked-down profiles, sandboxed
# CI, etc. -- fall back to a session-scoped tempdir with a one-time warning.
inline_cache_dir <- function() {
  base <- tryCatch(
    tools::R_user_dir("nutpieR", "cache"),
    error = function(e) NULL
  )
  use_fallback <- TRUE
  if (!is.null(base)) {
    created <- tryCatch(
      dir.create(base, showWarnings = FALSE, recursive = TRUE),
      error = function(e) FALSE,
      warning = function(w) FALSE
    )
    if ((isTRUE(created) || dir.exists(base)) && dir_writable(base)) {
      use_fallback <- FALSE
    }
  }
  if (use_fallback) {
    if (!isTRUE(getOption("nutpieR.warned_cache_fallback"))) {
      warning(
        "Could not write to R_user_dir for nutpieR cache; falling back ",
        "to a session-scoped tempdir. Cached models will not persist across ",
        "R sessions.", call. = FALSE
      )
      options(nutpieR.warned_cache_fallback = TRUE)
    }
    base <- file.path(tempdir(), "nutpieR-cache")
    dir.create(base, showWarnings = FALSE, recursive = TRUE)
  }
  models_dir <- file.path(base, "models")
  dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)
  models_dir
}

inline_cache_key <- function(content, bs_version, stanc_args, compile_args) {
  payload <- list(
    content = content,
    bs_version = bs_version,
    stanc_args = sort(as.character(stanc_args)),
    compile_args = sort(as.character(compile_args))
  )
  substr(digest::digest(payload, algo = "sha256"), 1L, 16L)
}

compile_inline <- function(code, stanc_args, compile_args, verbose, use_cache) {
  if (!use_cache) {
    build_dir <- tempfile("nutpieR-build-")
    dir.create(build_dir, recursive = TRUE)
    stan_file <- file.path(build_dir, "model.stan")
    writeLines(code, stan_file)
    lib_path <- compile_in_place(stan_file, stanc_args, compile_args, verbose)
    return(structure(
      list(lib_path = lib_path, stan_file = stan_file),
      class = "nutpie_model"
    ))
  }

  bs_version <- bridgestan_version()
  key <- inline_cache_key(code, bs_version, stanc_args, compile_args)
  cache_root <- inline_cache_dir()
  cache_dir <- file.path(cache_root, key)
  cache_stan <- file.path(cache_dir, "model.stan")
  cache_artifact <- file.path(cache_dir, "model_model.so")

  if (file.exists(cache_artifact) && file.exists(cache_stan)) {
    if (verbose >= 1L) message("Using cached compiled model.")
    return(structure(
      list(
        lib_path = normalizePath(cache_artifact, mustWork = TRUE),
        stan_file = normalizePath(cache_stan, mustWork = TRUE)
      ),
      class = "nutpie_model"
    ))
  }

  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(code, cache_stan)
  lib_path <- compile_in_place(cache_stan, stanc_args, compile_args, verbose)
  structure(
    list(lib_path = lib_path,
         stan_file = normalizePath(cache_stan, mustWork = TRUE)),
    class = "nutpie_model"
  )
}

#' Clear the nutpieR inline-code compile cache
#'
#' Removes the entire `R_user_dir("nutpieR", "cache")` tree, where
#' `nutpie_compile_model(code = ...)` stores its compiled artifacts. Inline
#' models will be recompiled on next use. Models compiled from a `stan_file`
#' are not affected -- those live next to the `.stan`; remove the
#' corresponding `<basename>_model.so` directly to invalidate.
#'
#' @return Invisibly `NULL`.
#' @export
nutpie_clear_cache <- function() {
  base <- tryCatch(
    tools::R_user_dir("nutpieR", "cache"),
    error = function(e) NULL
  )
  if (!is.null(base) && dir.exists(base)) {
    unlink(base, recursive = TRUE, force = TRUE)
  }
  fallback <- file.path(tempdir(), "nutpieR-cache")
  if (dir.exists(fallback)) {
    unlink(fallback, recursive = TRUE, force = TRUE)
  }
  invisible(NULL)
}

#' Path to the nutpieR inline-code compile cache directory
#'
#' Returns the directory under which `nutpie_compile_model(code = ...)`
#' stores its hashed compile artifacts. Useful for inspection,
#' troubleshooting, or `unlink()`-ing a single entry.
#'
#' @return A character string with the path to the cache root.
#' @export
nutpie_cache_dir <- function() {
  inline_cache_dir()
}
