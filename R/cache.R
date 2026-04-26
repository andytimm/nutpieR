# Compile-time caching for nutpie_compile_model().
#
# Two cache flavours:
#
#   stan_file = ...   in-place artifact next to the .stan, mtime-guarded.
#                     Matches cmdstanr's convention so users with mixed
#                     backends get the same artifact layout.
#   code = "..."      content-hashed cache under R_user_dir("nutpieR",
#                     "cache")/models/<hash16>/, with bridgestan_version()
#                     and sorted compile flags folded into the key so a
#                     toolchain bump invalidates entries automatically.

INLINE_STAN <- "model.stan"

.cache_state <- new.env(parent = emptyenv())

# bridgestan emits "<basename>_model.so" on every platform (Stan's makefile
# does not branch on OS extension). Treat the artifact name as a constant.
expected_artifact_path <- function(stan_file) {
  base <- tools::file_path_sans_ext(basename(stan_file))
  file.path(dirname(stan_file), paste0(base, "_model.so"))
}

# Returns the artifact path on a fresh in-place hit, NULL on miss.
in_place_hit <- function(stan_file) {
  artifact <- expected_artifact_path(stan_file)
  info <- file.info(c(artifact, stan_file))
  if (is.na(info$mtime[1L]) || info$mtime[1L] < info$mtime[2L]) return(NULL)
  artifact
}

dir_writable <- function(path) {
  dir.exists(path) && file.access(path, mode = 2L) == 0L
}

nutpie_model <- function(lib_path, stan_file) {
  structure(
    list(lib_path = lib_path, stan_file = stan_file),
    class = "nutpie_model"
  )
}

bs_version <- function() {
  if (is.null(.cache_state$bs_version)) {
    .cache_state$bs_version <- bridgestan_version()
  }
  .cache_state$bs_version
}

# Cache the resolved inline-cache root per-process, keyed on R_USER_CACHE_DIR
# so an envvar flip in tests (or interactively) re-resolves cleanly.
inline_cache_dir <- function() {
  env <- Sys.getenv("R_USER_CACHE_DIR", unset = "")
  if (!is.null(.cache_state$inline_root) &&
      identical(.cache_state$inline_root_env, env)) {
    return(.cache_state$inline_root)
  }

  base <- tools::R_user_dir("nutpieR", "cache")
  suppressWarnings(dir.create(base, recursive = TRUE))
  if (!dir_writable(base)) {
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
  models <- file.path(base, "models")
  dir.create(models, showWarnings = FALSE, recursive = TRUE)
  .cache_state$inline_root <- models
  .cache_state$inline_root_env <- env
  models
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

compile_in_place <- function(stan_path, stanc_args, compile_args, verbose) {
  build_path <- stan_path
  if (.Platform$OS.type == "windows") {
    # 8.3 short names truncate ".stan" to ".STA", which bridgestan rejects.
    # Apply shortPathName to the directory only -- the basename is always
    # "model.stan" (forced by compile_via_staging) so it has no spaces.
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
    message(sprintf("Compiled in %.1fs",
                    proc.time()[["elapsed"]] - start_time))
  }
  normalizePath(built_so, mustWork = TRUE)
}

# Build a .stan file in a clean tempdir. If `target` is non-NULL the .so is
# moved there and the staging dir is unlinked; otherwise the artifact stays
# in tempdir for the session (R cleans it up at session end).
compile_via_staging <- function(stan_file, stanc_args, compile_args, verbose,
                                target = NULL) {
  build_dir <- tempfile("nutpieR-build-")
  dir.create(build_dir, recursive = TRUE)
  if (!is.null(target)) {
    on.exit(unlink(build_dir, recursive = TRUE), add = TRUE)
  }

  build_stan <- file.path(build_dir, INLINE_STAN)
  if (!file.copy(stan_file, build_stan, overwrite = TRUE)) {
    stop("Could not copy ", stan_file, " into staging dir ", build_dir,
         call. = FALSE)
  }

  built_so <- compile_in_place(build_stan, stanc_args, compile_args, verbose)
  if (is.null(target)) return(built_so)

  # Try rename first (zero-copy on same volume); fall back to copy across
  # devices or when the target slot is locked.
  if (!suppressWarnings(file.rename(built_so, target)) &&
      !file.copy(built_so, target, overwrite = TRUE)) {
    stop("Could not move compiled artifact to ", target,
         ". Is the directory writable?", call. = FALSE)
  }
  normalizePath(target, mustWork = TRUE)
}

compile_inline <- function(code, stanc_args, compile_args, verbose, use_cache) {
  if (!use_cache) {
    build_dir <- tempfile("nutpieR-build-")
    dir.create(build_dir, recursive = TRUE)
    stan_file <- file.path(build_dir, INLINE_STAN)
    writeLines(code, stan_file)
    return(nutpie_model(
      compile_in_place(stan_file, stanc_args, compile_args, verbose),
      stan_file
    ))
  }

  key <- inline_cache_key(code, bs_version(), stanc_args, compile_args)
  cache_dir <- file.path(inline_cache_dir(), key)
  cache_stan <- file.path(cache_dir, INLINE_STAN)
  cache_artifact <- expected_artifact_path(cache_stan)

  if (file.exists(cache_artifact) && file.exists(cache_stan)) {
    if (verbose >= 1L) message("Using cached compiled model.")
    return(nutpie_model(
      normalizePath(cache_artifact, mustWork = TRUE),
      normalizePath(cache_stan, mustWork = TRUE)
    ))
  }

  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(code, cache_stan)
  nutpie_model(
    compile_in_place(cache_stan, stanc_args, compile_args, verbose),
    normalizePath(cache_stan, mustWork = TRUE)
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
  for (d in c(tools::R_user_dir("nutpieR", "cache"),
              file.path(tempdir(), "nutpieR-cache"))) {
    if (dir.exists(d)) unlink(d, recursive = TRUE, force = TRUE)
  }
  rm(list = ls(.cache_state), envir = .cache_state)
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
