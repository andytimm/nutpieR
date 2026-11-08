# All native failure/interrupt cases run under the outer test's process timeout.
library(nutpieR)
args <- commandArgs(trailingOnly = TRUE)
fixtures <- normalizePath(args[[1]], mustWork = TRUE)
core <- normalizePath(args[[2]], mustWork = TRUE)
mode <- args[[3]]
work <- tempfile("density-kernel-lifecycle-")
dir.create(work)
old <- setwd(work)
file.copy(file.path(fixtures, "gaussian.c"), "gaussian.c")
file.copy(file.path(core, "lifecycle.c"), "lifecycle.c")
Sys.setenv(PKG_CPPFLAGS = paste0('-I"', system.file("include", package="nutpieR"), '"'))
status <- system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", "lifecycle.c"),
                  stdout="build.log", stderr="build.log", timeout=90)
if (status != 0L) stop(paste(readLines("build.log"), collapse="\n"))
log <- file.path(work, "events.log")
Sys.setenv(NUTPIER_LIFECYCLE_LOG=log)
ref <- nutpie_compile_model(file.path(fixtures, "gaussian.stan"))
bound <- nutpie_attach_density_kernel(ref,
  normalizePath(paste0("lifecycle", .Platform$dynlib.ext)), list(n=2L, mu=1, sigma=2))
events <- function() {
  if (!file.exists(log)) return(data.frame(kind=character(), id=integer()))
  read.table(log, col.names=c("kind", "id"), colClasses=c("character", "integer"))
}
# Check while the binding is still reachable: every created workspace must
# already be destroyed, once, and only after construction; no bound drop yet.
joined <- function() {
  e <- events()
  created <- e$id[e$kind == "C"]
  destroyed <- e$id[e$kind == "D"]
  stopifnot(!anyDuplicated(created), !anyDuplicated(destroyed),
            identical(sort(created), sort(destroyed)), !any(e$kind == "B"))
  for (id in created) stopifnot(which(e$kind == "C" & e$id == id) <
                                which(e$kind == "D" & e$id == id))
  invisible(e)
}
sample_ok <- function() {
  fit <- nutpie_sample(bound,
    num_draws=20L, num_warmup=200L, num_chains=3L, cores=1L,
    seed=42L, progress="none")
  stopifnot(inherits(fit, "draws_array"), all(is.finite(fit)),
            identical(dim(fit)[1:2], c(20L, 3L)))
  joined()
  invisible(fit)
}
error <- function(expr, pattern) {
  e <- tryCatch({force(expr); NULL}, error=identity)
  stopifnot(inherits(e, "error"), grepl(pattern, conditionMessage(e), ignore.case=TRUE))
  cat("Caught:", conditionMessage(e), "\n")
}
if (mode == "cleanup") {
  sample_ok()
  # Three queued chains plus the schema-template math each own a workspace.
  stopifnot(sum(events()$kind == "C") == 4L)
  for (offset in c(1L, 2L)) {
    before <- nrow(events())
    at <- max(events()$id) + offset
    Sys.setenv(NUTPIER_LIFECYCLE_FAIL_AT=at)
    error(nutpie_sample(bound, num_draws=20L, num_warmup=200L,
      num_chains=3L, cores=1L, seed=42L, progress="none"),
      "injected workspace factory failure")
    Sys.unsetenv("NUTPIER_LIFECYCLE_FAIL_AT")
    e <- joined()
    fresh <- e[seq_len(nrow(e)) > before, ]
    stopifnot(sum(fresh$kind == "F") == 1L, fresh$id[fresh$kind == "F"] == at)
    if (offset == 1L) stopifnot(!any(fresh$kind == "C")) # schema factory
    if (offset == 2L) stopifnot(any(fresh$kind == "C")) # worker factory
    gc()
    sample_ok()
  }
  for (adaptation in c("diag", "low_rank")) {
    fit <- nutpie_sample(bound, num_draws=20L, num_warmup=200L, num_chains=2L,
      cores=1L, seed=42L, progress="none", adaptation=adaptation,
      walnuts=if (adaptation == "diag") TRUE else
        list(energy_criterion="endpoint", target_macro_steps=15L))
    stopifnot(inherits(fit, "draws_array"), all(is.finite(fit)),
              identical(dim(fit)[1:2], c(20L, 2L)))
    joined()
  }
  rm(ref)
  gc()
  sample_ok() # binding owns the reference/library even after caller GC
} else if (mode == "callback") {
  # Bypass only the public R callback guard to test the native host's policy.
  # Normal nutpie_sample() still supplies all actual model and result plumbing.
  local({
    calls <- 0L
    testthat::local_mocked_bindings(
      protect_progress_callback=identity,
      make_text_progress_callback=function(...) function(snapshot) {
        calls <<- calls + 1L
        stop("injected lifecycle callback failure")
      }, .package="nutpieR")
    for (model in list(bound, ref)) {
      calls <- 0L
      output <- capture.output(fit <- nutpie_sample(model,
        data=if (identical(model, ref)) list(n=2L, mu=1, sigma=2) else NULL,
        num_draws=20L, num_warmup=200L, num_chains=3L, cores=1L,
        seed=42L, progress="text", refresh=1L))
      stopifnot(calls == 1L, inherits(fit, "draws_array"), all(is.finite(fit)),
        identical(dim(fit)[1:2], c(20L, 3L)),
        sum(grepl("nutpieR: progress callback failed", output, fixed=TRUE)) == 1L)
      joined()
    }
  })
  sample_ok()
} else if (mode == "callback-interrupt") {
  local({
    calls <- 0L
    testthat::local_mocked_bindings(
      make_text_progress_callback=function(...) function(snapshot) {
        calls <<- calls + 1L
        stop(structure(list(message="injected callback interrupt", call=NULL),
                       class=c("interrupt", "condition")))
      }, .package="nutpieR")
    interrupted <- tryCatch({
      nutpie_sample(bound, num_draws=100000L, num_warmup=200L, num_chains=3L,
        cores=1L, seed=42L, progress="text", refresh=1L)
      NULL
    }, interrupt=identity, error=identity)
    stopifnot(calls == 1L, inherits(interrupted, "condition"),
      inherits(interrupted, "interrupt") ||
        grepl("interrupt", conditionMessage(interrupted), ignore.case=TRUE))
    joined()
  })
  sample_ok()
} else if (mode %in% c("interrupt-none", "interrupt-text", "interrupt-callback")) {
  stopifnot(.Platform$OS.type != "windows")
  progress <- if (mode == "interrupt-none") "none" else "text"
  if (mode == "interrupt-callback") {
    # The helper signals while R executes inside the protected native callback.
    testthat::local_mocked_bindings(
      make_text_progress_callback=function(...) function(snapshot) {
        cat("K 0\n", file=log, append=TRUE)
        Sys.sleep(1)
      }, .package="nutpieR")
  }
  # Keep evaluations alive until the helper observes native worker entry.
  Sys.setenv(NUTPIER_LIFECYCLE_SLOW="1")
  done <- file.path(work, "signal-sent")
  system2(file.path(R.home("bin"), "Rscript"),
    c(shQuote(file.path(core, "lifecycle-signal.R")), Sys.getpid(),
      shQuote(log), shQuote(done), nrow(events()),
      shQuote(if (mode == "interrupt-callback") "^K " else "^E ")),
    stdout=file.path(work, "signal.log"), stderr=file.path(work, "signal.log"), wait=FALSE)
  interrupted <- tryCatch({
    nutpie_sample(bound, num_draws=100000L, num_warmup=200L, num_chains=3L,
      cores=1L, seed=42L, progress=progress, refresh=1L)
    NULL
  }, interrupt=identity, error=identity)
  stopifnot(inherits(interrupted, "condition"),
    inherits(interrupted, "interrupt") || grepl("interrupt", conditionMessage(interrupted), ignore.case=TRUE))
  Sys.unsetenv("NUTPIER_LIFECYCLE_SLOW")
  joined()
  stopifnot(file.exists(done), identical(readLines(done), "sent"))
  gc()
  sample_ok()
} else stop("unknown lifecycle mode")
joined()
# No fit or helper may keep the native binding alive past this explicit GC.
rm(bound)
gc()
gc() # finalizers can release other external pointers for the next collection
e <- events()
cat("Bound destruction count:", sum(e$kind == "B"), "\n")
stopifnot(sum(e$kind == "B") == 1L, tail(e$kind, 1L) == "B")
Sys.unsetenv("NUTPIER_LIFECYCLE_LOG")
setwd(old)
unlink(work, recursive=TRUE)
cat("density kernel lifecycle", mode, "passed\n")
