# Benchmark R-side result assembly on posteriorDB models.
#
# This isolates the post-Rust conversion path by tracing assemble_sample_result():
#   raw Arrow/Rust result -> posterior::draws_array + diagnostics/warmup attrs.
#
# Usage:
#   devtools::load_all(quiet = TRUE)
#   source("benchmarks/result_assembly_benchmark.R")
#   res <- run_result_assembly_benchmark()
#
# The defaults intentionally use a small but nontrivial posteriorDB subset so the
# script is useful in PR loops. Increase num_draws/num_warmup for release-scale
# memory checks.

run_result_assembly_benchmark <- function(
    posteriors = c(
      "eight_schools-eight_schools_noncentered",
      "diamonds-diamonds",
      "irt_2pl-irt_2pl"
    ),
    num_draws = 100L,
    num_warmup = 100L,
    num_chains = 2L,
    seed = 20260614L,
    save_warmup = FALSE,
    sample_args = list(),
    results_dir = file.path("benchmarks", "results"),
    output_csv = file.path(results_dir, "result_assembly_benchmark.csv")) {
  stopifnot(requireNamespace("posteriordb", quietly = TRUE))
  stopifnot(requireNamespace("posterior", quietly = TRUE))

  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  pdb <- posteriordb::pdb_default()
  rows <- lapply(posteriors, benchmark_result_assembly_one,
                 pdb = pdb,
                 num_draws = num_draws,
                 num_warmup = num_warmup,
                 num_chains = num_chains,
                 seed = seed,
                 save_warmup = save_warmup,
                 sample_args = sample_args)
  out <- do.call(rbind, rows)
  utils::write.csv(out, output_csv, row.names = FALSE)
  message("Wrote ", output_csv)
  out
}

benchmark_result_assembly_one <- function(posterior_name, pdb, num_draws,
                                          num_warmup, num_chains, seed,
                                          save_warmup, sample_args = list()) {
  message("[", posterior_name, "] compile/sample")
  po <- posteriordb::posterior(posterior_name, pdb)
  stan_file <- tempfile(fileext = ".stan")
  writeLines(posteriordb::stan_code(po), stan_file)
  on.exit(unlink(stan_file), add = TRUE)

  data <- posteriordb::pdb_data(po)
  model <- nutpie_compile_model(stan_file, verbose = 0L)

  assembly <- new.env(parent = emptyenv())
  assembly$elapsed_s <- NA_real_
  assembly$alloc_bytes <- NA_real_
  assembly$alloc_events <- NA_integer_
  assembly$memfile <- tempfile("nutpier-assembly-profmem-", fileext = ".out")
  assign(".nutpier_assembly_bench_env", assembly, envir = globalenv())
  on.exit(rm(".nutpier_assembly_bench_env", envir = globalenv()), add = TRUE)
  on.exit(unlink(assembly$memfile), add = TRUE)

  trace(
    "assemble_sample_result",
    where = asNamespace("nutpieR"),
    tracer = quote({
      .nutpier_assembly_bench_env$start <- proc.time()[["elapsed"]]
      utils::Rprofmem(.nutpier_assembly_bench_env$memfile)
    }),
    exit = quote({
      utils::Rprofmem(NULL)
      .nutpier_assembly_bench_env$elapsed_s <-
        proc.time()[["elapsed"]] - .nutpier_assembly_bench_env$start
      mem <- get("parse_rprofmem", envir = globalenv())(
        .nutpier_assembly_bench_env$memfile
      )
      .nutpier_assembly_bench_env$alloc_bytes <- mem$bytes
      .nutpier_assembly_bench_env$alloc_events <- mem$events
    }),
    print = FALSE
  )
  on.exit(untrace("assemble_sample_result", where = asNamespace("nutpieR")), add = TRUE)

  sample_start <- proc.time()[["elapsed"]]
  err <- NULL
  draws <- tryCatch(
    do.call(
      nutpie_sample,
      c(
        list(
          model = model,
          data = data,
          num_draws = num_draws,
          num_warmup = num_warmup,
          num_chains = num_chains,
          seed = seed,
          refresh = 0L,
          progress = "none",
          save_warmup = save_warmup
        ),
        sample_args
      )
    ),
    error = function(e) {
      err <<- conditionMessage(e)
      NULL
    }
  )
  sample_elapsed_s <- proc.time()[["elapsed"]] - sample_start

  if (is.null(draws)) {
    return(data.frame(
      posterior = posterior_name,
      ok = FALSE,
      error = err,
      variables = NA_integer_,
      draws = num_draws,
      chains = num_chains,
      save_warmup = save_warmup,
      sample_args = paste(names(sample_args), unlist(sample_args, use.names = FALSE),
                          sep = "=", collapse = ";"),
      sample_elapsed_s = sample_elapsed_s,
      assembly_elapsed_s = assembly$elapsed_s,
      assembly_alloc_mb = assembly$alloc_bytes / 1024^2,
      assembly_alloc_events = assembly$alloc_events,
      draws_object_mb = NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  data.frame(
    posterior = posterior_name,
    ok = TRUE,
    error = NA_character_,
    variables = length(posterior::variables(draws)),
    draws = num_draws,
    chains = num_chains,
    save_warmup = save_warmup,
    sample_args = paste(names(sample_args), unlist(sample_args, use.names = FALSE),
                        sep = "=", collapse = ";"),
    sample_elapsed_s = sample_elapsed_s,
    assembly_elapsed_s = assembly$elapsed_s,
    assembly_alloc_mb = assembly$alloc_bytes / 1024^2,
    assembly_alloc_events = assembly$alloc_events,
    draws_object_mb = as.numeric(utils::object.size(draws)) / 1024^2,
    stringsAsFactors = FALSE
  )
}

parse_rprofmem <- function(path) {
  if (!file.exists(path)) return(list(bytes = NA_real_, events = NA_integer_))
  lines <- readLines(path, warn = FALSE)
  bytes <- suppressWarnings(as.numeric(sub(" .*", "", lines)))
  bytes <- bytes[is.finite(bytes)]
  list(bytes = sum(bytes), events = length(bytes))
}
