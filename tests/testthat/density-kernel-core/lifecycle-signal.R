# Separate process: signal only after a worker has entered evaluation.
args <- commandArgs(trailingOnly = TRUE)
pid <- as.integer(args[[1]])
log <- args[[2]]
done <- args[[3]]
start <- as.integer(args[[4]])
pattern <- args[[5]]
deadline <- Sys.time() + 10
repeat {
  lines <- if (file.exists(log)) readLines(log, warn = FALSE) else character()
  if (any(grepl(pattern, lines[seq_along(lines) > start]))) {
    Sys.sleep(0.3)
    writeLines("sent", done)
    tools::pskill(pid, tools::SIGINT)
    quit(status = 0L)
  }
  if (Sys.time() >= deadline) stop("worker never entered evaluation")
  Sys.sleep(0.02)
}
