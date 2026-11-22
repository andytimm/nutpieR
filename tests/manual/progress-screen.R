# Driven by progress-screen.py in a fresh PTY.
source("R/progress.R")
if (utils::packageVersion("cli") < "3.6.6.9000") stop("Development cli required")
options(cli.progress_handlers = "cli", cli.progress_show_after = 0,
        cli.progress_multiline = TRUE, cli.unicode = FALSE,
        cli.width = as.integer(Sys.getenv("COLUMNS")))
# The driver splits the byte stream here without printing checkpoint labels.
checkpoint <- function() cat("\036", file = stderr())
snapshot <- function(draws, div = 0L) {
  lapply(seq_along(draws), function(i) list(
    chain = i, finished_draws = draws[i], total_draws = 100L,
    started = draws[i] > 0, tuning = draws[i] < 50,
    divergences = if (i == 1L) div else 0L,
    total_num_gradients = 7 * draws[i], latest_num_steps = 7L
  ))
}
# Start with the cursor at the bottom, as it commonly is in a console.
writeLines(sprintf("prior %d", seq_len(as.integer(commandArgs(TRUE)[2]))), stderr())
cb <- make_cli_progress_callback(4L, 50L, 50L)
cb(snapshot(c(40L, 30L, 20L, 0L)))
checkpoint()
resize <- commandArgs(TRUE)[1] == "resize"
if (resize) stopifnot(system("stty rows 4 < /dev/tty") == 0L)
cb(snapshot(c(75L, 45L, 40L, 5L), div = 1L))
cb(snapshot(c(80L, 45L, 40L, 5L), div = 1L))
checkpoint()
if (resize) {
  stopifnot(system("stty rows 24 < /dev/tty") == 0L)
  cb(snapshot(c(90L, 65L, 55L, 25L), div = 1L))
  checkpoint()
}
success <- commandArgs(TRUE)[1] == "success"
if (success) cb(snapshot(rep(100L, 4L), div = 1L))
finish_progress_callback(cb, success = success)
cat("after sampling\n", file = stderr())
