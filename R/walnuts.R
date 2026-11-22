# Resolve experimental WALNUTS controls without copying upstream defaults into
# the payload. Rust overlays supplied fields on WalnutsOptions::default().
resolve_walnuts <- function(walnuts) {
  if (is.null(walnuts)) return(NULL)
  if (identical(walnuts, TRUE)) walnuts <- list()
  if (!is.list(walnuts) || is.object(walnuts)) {
    stop("`walnuts` must be NULL, TRUE, or a named list of controls.", call. = FALSE)
  }
  allowed <- c("max_step_halvings", "max_error", "min_micro_steps",
               "energy_criterion", "target_macro_steps", "max_min_micro_steps",
               "max_leapfrog_steps")
  nms <- names(walnuts)
  if (length(walnuts) &&
      (is.null(nms) || anyNA(nms) || any(!nzchar(nms)))) {
    stop("`walnuts` controls must all be named.", call. = FALSE)
  }
  if (anyDuplicated(nms)) {
    stop("`walnuts` controls must not have duplicate names.", call. = FALSE)
  }
  unknown <- setdiff(nms, allowed)
  if (length(unknown)) {
    stop("Unknown `walnuts` control(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }
  counts <- intersect(nms, c("max_step_halvings", "min_micro_steps",
                             "max_min_micro_steps", "max_leapfrog_steps"))
  for (name in counts) {
    if (name == "max_leapfrog_steps" && is.null(walnuts[[name]])) next
    walnuts[[name]] <- check_count(
      walnuts[[name]], paste0("walnuts$", name),
      min = if (name == "max_step_halvings") 0L else 1L,
      max = if (name == "max_step_halvings") 63L else .Machine$integer.max
    )
  }
  for (name in intersect(nms, c("max_error", "target_macro_steps"))) {
    if (name == "target_macro_steps" && is.null(walnuts[[name]])) next
    value <- walnuts[[name]]
    if (length(value) != 1L || !is.numeric(value) ||
        !is.finite(value) || value <= 0) {
      stop("`walnuts$", name, "` must be a positive finite number.", call. = FALSE)
    }
    walnuts[[name]] <- as.double(value)
  }
  if ("energy_criterion" %in% nms) {
    criterion <- walnuts$energy_criterion
    choices <- c(max_min = "MaxMin", endpoint = "Endpoint")
    if (!is.character(criterion) || length(criterion) != 1L ||
        is.na(criterion) || !criterion %in% names(choices)) {
      stop('`walnuts$energy_criterion` must be "max_min" or "endpoint".',
           call. = FALSE)
    }
    walnuts$energy_criterion <- unname(choices[[criterion]])
  }

  # Only these effective defaults are needed to validate interactions.
  min_steps <- walnuts$min_micro_steps %||% 1L
  max_steps <- walnuts$max_min_micro_steps %||% 1024L
  halvings <- walnuts$max_step_halvings %||% 8L
  adapting <- !is.null(walnuts$target_macro_steps)
  if (adapting && max_steps < min_steps) {
    stop("`walnuts$max_min_micro_steps` must be >= `min_micro_steps` when adapting.",
         call. = FALSE)
  }
  largest <- if (adapting) max_steps else min_steps
  # u64 arithmetic requires largest * 2^halvings < 2^64. Comparing
  # against a power of two avoids rounding u64::MAX in R doubles.
  if (largest >= 2^(64 - halvings)) {
    stop("`walnuts` micro steps * 2^max_step_halvings exceeds the u64 range.",
         call. = FALSE)
  }
  # An empty list enables upstream defaults; serialize it as {}, not [].
  if (!length(walnuts)) return("{}")
  as.character(jsonlite::toJSON(walnuts, auto_unbox = TRUE, digits = NA,
                               null = "null"))
}
