#' @keywords internal
#' @useDynLib nutpieR, .registration = TRUE
"_PACKAGE"

# Null-coalescing operator. Base R has `%||%` since 4.4, but we depend on
# R >= 4.2.
#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a
