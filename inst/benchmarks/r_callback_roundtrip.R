# Quantify the cost of the R-callback hot loop for the R-density sampler
# (issue #26). Two questions:
#
#  1. FFI floor: how expensive is one Rust->R->Rust round trip? Proxied by a
#     no-op R closure call (extendr's Function::call adds Robj marshaling on
#     top, same order of magnitude).
#
#  2. Redundant compute: passing SEPARATE fn/grad re-runs the R-side
#     preconditioning solve `Matrix::solve(Lt, y)` twice per leapfrog step;
#     a COMBINED fn_grad runs it once. This is the part that scales with the
#     model. We mimic SPDE sparsity with a 2D lattice GMRF precision matrix.

suppressMessages(library(Matrix))

# median wall time per call, in microseconds
time_us <- function(expr, reps = 200L) {
  expr <- substitute(expr)
  env <- parent.frame()
  t <- replicate(reps, {
    a <- Sys.time()
    eval(expr, env)
    as.numeric(Sys.time() - a, units = "secs")
  })
  median(t) * 1e6
}

# --- 1. FFI floor: no-op R closure call ----------------------------------
noop <- function(y) NULL
for (d in c(10L, 100L, 1000L)) {
  y <- rnorm(d)
  cat(sprintf("no-op R call, dim=%-5d : %7.2f us\n", d, time_us(noop(y), 2000L)))
}
cat("\n")

# --- 2. Redundant preconditioning solve at SPDE scale --------------------
# 2D lattice GMRF precision (5-point stencil): sparse, SPDE-like.
lattice_precision <- function(n) {                 # n x n grid -> n^2 params
  I <- .symDiagonal(n)
  D1 <- bandSparse(n, k = c(-1, 0, 1),
                   diag = list(rep(-1, n - 1), rep(2, n), rep(-1, n - 1)))
  Q <- kronecker(I, D1) + kronecker(D1, I)
  Q + Diagonal(n * n, 0.1)                          # ensure SPD
}

for (n in c(20L, 40L, 70L)) {                       # 400, 1600, 4900 params
  Q <- lattice_precision(n)
  chd <- Matrix::Cholesky(Q, super = TRUE, perm = TRUE)
  L  <- Matrix::tril(drop0(as(chd, "sparseMatrix")))
  Lt <- Matrix::t(L)
  d  <- nrow(Q)
  y  <- rnorm(d)
  one_solve <- time_us(Matrix::solve(Lt, y), 200L)
  cat(sprintf(
    "GMRF dim=%-5d nnz(Lt)=%-8d : 1 solve=%7.1f us  |  separate(2x)=%7.1f us  combined(1x)=%7.1f us  extra/leapfrog=%6.1f us\n",
    d, length(Lt@x), one_solve, 2 * one_solve, one_solve, one_solve))
}
