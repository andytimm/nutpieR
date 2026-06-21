# Tests for nutpie_sample_r() — sampling an R-supplied log-density (issue #26).

# A correlated 2D Gaussian with known moments, used across several tests.
mvn_logp <- function(mu, Sigma) {
  P <- solve(Sigma)
  function(y) as.numeric(-0.5 * t(y - mu) %*% P %*% (y - mu))
}
mvn_grad <- function(mu, Sigma) {
  P <- solve(Sigma)
  function(y) as.numeric(-P %*% (y - mu))
}

test_that("recovers the moments of a known Gaussian", {
  fn <- mvn_logp(mvn_mu, mvn_sigma)
  gr <- mvn_grad(mvn_mu, mvn_sigma)
  fit <- nutpie_sample_r(fn, gr, ndim = 2, num_draws = 3000, num_warmup = 1000,
                         seed = 42)

  expect_s3_class(fit, "draws_array")
  expect_equal(dim(fit), c(3000L, 1L, 2L))

  s <- posterior::summarise_draws(fit, "mean", "sd", "rhat")
  expect_equal(s$mean, mvn_mu, tolerance = 0.1)
  expect_equal(s$sd, sqrt(diag(mvn_sigma)), tolerance = 0.1)
  expect_true(all(s$rhat < 1.05))

  emp_corr <- cor(as.matrix(posterior::as_draws_matrix(fit)))[1, 2]
  expect_equal(emp_corr, 0.8, tolerance = 0.1)
})

test_that("diagnostics object is well-formed", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 500, num_warmup = 500, seed = 1)
  d <- nutpie_diagnostics(fit)
  expect_s3_class(d, "nutpie_diagnostics")
  expect_true(all(c("diverging", "n_steps", "step_size") %in% names(d)))
  expect_length(d$diverging, 500L)
  expect_type(d$n_steps, "integer")
  expect_equal(d$draw, seq_len(500L))
})

test_that("diagnostics include energy, tree depth, and acceptance", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 400, num_warmup = 400, seed = 8)
  d <- nutpie_diagnostics(fit)
  expect_true(all(c("depth", "maxdepth_reached", "energy", "logp",
                    "mean_tree_accept") %in% names(d)))
  expect_true(all(is.finite(d$energy)))
  expect_true(all(is.finite(d$logp)))
  expect_true(all(d$depth >= 0L))
  # Mean acceptance should sit near the NUTS target (0.8 by default).
  expect_equal(mean(d$mean_tree_accept), 0.8, tolerance = 0.15)
  # E-BFMI is computable now that energy is present.
  expect_true(is.finite(ebfmi_per_chain(d)))
})

test_that("reported logp matches the supplied density at each draw", {
  fn <- mvn_logp(mvn_mu, mvn_sigma)
  fit <- nutpie_sample_r(fn, mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 200, num_warmup = 200, seed = 6)
  d <- nutpie_diagnostics(fit)
  dm <- as.matrix(posterior::as_draws_matrix(fit))
  expected <- unname(apply(dm, 1L, fn))
  expect_equal(d$logp, expected, tolerance = 1e-8, ignore_attr = TRUE)
})

test_that("nutpie_nuts_params is fully populated and aligned", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 300, num_warmup = 200, seed = 4)
  np <- nutpie_nuts_params(fit)
  expect_equal(nrow(np), 300L * 6L) # 6 NUTS params x draws, no recycling
  d <- nutpie_diagnostics(fit)
  expect_equal(np$Value[np$Parameter == "n_leapfrog__"], as.numeric(d$n_steps))
  expect_equal(np$Value[np$Parameter == "divergent__"], as.numeric(d$diverging))
  expect_equal(np$Value[np$Parameter == "energy__"], as.numeric(d$energy))
  expect_equal(np$Value[np$Parameter == "treedepth__"], as.numeric(d$depth))
})

test_that("nutpie_nuts_params NA-fills genuinely absent fields, no recycling", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 100, num_warmup = 100, seed = 4)
  # Simulate a backend that omits energy: the short vector must not recycle
  # into mislabeled columns (the bug or_na() guards against).
  diag <- attr(fit, "diagnostics")
  diag$energy <- NULL
  attr(fit, "diagnostics") <- diag
  np <- nutpie_nuts_params(fit)
  expect_equal(nrow(np), 100L * 6L)
  expect_true(all(is.na(np$Value[np$Parameter == "energy__"])))
  expect_equal(np$Value[np$Parameter == "divergent__"],
               as.numeric(attr(fit, "diagnostics")$diverging))
})

test_that("progress prints when enabled and is silent when off", {
  args <- list(mvn_logp(mvn_mu, mvn_sigma), mvn_grad(mvn_mu, mvn_sigma),
               ndim = 2, num_draws = 100, num_warmup = 100, seed = 2)
  expect_output(do.call(nutpie_sample_r, c(args, progress = TRUE)),
                "Sampling \\(R density\\)")
  expect_silent(do.call(nutpie_sample_r, c(args, progress = FALSE)))
})

test_that("seed makes runs reproducible", {
  args <- list(mvn_logp(mvn_mu, mvn_sigma), mvn_grad(mvn_mu, mvn_sigma),
               ndim = 2, num_draws = 200, num_warmup = 200, seed = 99)
  a <- do.call(nutpie_sample_r, args)
  b <- do.call(nutpie_sample_r, args)
  expect_identical(as.matrix(posterior::as_draws_matrix(a)),
                   as.matrix(posterior::as_draws_matrix(b)))
})

test_that("default variable names are y1..yd, ndim inferred from init", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         init = c(0, 0), num_draws = 100, num_warmup = 100,
                         seed = 1)
  expect_equal(posterior::variables(fit), c("y1", "y2"))
})

test_that("expand maps draws to named reported quantities", {
  ex <- function(y) c(a = y[1], b = y[2], total = y[1] + y[2])
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 300, num_warmup = 300,
                         seed = 7, expand = ex)
  expect_equal(posterior::variables(fit), c("a", "b", "total"))
  dm <- as.matrix(posterior::as_draws_matrix(fit))
  expect_equal(unname(dm[, "total"]), unname(dm[, "a"] + dm[, "b"]))
})

test_that("expand returning a varying-length vector errors clearly", {
  # Returns length 2 on draw 1 but length 1 afterwards — must not silently
  # recycle into the row.
  flaky <- local({
    n <- 0L
    function(y) {
      n <<- n + 1L
      if (n == 1L) c(a = y[1], b = y[2]) else 99
    }
  })
  expect_error(
    nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma), mvn_grad(mvn_mu, mvn_sigma),
                    ndim = 2, num_draws = 50, num_warmup = 50, seed = 1,
                    expand = flaky),
    "fixed-length"
  )
})

test_that("init may be a function of chain_id", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, init = function(chain) c(5, 5),
                         num_draws = 100, num_warmup = 100, seed = 3)
  expect_s3_class(fit, "draws_array")
})

test_that("save_warmup returns warmup draws and diagnostics", {
  fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                         mvn_grad(mvn_mu, mvn_sigma),
                         ndim = 2, num_draws = 200, num_warmup = 150,
                         seed = 5, save_warmup = TRUE)
  wd <- nutpie_warmup_draws(fit)
  expect_s3_class(wd, "draws_array")
  expect_equal(dim(wd), c(150L, 1L, 2L))
  expect_length(nutpie_warmup_diagnostics(fit)$diverging, 150L)
})

# --- input validation --------------------------------------------------------

test_that("invalid arguments are rejected", {
  fn <- mvn_logp(mvn_mu, mvn_sigma)
  gr <- mvn_grad(mvn_mu, mvn_sigma)
  expect_error(nutpie_sample_r(1, gr, ndim = 2), "`fn` must be a function")
  expect_error(nutpie_sample_r(fn, 1, ndim = 2), "`grad` must be a function")
  expect_error(nutpie_sample_r(fn, gr), "Supply either `ndim` or an `init`")
  expect_error(nutpie_sample_r(fn, gr, ndim = 2, init = c(0, 0, 0)),
               "`init` has length 3 but `ndim` is 2")
  expect_error(nutpie_sample_r(fn, gr, ndim = 2, expand = 1),
               "`expand` must be NULL or a function")
})

test_that("a non-finite init is reported clearly", {
  expect_error(
    nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma), mvn_grad(mvn_mu, mvn_sigma),
                    init = c(Inf, 0)),
    "non-finite"
  )
})

test_that("an error in the user's fn surfaces, not a divergence storm", {
  bad <- function(y) stop("boom")
  expect_error(
    nutpie_sample_r(bad, mvn_grad(mvn_mu, mvn_sigma), ndim = 2,
                    num_draws = 10, num_warmup = 10, seed = 1),
    "boom"
  )
})

# --- cross-check against the Stan path on the same density -------------------

test_that("R path matches the Stan path on the same Gaussian", {
  skip_if(is.null(test_models$mvn), "MVN model not compiled")
  data <- list(mu = mvn_mu, Sigma = mvn_sigma)

  stan_fit <- nutpie_sample(test_models$mvn, data = data, num_chains = 1,
                            num_draws = 3000, num_warmup = 1000, seed = 42)
  r_fit <- nutpie_sample_r(mvn_logp(mvn_mu, mvn_sigma),
                           mvn_grad(mvn_mu, mvn_sigma),
                           ndim = 2, num_draws = 3000, num_warmup = 1000,
                           seed = 42)

  stan_s <- posterior::summarise_draws(stan_fit, "mean", "sd")
  r_s <- posterior::summarise_draws(r_fit, "mean", "sd")

  # Same posterior moments to Monte Carlo error.
  expect_equal(r_s$mean, stan_s$mean, tolerance = 0.1)
  expect_equal(r_s$sd, stan_s$sd, tolerance = 0.1)

  # And comparable sampling efficiency (the diagonal-metric ESS on a 0.8
  # correlation is identical in both implementations to within noise).
  stan_steps <- mean(nutpie_diagnostics(stan_fit)$n_steps)
  r_steps <- mean(nutpie_diagnostics(r_fit)$n_steps)
  expect_equal(r_steps, stan_steps, tolerance = 0.5)
})

# --- RTMB preconditioning reprex (issue #26) --------------------------------
# Cole's exact use case: a preconditioned RTMB objective driven through nutpie.
# Heavy (compiles a TMB model), so guarded behind RTMB availability.

test_that("preconditioned RTMB model (8 schools) samples sanely", {
  skip_on_cran()
  skip_if_not_installed("RTMB")
  skip_if_not_installed("Matrix")

  dat <- list(y = c(28, 8, -3, 7, -1, 1, 18, 12),
              sigma = c(15, 10, 16, 11, 9, 11, 10, 18))
  pars <- list(eta = rep(1, 8), mu = 0, logtau = 1)
  f <- function(pars) {
    RTMB::getAll(dat, pars)
    theta <- mu + exp(logtau) * eta
    lp <- sum(RTMB::dnorm(eta, 0, 1, log = TRUE)) +
      sum(RTMB::dnorm(y, theta, sigma, log = TRUE)) + logtau
    -lp
  }

  obj <- RTMB::MakeADFun(func = f, parameters = pars, random = "eta",
                         silent = TRUE)
  opt <- with(obj, nlminb(par, fn, gr))
  Q <- RTMB::sdreport(obj, getJointPrecision = TRUE)$jointPrecision
  chd <- Matrix::Cholesky(Q, super = TRUE, perm = TRUE)
  L <- Matrix::tril(Matrix::drop0(as(chd, "sparseMatrix")))
  Lt <- Matrix::t(L)
  perm <- chd@perm + 1L
  iperm <- Matrix::invPerm(perm)

  obj2 <- RTMB::MakeADFun(func = obj$env$data, parameters = obj$env$parList(),
                          map = obj$env$map, random = NULL, silent = TRUE,
                          DLL = obj$env$DLL)

  fn_y <- function(y) -obj2$fn(as.numeric(Matrix::solve(Lt, y)[iperm]))
  gr_y <- function(y) {
    x <- as.numeric(Matrix::solve(Lt, y)[iperm])
    as.numeric(-Matrix::solve(L, as.numeric(obj2$gr(x))[perm]))
  }

  ndim <- length(obj2$par)
  fit <- nutpie_sample_r(fn_y, gr_y, ndim = ndim,
                         num_draws = 1000, num_warmup = 1000, seed = 123)

  expect_s3_class(fit, "draws_array")
  # Few/no divergences and R-hat ~1 is the bar for "the adaptation worked".
  expect_lt(sum(nutpie_diagnostics(fit)$diverging), 50L)
  expect_true(all(posterior::summarise_draws(fit, "rhat")$rhat < 1.1))
})
