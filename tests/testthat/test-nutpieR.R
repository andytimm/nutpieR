test_that("sample_normal returns correct dimensions", {
  draws <- sample_normal(100L, 2L, 42L)
  expect_true(is.matrix(draws))
  expect_equal(nrow(draws), 200)
  expect_equal(ncol(draws), 10)
})

test_that("sample_normal means are near 0 and SDs near 1", {
  draws <- sample_normal(500L, 4L, 42L)
  col_means <- colMeans(draws)
  col_sds <- apply(draws, 2, sd)
  expect_true(all(abs(col_means) < 0.1))
  expect_true(all(abs(col_sds - 1) < 0.15))
})

test_that("resolve_data handles NULL", {
  expect_equal(nutpieR:::resolve_data(NULL), "")
})

test_that("resolve_data handles JSON string", {
  json <- '{"N": 10}'
  expect_equal(nutpieR:::resolve_data(json), json)
})

test_that("resolve_data handles list", {
  result <- nutpieR:::resolve_data(list(N = 10, y = c(0, 1)))
  parsed <- jsonlite::fromJSON(result)
  expect_equal(parsed$N, 10)
  expect_equal(parsed$y, c(0, 1))
})

test_that("resolve_data handles .json file", {
  tmp <- tempfile(fileext = ".json")
  on.exit(unlink(tmp))
  writeLines('{"N": 5}', tmp)
  result <- nutpieR:::resolve_data(tmp)
  expect_equal(jsonlite::fromJSON(result)$N, 5)
})

test_that("resolve_data rejects invalid input", {
  expect_error(nutpieR:::resolve_data(123))
})

test_that("nutpie_compile_model returns nutpie_model", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")
  expect_s3_class(test_models$bernoulli, "nutpie_model")
  expect_true(file.exists(test_models$bernoulli$lib_path))
})

test_that("nutpie_sample returns draws_array with correct dims", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 200, num_chains = 2, seed = 42, refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::ndraws(draws), 400)
  expect_equal(posterior::nchains(draws), 2)
  expect_equal(posterior::niterations(draws), 200)
  expect_true("theta" %in% posterior::variables(draws))
})

test_that("nutpie_sample bernoulli theta is reasonable", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 500, num_chains = 4, seed = 123, refresh = 0
  )

  summ <- posterior::summarize_draws(draws)
  theta_mean <- summ$mean[summ$variable == "theta"]
  # With 2/10 successes + Beta(1,1) prior, posterior is Beta(3,9), mean = 0.25
  expect_true(abs(theta_mean - 0.25) < 0.05)
})

test_that("nutpie_sample normal model param names correct", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 200, num_chains = 2, seed = 42, refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  vars <- posterior::variables(draws)
  expect_true("mu" %in% vars)
  expect_true("sigma" %in% vars)

  # y = 1..5, so posterior mean of mu should be near 3, sigma near 1.6
  # sigma posterior is wide with n=5, so use generous tolerance
  summ <- posterior::summarize_draws(draws)
  mu_mean <- summ$mean[summ$variable == "mu"]
  sigma_mean <- summ$mean[summ$variable == "sigma"]
  expect_true(abs(mu_mean - 3.0) < 0.5)
  expect_true(abs(sigma_mean - 1.6) < 1.0)
})

test_that("nutpie_sample accepts sampling parameters", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_warmup = 200, num_chains = 2,
    max_treedepth = 8, target_accept = 0.9, seed = 42,
    refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 100)
  expect_equal(posterior::nchains(draws), 2)
})

test_that("nutpie_diagnostics exposes the load-bearing fields", {
  # Narrow contract: only the fields nutpieR's own code, docs, and the
  # standard MCMC convergence-diagnostic vocabulary depend on. Schema
  # additions in nuts-rs pass silently; the generic shape invariants are
  # checked separately below.
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0
  )

  diag <- nutpie_diagnostics(draws)
  expect_type(diag, "list")

  must_have <- c("diverging", "depth", "logp", "energy", "n_steps",
                 "step_size_bar", "mean_tree_accept")
  for (field in must_have) {
    expect_true(field %in% names(diag), info = paste("missing field:", field))
  }

  expect_type(diag$diverging, "logical")
  expect_type(diag$depth, "integer")
  expect_type(diag$n_steps, "integer")
  expect_type(diag$logp, "double")
  expect_type(diag$energy, "double")

  expect_true(all(diag$depth > 0))
  expect_true(all(is.finite(diag$logp)))
  expect_true(all(diag$mean_tree_accept >= 0 & diag$mean_tree_accept <= 1))
})

test_that("schema-driven extraction preserves passthrough invariants", {
  # Whatever nuts-rs emits, the extractor must surface it with the right
  # shape. We don't pin the exact field set — that's a contract between
  # nuts-rs and its callers, not nutpieR. We only check generic invariants
  # over what came through, plus types for the small subset where we're
  # making a typing promise to users.
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0
  )
  diag <- nutpie_diagnostics(draws)

  # Length invariant: every surfaced field is one entry per draw, regardless
  # of whether it's a scalar vector or a list-of-vectors.
  n <- posterior::ndraws(draws)
  for (f in names(diag)) {
    expect_equal(length(diag[[f]]), n, info = paste("bad length:", f))
  }

  # Typing promise (matches the roxygen for nutpie_diagnostics): count
  # fields surface as integer-when-fits, float fields as double, flag
  # fields as logical. Only enforced for fields that are actually present.
  type_promise <- list(
    integer = c("depth", "n_steps", "chain", "draw", "index_in_trajectory"),
    double  = c("logp", "energy", "step_size", "step_size_bar",
                "mean_tree_accept"),
    logical = c("diverging", "tuning", "maxdepth_reached"),
    list    = c("unconstrained_draw", "gradient")
  )
  for (typ in names(type_promise)) {
    for (f in intersect(type_promise[[typ]], names(diag))) {
      expect_type(diag[[f]], typ)
    }
  }

  # Bernoulli has 1 unconstrained parameter, so per-draw vector fields
  # should be length 1.
  if ("unconstrained_draw" %in% names(diag)) {
    expect_equal(length(diag$unconstrained_draw[[1]]), 1)
  }
  if ("gradient" %in% names(diag)) {
    expect_equal(length(diag$gradient[[1]]), 1)
  }
})

test_that("logp is non-zero floating-point — guards silent default-fill", {
  # If extract_diagnostics ever silently fails to populate logp (renamed
  # column, dtype mismatch, etc.) it stays at the zero-init buffer and prints
  # as integers. This test catches that class of bug.
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0
  )
  lp <- nutpie_diagnostics(draws)$logp

  expect_type(lp, "double")
  expect_true(any(lp != 0))
  expect_true(any(abs(lp - round(lp)) > 1e-8))
})

test_that("nutpie_diagnostics errors on non-nutpie object", {
  expect_error(nutpie_diagnostics(1:10), "No diagnostics found")
})

test_that("num_warmup and num_draws attributes are set", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_warmup = 200, num_chains = 2, seed = 42,
    refresh = 0
  )

  expect_equal(attr(draws, "num_warmup"), 200L)
  expect_equal(attr(draws, "num_draws"), 100L)
})

test_that("num_warmup attribute is set even without save_warmup", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0
  )

  expect_equal(attr(draws, "num_warmup"), 400L)  # default
  expect_equal(attr(draws, "num_draws"), 50L)
})

# --- Item 1: Compile from Stan code string ---

test_that("nutpie_compile_model accepts code string", {
  skip_if(is.null(test_models$code_string), "Code string model not compiled")
  expect_s3_class(test_models$code_string, "nutpie_model")
  expect_true(file.exists(test_models$code_string$lib_path))
})

test_that("nutpie_compile_model rejects both stan_file and code", {
  stan_file <- test_path("test_models", "bernoulli.stan")
  skip_if_not(file.exists(stan_file), "Stan file not found")
  expect_error(
    nutpie_compile_model(stan_file = stan_file, code = "data {}"),
    "exactly one"
  )
})

test_that("nutpie_compile_model rejects neither stan_file nor code", {
  expect_error(nutpie_compile_model(), "exactly one")
})

# --- Item 4: Graceful error handling ---

test_that("sampling with bad data gives R error, not crash", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  expect_error(
    nutpie_sample(test_models$bernoulli, data = '{"bad": "json format"}',
                  num_draws = 10, num_chains = 1, seed = 42, refresh = 0)
  )
})

# --- Item 5: save_warmup ---

test_that("save_warmup returns warmup draws", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_warmup = 50, num_chains = 2, seed = 42,
    refresh = 0, save_warmup = TRUE
  )

  warmup <- nutpie_warmup_draws(draws)
  expect_s3_class(warmup, "draws_array")
  expect_equal(posterior::niterations(warmup), 50)
  expect_equal(posterior::nchains(warmup), 2)
  expect_true("theta" %in% posterior::variables(warmup))

  warmup_diag <- nutpie_warmup_diagnostics(draws)
  expect_type(warmup_diag, "list")
  expect_equal(length(warmup_diag$diverging), 100)  # 50 * 2 chains

  # Warmup diagnostics should use the same schema-driven extractor.
  expect_true("logp" %in% names(warmup_diag))
  expect_true("tuning" %in% names(warmup_diag))
  expect_type(warmup_diag$logp, "double")
  expect_true(any(warmup_diag$logp != 0))
  # During warmup, tuning should be TRUE for at least some draws.
  expect_true(any(warmup_diag$tuning))
})

test_that("nutpie_warmup_draws errors without save_warmup", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0
  )

  expect_error(nutpie_warmup_draws(draws), "save_warmup")
})

# --- Item 6: cores parameter ---

test_that("cores parameter works", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 4, cores = 2, seed = 42, refresh = 0
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::nchains(draws), 4)
})

# --- Item 7: store_divergences and store_mass_matrix ---

test_that("store_divergences exposes divergence detail when divergences occur", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  # Force divergences via a tiny max_treedepth + low target_accept on a model
  # that's easy to push into pathology. We don't always get divergences, so
  # the test is conditional on actually having some.
  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 200, num_chains = 4, seed = 42, refresh = 0,
    store_divergences = TRUE, max_treedepth = 1, target_accept = 0.5
  )
  diag <- nutpie_diagnostics(draws)
  skip_if_not(sum(diag$diverging) > 0, "no divergences in this run")

  expect_true("divergence_start" %in% names(diag))
  expect_true("divergence_end" %in% names(diag))
  expect_true("divergence_momentum" %in% names(diag))
  expect_true("divergence_start_gradient" %in% names(diag))
  expect_type(diag$divergence_start, "list")
  expect_equal(length(diag$divergence_start), length(diag$diverging))
  # Non-NULL entries align with diverging draws
  div_idx <- which(diag$diverging)
  expect_true(all(!vapply(diag$divergence_start[div_idx], is.null, logical(1))))
})

test_that("store_divergences = FALSE drops all-null divergence list columns", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0
  )
  diag <- nutpie_diagnostics(draws)
  expect_false("divergence_start" %in% names(diag))
  expect_false("mass_matrix_inv" %in% names(diag))
})

test_that("store_mass_matrix adds mass_matrix_inv field", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    store_mass_matrix = TRUE
  )

  diag <- nutpie_diagnostics(draws)
  expect_true("mass_matrix_inv" %in% names(diag))
  expect_type(diag$mass_matrix_inv, "list")
  expect_equal(length(diag$mass_matrix_inv), 50)
  # At least some entries should be non-NULL numeric vectors
  non_null <- Filter(Negate(is.null), diag$mass_matrix_inv)
  expect_true(length(non_null) > 0)
  expect_type(non_null[[1]], "double")
})

# --- Item 8: low-rank mass matrix ---

test_that("low_rank_modified_mass_matrix produces valid draws", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 100, num_chains = 2, seed = 42, refresh = 0,
    low_rank_modified_mass_matrix = TRUE
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 100)
  expect_equal(posterior::nchains(draws), 2)
  expect_true("theta" %in% posterior::variables(draws))

  # Should get similar posterior as standard mass matrix
  summ <- posterior::summarize_draws(draws)
  theta_mean <- summ$mean[summ$variable == "theta"]
  expect_true(abs(theta_mean - 0.25) < 0.1)
})

test_that("low_rank + store_mass_matrix surfaces mass matrix without errors", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  draws <- nutpie_sample(test_models$bernoulli,
    data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    low_rank_modified_mass_matrix = TRUE, store_mass_matrix = TRUE
  )
  diag <- nutpie_diagnostics(draws)
  expect_true("logp" %in% names(diag))
  expect_type(diag$logp, "double")
  expect_true(any(diag$logp != 0))
})

# --- Issue #4: scalar init_mean auto-expansion ---

test_that("scalar init_mean is auto-expanded to correct length", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  expect_warning(
    draws <- nutpie_sample(test_models$bernoulli,
      data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
      num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
      init_mean = 0
    ),
    "init_mean.*deprecated"
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 50)
  expect_true("theta" %in% posterior::variables(draws))
})

test_that("vector init_mean still works", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  # bernoulli has 1 unconstrained parameter
  expect_warning(
    draws <- nutpie_sample(test_models$bernoulli,
      data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
      num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
      init_mean = c(0.5)
    ),
    "init_mean.*deprecated"
  )

  expect_s3_class(draws, "draws_array")
  expect_equal(posterior::niterations(draws), 50)
})

test_that("wrong-length init_mean vector errors", {
  skip_if(is.null(test_models$bernoulli), "Bernoulli model not compiled")

  expect_error(
    suppressWarnings(nutpie_sample(test_models$bernoulli,
      data = list(N = 10, y = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 1)),
      num_draws = 10, num_chains = 1, seed = 42, refresh = 0,
      init_mean = c(0.1, 0.2, 0.3)
    ))
  )
})

# --- Issue #5: pars and include ---

test_that("pars whitelist keeps only selected parameters", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = "mu"
  )

  vars <- posterior::variables(draws)
  expect_equal(vars, "mu")
})

test_that("pars blacklist excludes selected parameters", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
    pars = "mu", include = FALSE
  )

  vars <- posterior::variables(draws)
  expect_false("mu" %in% vars)
  expect_true("sigma" %in% vars)
})

test_that("pars = NULL returns all parameters (default)", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_chains = 1, seed = 42, refresh = 0
  )

  vars <- posterior::variables(draws)
  expect_true("mu" %in% vars)
  expect_true("sigma" %in% vars)
})

test_that("pars errors on unknown parameter names", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
      pars = "nonexistent"
    ),
    "Unknown parameter"
  )
})

test_that("pars exclusion of all parameters errors", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  expect_error(
    nutpie_sample(test_models$normal,
      data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
      num_draws = 50, num_chains = 1, seed = 42, refresh = 0,
      pars = c("mu", "sigma"), include = FALSE
    ),
    "remove all variables"
  )
})

test_that("pars filters warmup draws too", {
  skip_if(is.null(test_models$normal), "Normal model not compiled")

  draws <- nutpie_sample(test_models$normal,
    data = list(N = 5, y = c(1.0, 2.0, 3.0, 4.0, 5.0)),
    num_draws = 50, num_warmup = 50, num_chains = 1, seed = 42,
    refresh = 0, save_warmup = TRUE, pars = "sigma"
  )

  expect_equal(posterior::variables(draws), "sigma")
  warmup <- nutpie_warmup_draws(draws)
  expect_equal(posterior::variables(warmup), "sigma")
})
