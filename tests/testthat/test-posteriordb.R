skip_if_not_installed("posteriordb")
skip_on_cran()

# Helper: compile posteriordb model, sample, compare to reference
test_posteriordb_model <- function(posterior_name, tol_mean = 1.0, tol_sd = 1.0,
                                   num_draws = 1000, num_chains = 4) {
  pdb <- posteriordb::pdb_default()
  po <- posteriordb::posterior(posterior_name, pdb)

  # Write Stan code to tempfile
  stan_code <- posteriordb::stan_code(po)
  stan_file <- tempfile(fileext = ".stan")
  writeLines(stan_code, stan_file)
  on.exit(unlink(stan_file), add = TRUE)

  # Get data and reference draws
  data <- posteriordb::pdb_data(po)
  ref_draws <- posteriordb::reference_posterior_draws(po)
  ref_summ <- posterior::summarize_draws(ref_draws)

  # Compile and sample
  model <- nutpie_compile_model(stan_file)
  draws <- nutpie_sample(model,
    data = data,
    num_draws = num_draws,
    num_chains = num_chains,
    seed = 1234
  )

  expect_s3_class(draws, "draws_array")

  our_summ <- posterior::summarize_draws(draws)

  # Compare means and SDs for each variable present in reference
  shared_vars <- intersect(ref_summ$variable, our_summ$variable)
  expect_true(length(shared_vars) > 0,
    info = paste("No shared variables between reference and our draws for", posterior_name))

  for (v in shared_vars) {
    ref_mean <- ref_summ$mean[ref_summ$variable == v]
    our_mean <- our_summ$mean[our_summ$variable == v]
    ref_sd <- ref_summ$sd[ref_summ$variable == v]
    our_sd <- our_summ$sd[our_summ$variable == v]

    expect_true(abs(our_mean - ref_mean) < tol_mean,
      info = sprintf("%s: %s mean %.3f vs ref %.3f (tol %.1f)",
        posterior_name, v, our_mean, ref_mean, tol_mean))

    expect_true(abs(our_sd - ref_sd) < tol_sd,
      info = sprintf("%s: %s sd %.3f vs ref %.3f (tol %.1f)",
        posterior_name, v, our_sd, ref_sd, tol_sd))
  }

  draws
}

test_that("eight_schools centered model matches posteriordb reference", {
  test_posteriordb_model("eight_schools-eight_schools_centered",
    tol_mean = 1.5, tol_sd = 1.5)
})

test_that("eight_schools non-centered model matches posteriordb reference", {
  test_posteriordb_model("eight_schools-eight_schools_noncentered",
    tol_mean = 1.5, tol_sd = 1.5)
})
