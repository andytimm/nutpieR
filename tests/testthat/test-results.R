test_that("matrix_to_draws_array creates a valid posterior draws array", {
  flat <- matrix(
    seq_len(12),
    nrow = 4,
    dimnames = list(NULL, c("alpha", "beta.1", "beta.2"))
  )

  draws <- nutpieR:::matrix_to_draws_array(flat, n_draws = 2L, n_chains = 2L)

  expect_identical(class(draws), c("draws_array", "draws", "array"))
  expect_identical(dim(draws), c(2L, 2L, 3L))
  expect_identical(
    dimnames(draws),
    list(
      iteration = c("1", "2"),
      chain = c("1", "2"),
      variable = c("alpha", "beta[1]", "beta[2]")
    )
  )
  expect_equal(as.numeric(draws[, 1, "alpha"]), c(1, 2))
  expect_equal(as.numeric(draws[, 2, "alpha"]), c(3, 4))
  expect_no_error(posterior::summarise_draws(draws))
})
