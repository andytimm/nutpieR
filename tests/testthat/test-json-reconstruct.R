test_that("reconstruct_stan_json handles scalar", {
  out <- nutpieR:::reconstruct_stan_json(c(1.5), c("sigma"))
  expect_equal(out, list(sigma = 1.5))
})

test_that("reconstruct_stan_json handles 1D array", {
  out <- nutpieR:::reconstruct_stan_json(
    c(1, 2, 3), c("theta.1", "theta.2", "theta.3")
  )
  expect_equal(dim(out$theta), 3L)
  expect_equal(as.numeric(out$theta), c(1, 2, 3))
})

test_that("reconstruct_stan_json handles 2D matrix (column-major)", {
  # BridgeStan emits column-major: M[1,1], M[2,1], M[1,2], M[2,2]
  out <- nutpieR:::reconstruct_stan_json(
    c(1, 2, 3, 4),
    c("M.1.1", "M.2.1", "M.1.2", "M.2.2")
  )
  expect_equal(dim(out$M), c(2L, 2L))
  # Column-major fill: c(1,2,3,4) laid into dim c(2,2) gives
  # M[1,1]=1, M[2,1]=2, M[1,2]=3, M[2,2]=4 — exactly the input mapping.
  expect_equal(out$M[1, 1], 1)
  expect_equal(out$M[2, 1], 2)
  expect_equal(out$M[1, 2], 3)
  expect_equal(out$M[2, 2], 4)
})

test_that("reconstruct_stan_json handles mixed params", {
  out <- nutpieR:::reconstruct_stan_json(
    c(5, 1, 2, 3),
    c("sigma", "theta.1", "theta.2", "theta.3")
  )
  expect_equal(out$sigma, 5)
  expect_equal(as.numeric(out$theta), c(1, 2, 3))
})

test_that("reconstruct_stan_json rejects inconsistent ranks", {
  expect_error(
    nutpieR:::reconstruct_stan_json(
      c(1, 2, 3), c("a.1", "a.1.1", "a.2")
    ),
    "Inconsistent index rank"
  )
})

test_that("reconstruct_stan_json handles empty input", {
  expect_equal(nutpieR:::reconstruct_stan_json(numeric(), character()), list())
})
