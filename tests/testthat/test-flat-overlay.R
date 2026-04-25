test_that("flat_overlay handles scalar", {
  out <- nutpieR:::flat_overlay(list(sigma = 1.5), c("sigma"))
  expect_equal(out, 1.5)
})

test_that("flat_overlay handles 1D array", {
  out <- nutpieR:::flat_overlay(
    list(theta = c(1, 2, 3)),
    c("theta.1", "theta.2", "theta.3")
  )
  expect_equal(out, c(1, 2, 3))
})

test_that("flat_overlay handles 2D matrix (column-major)", {
  # BridgeStan emits column-major: M.1.1, M.2.1, M.1.2, M.2.2
  m <- matrix(c(1, 2, 3, 4), 2, 2)
  out <- nutpieR:::flat_overlay(
    list(M = m),
    c("M.1.1", "M.2.1", "M.1.2", "M.2.2")
  )
  expect_equal(out, c(1, 2, 3, 4))
})

test_that("flat_overlay handles 3D array (last-index-major)", {
  # BridgeStan emits last-index-major, which is R's native storage order:
  # first index varies fastest, last slowest.
  a <- array(1:8, c(2, 2, 2))
  out <- nutpieR:::flat_overlay(
    list(A = a),
    c("A.1.1.1", "A.2.1.1", "A.1.2.1", "A.2.2.1",
      "A.1.1.2", "A.2.1.2", "A.1.2.2", "A.2.2.2")
  )
  expect_equal(out, as.numeric(1:8))
})

test_that("flat_overlay handles mixed params", {
  out <- nutpieR:::flat_overlay(
    list(sigma = 5, theta = c(1, 2, 3)),
    c("sigma", "theta.1", "theta.2", "theta.3")
  )
  expect_equal(out, c(5, 1, 2, 3))
})

test_that("flat_overlay errors on unknown parameter", {
  expect_error(
    nutpieR:::flat_overlay(list(nonexistent = 0), c("sigma")),
    "Unknown parameter"
  )
})

test_that("flat_overlay errors on missing name without defaults", {
  expect_error(
    nutpieR:::flat_overlay(list(sigma = 1), c("mu", "sigma")),
    "Missing required"
  )
})

test_that("flat_overlay fills gaps from defaults", {
  out <- nutpieR:::flat_overlay(
    list(sigma = 7),
    c("mu", "sigma"),
    defaults = c(42, 99)
  )
  expect_equal(out, c(42, 7))
})

test_that("flat_overlay errors on shape mismatch", {
  expect_error(
    nutpieR:::flat_overlay(
      list(theta = c(1, 2)),
      c("theta.1", "theta.2", "theta.3")
    ),
    "2 values but expected 3"
  )
})

test_that("flat_overlay errors on defaults length mismatch", {
  expect_error(
    nutpieR:::flat_overlay(list(), c("a", "b"), defaults = c(1, 2, 3)),
    "defaults length"
  )
})

test_that("flat_overlay handles empty input", {
  expect_equal(nutpieR:::flat_overlay(list(), character()), numeric(0))
})
