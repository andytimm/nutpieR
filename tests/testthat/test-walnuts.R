test_that("WALNUTS is opt-in and preserves upstream defaults", {
  expect_null(resolve_walnuts(NULL))
  expect_identical(resolve_walnuts(TRUE), "{}")
  expect_identical(resolve_walnuts(list()), "{}")
  expect_error(resolve_walnuts(FALSE), "NULL, TRUE, or a named list")
  expect_error(resolve_walnuts(1), "NULL, TRUE, or a named list")
})

test_that("WALNUTS serializes only supplied controls", {
  controls <- list(max_step_halvings = 8, max_error = 0.2,
                   min_micro_steps = 2, energy_criterion = "endpoint",
                   target_macro_steps = 15, max_min_micro_steps = 128,
                   max_leapfrog_steps = 1000)
  out <- jsonlite::fromJSON(resolve_walnuts(controls))
  expect_identical(names(out), names(controls))
  controls$energy_criterion <- "Endpoint"
  expect_equal(out, controls)
  expect_identical(jsonlite::fromJSON(resolve_walnuts(
    list(energy_criterion = "max_min")))$energy_criterion, "MaxMin")
  expect_identical(jsonlite::fromJSON(resolve_walnuts(
    list(max_error = 0.1))), list(max_error = 0.1))
  expect_null(jsonlite::fromJSON(resolve_walnuts(
    list(target_macro_steps = NULL)))$target_macro_steps)
})

test_that("WALNUTS rejects malformed controls", {
  expect_error(resolve_walnuts(list(1)), "all be named")
  expect_error(resolve_walnuts(setNames(list(1), NA_character_)), "all be named")
  expect_error(resolve_walnuts(list(max_error = 1, max_error = 2)), "duplicate")
  expect_error(resolve_walnuts(list(max_eror = 1)), "Unknown")
  for (bad in list(NULL, NA, Inf, NaN, 0, -1, TRUE, "1", c(1, 2))) {
    expect_error(resolve_walnuts(list(max_error = bad)), "positive finite")
  }
  for (bad in list(NA, Inf, NaN, 0, -1, TRUE, "1", c(1, 2))) {
    expect_error(resolve_walnuts(list(target_macro_steps = bad)), "positive finite")
  }
  for (bad in list(NULL, NA, "MaxMin", "max", c("max_min", "endpoint"))) {
    expect_error(resolve_walnuts(list(energy_criterion = bad)), "energy_criterion")
  }
})

test_that("WALNUTS checks count ranges and upstream overflow", {
  for (name in c("min_micro_steps", "max_min_micro_steps", "max_leapfrog_steps")) {
    for (bad in list(0, -1, 1.5, NA, Inf, TRUE, .Machine$integer.max + 1)) {
      expect_error(resolve_walnuts(setNames(list(bad), name)), "walnuts")
    }
    expect_no_error(resolve_walnuts(setNames(list(.Machine$integer.max), name)))
  }
  expect_no_error(resolve_walnuts(list(max_step_halvings = 0)))
  expect_no_error(resolve_walnuts(list(max_step_halvings = 63)))
  expect_error(resolve_walnuts(list(max_step_halvings = 64)), "<= 63")
  expect_error(resolve_walnuts(list(max_step_halvings = 63, min_micro_steps = 2)),
               "u64 range")
  expect_no_error(resolve_walnuts(list(max_step_halvings = 62, min_micro_steps = 3)))
  expect_error(resolve_walnuts(list(max_step_halvings = 62, min_micro_steps = 4)),
               "u64 range")
  expect_error(resolve_walnuts(list(target_macro_steps = 15, min_micro_steps = 1025)),
               "must be >=")
  expect_error(resolve_walnuts(list(target_macro_steps = 15, max_step_halvings = 54)),
               "u64 range")
  expect_no_error(resolve_walnuts(list(min_micro_steps = 1025)))
})
