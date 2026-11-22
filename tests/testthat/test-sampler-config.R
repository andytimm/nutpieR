test_that("effective settings preserve disabled options as JSON null", {
  cfg <- jsonlite::fromJSON(rename_sampler_config(
    '{"num_tune":200,"walnuts":null,"gradient_clipping":10000000000,"max_leapfrog_steps":null}'
  ))
  expect_null(cfg$walnuts)
  expect_null(cfg$max_leapfrog_steps)
  expect_equal(cfg$num_warmup, 200)
  expect_equal(cfg$gradient_clipping, 1e10)
})
