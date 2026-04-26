# Tests

Run the default tier with `devtools::test()`. Warm runs use the compile cache
seeded by the first run on a given machine.

## Optional gates

| Env var | Default | Effect |
|---|---|---|
| `NUTPIER_RUN_SLOW_TESTS` | unset | Set to `1` to run `test-posteriordb.R`, which downloads posteriordb fixtures and runs reference comparisons (~+70s). |
| `NUTPIER_DISABLE_COMPILE_CACHE` | unset | Set to `1` to bypass the on-disk compile cache and recompile from scratch on every `nutpie_compile_model()` call. Useful when isolating a cache-related bug. |

```sh
NUTPIER_RUN_SLOW_TESTS=1 Rscript -e 'devtools::test()'
NUTPIER_DISABLE_COMPILE_CACHE=1 Rscript -e 'devtools::test()'
```

## Compile-failure surfacing

`tests/testthat/helper-models.R` compiles the test models once and caches them
in an environment that the test files read. By default — when the session is
interactive and `CI` is unset — compile errors are swallowed and dependent
tests skip with a message. In **non-interactive** runs, or when `CI` is set,
errors propagate so a green run actually proves the models built. No env var
to remember; this is the right default for both local dev (without a Rust
toolchain) and CI.

## Cross-implementation comparison

`inst/scripts/compare-with-python.R` runs nutpieR and Python `nutpie` on the
same model + seed and prints the posterior-mean difference. Run it manually
when bumping `nuts-rs` or `nutpie` (Python) versions. Not part of the test
suite.
