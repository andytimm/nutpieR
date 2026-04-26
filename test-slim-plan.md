# Test slim PR

Branch: `slim-tests` off `main`.

**Status update (post compile-cache PR #15):** compile-cache reduced warm `devtools::test()` from 187s → ~25s and the posteriordb gate already landed (under `NUTPIER_RUN_SLOW_TESTS=1`, not the `NUTPIER_RUN_POSTERIORDB` originally proposed). What remains is a smaller-but-still-real runtime win plus failure-locality cleanup.

## Goals

1. **Trim another ~5s off the warm baseline.** ~16 sampler-integration tests in `test-nutpieR.R` and `test-init.R` exist to verify R-side wiring that a unit test covers faster (each sampler run is ~200-500ms; the cache PR didn't touch sampler runs). Realistic post-slim target: ~20s warm, ~20% on top of compile-cache.
2. **Surface compile failures in CI.** Today `helper-models.R` swallows errors and downstream tests skip silently — a green CI doesn't prove anything compiled.
3. **Better failure locality.** When a unit-level R helper regresses, an end-to-end sampler test points at the wrong layer. Direct tests of `resolve_*` helpers fail at the bug.
4. **Don't reduce coverage of the package-specific contract.** Keep the boundary/conversion tests (diagnostics typing, list-column dropping, `pars` flag resolution, init shape dispatch).
5. **Move `test-reticulate.R` to `inst/scripts/`.** It's a cross-implementation sanity check, not a regression.

Not goals: changing R or Rust code outside the small `resolve_constrain_flags` carve-out below.

## Already done by compile-cache PR #15

| Original item | Status |
|---|---|
| Gate `test-posteriordb.R` | ✅ landed, env var is `NUTPIER_RUN_SLOW_TESTS=1` |
| Cut Stan-compilation cost in default loop | ✅ replaced by the cache (helper-models warm-cache hits in <1s; cold seeds the cache once per dev box) |
| Add a slim test file for cache state-machine logic | ✅ `tests/testthat/test-compile-cache.R` (mocked + 2 real smokes) |

## File-by-file (remaining work)

### `helper-models.R` — modify

Current state: `try_compile` swallows all errors; `test_models$<name>` is `NULL` if compile fails; every test guards with `skip_if(is.null(...))`.

Change:
- Read `NUTPIER_REQUIRE_MODELS` env var. When `"1"`, `try_compile` no longer swallows — let the error propagate so the test file errors out hard. Default behavior unchanged for local devs without a Rust toolchain.
- Set `NUTPIER_REQUIRE_MODELS=1` in the GitHub Actions / R-universe CI workflow (separate one-line change).

```r
try_compile <- function(name, ...) {
  if (identical(Sys.getenv("NUTPIER_REQUIRE_MODELS"), "1")) {
    return(nutpie_compile_model(...))
  }
  tryCatch(nutpie_compile_model(...), error = function(e) NULL)
}
```

### `test-flat-overlay.R` — keep as-is

Already pure unit tests, fast, focused. No changes.

### `test-helpers.R` — keep as-is, mostly

Already mostly unit-style (opens BS handles, no sampling). Two micro-changes:
- The two `unconstrained = TRUE/FALSE` deprecation tests can collapse into one parameterized test. Optional; saves ~30 lines.

### `test-init.R` — consolidate

Current: 17 tests, ~10 of which call `nutpie_sample()` end-to-end just to assert "doesn't error".

Keep as integration (sampler runs):
- `init = list(mu = 0, sigma = 1)` full constrained → KEEP (one sampler run, the canonical happy path).
- `init = function(chain_id)` distinct starts → KEEP (already half-unit; the sampler half verifies the thread-local chain-id mechanism actually plumbs through).
- `partial init reproducible from sampler seed` → KEEP (this is the seed-stability contract, hard to test without two sampler runs).

Convert to direct `resolve_init` unit tests (no sampler):
- `init = list(sigma = 1)` partial fill → already covered by existing `partial named-list init fills missing params per chain`. **Drop the duplicate sampler run.**
- `init = list-of-lists` per-chain → covered by `resolve_init` shape inspection. Drop sampler run.
- `init = <JSON file path>` → write the JSON, call `resolve_init` directly, assert positions length. Drop sampler run.
- `init = 0` and `init = <positive scalar>` → call `resolve_init` directly, assert `length(positions)` and that all positions are zero (for `init = 0`). Drop two sampler runs.

Keep as fast unit tests (already are):
- unknown parameter, mutually-exclusive `init`/`init_mean`, wrong per-chain length, negative scalar errors, function-with-zero-args errors, function-returning-non-list errors, partial-init-survives-bad-GQ, fully-specified broadcasts.

**Net for `test-init.R`**: ~17 tests → ~14 tests, sampler runs 8 → 3.

### `test-nutpieR.R` — restructure

This is the bulk of the slim. Current ~40 tests, ~22 sampler runs. Target ~25 tests, ~6 sampler runs.

#### Drop entirely

- `sample_normal returns correct dimensions` (lines 1-6) — tests an internal Rust demo function, not user-facing surface.
- `sample_normal means are near 0 and SDs near 1` (lines 8-14) — tests nuts-rs distributional correctness, not nutpieR. Probabilistic tolerances also add CI noise.
- `nutpie_sample accepts sampling parameters` (lines 127-140) — redundant with the primary dimension test; doesn't assert anything specific to the parameters being passed.
- `cores parameter works` (lines 342-352) — only asserts "doesn't error and returns 4 chains", which is already covered.
- `pars = NULL returns all parameters (default)` (lines 547-558) — covered by the canonical sampling test.
- `pars = NULL keeps full constrained set on tp_gq model` (lines 700-713) — same.

#### Convert to unit tests (no sampling)

- `pars exclusion of all parameters errors` (lines 573-583) → call `resolve_keep_indices` directly with an exhaustive blacklist on a name vector.
- `pars errors on unknown parameter names` (lines 560-571) → call `resolve_constrain_flags` directly with a fake handle name list. (Actually these need a `handle`; either keep via the bs_open/no-sample pattern or factor `resolve_constrain_flags` to take name vectors instead of a handle. Latter is cleaner; consider as part of this PR.)
- `empty whitelist errors instead of silently keeping block params` (lines 731-745) → `resolve_keep_indices` direct call.
- `empty blacklist keeps all` (lines 747-759) → same.
- `nutpie_sample rejects malformed sampler counts` (lines 44-68) — already a unit-ish test against R-side `check_count`. Keep, but consider extracting `check_count` and testing it without going through `nutpie_sample` at all. Cuts ~50ms each.

#### Merge / consolidate

- `nutpie_sample returns draws_array with correct dims` + `nutpie_sample bernoulli theta is reasonable` → one test that does both shape + posterior-mean assertion in a single sampler run.
- `num_warmup and num_draws attributes are set` + `num_warmup attribute is set even without save_warmup` → one test asserting both default and explicit cases via a single run (use the canonical sampling run from above and add attribute assertions to it).
- `nutpie_warmup_draws errors without save_warmup` → fold into `nutpie_diagnostics errors on non-nutpie object` style block: error-path assertions don't need their own `nutpie_sample` call; reuse the draws from the save_warmup test.
- `pars whitelist` + `pars blacklist` + `pars filters warmup draws too` → one test on the normal model that asserts whitelist, blacklist, and warmup-filter behaviour from a single `save_warmup = TRUE, pars = "sigma"` run.
- The four `pars whitelist on {block,TP,GQ}` tests on `tp_gq` → one test that asserts `vars` for each of three calls. (These three calls are unavoidable — they exercise different `(include_tp, include_gq)` paths.)
- `store_divergences exposes divergence detail` already conditional on `sum(diag$diverging) > 0` — keep as-is.
- `store_unconstrained / store_gradient` + `store_mass_matrix` → one test with both flags `TRUE` plus `store_mass_matrix = TRUE`.
- `low_rank_modified_mass_matrix produces valid draws` + `low_rank + store_mass_matrix surfaces mass matrix` → one combined test.
- The three `init_mean` deprecation tests → one parameterized test over scalar / vector / wrong-length cases. Already mostly there.

#### Result

After slim, `test-nutpieR.R` runs roughly:

| Block | Sampler runs |
|---|---|
| Bernoulli end-to-end (dims + posterior + attributes) | 1 |
| Normal end-to-end + `pars` (whitelist + blacklist + warmup) | 1 |
| `tp_gq` `pars` flag-path coverage | 3 |
| `save_warmup` with diagnostics + warmup_draws | 1 (folded into Bernoulli if practical, else +1) |
| `store_*` flags surface columns | 1 |
| `store_divergences` detail (conditional skip) | 1 |
| Low-rank smoke + mass matrix | 1 |
| Bad-data error path | 1 |
| `init_mean` deprecation | 1 |

Plus all the resolve_* unit tests, schema-typing test, and structural error tests with no sampler runs.

Headcount: ~9–10 sampler runs in `test-nutpieR.R` (down from ~22).

### `test-posteriordb.R` — already gated

Compile-cache PR #15 added the gate. Env var is `NUTPIER_RUN_SLOW_TESTS=1`; CLAUDE.md documents it. Nothing to do here in this PR beyond updating any cross-references in `tests/README.md` (see below) to use the actual env var name.

### `test-reticulate.R` — move to `inst/scripts/`

Hardcoded local Python path (`test-reticulate.R:4`) and a stale comment about `num_tune = 300` while `nutpie_sample()` now defaults to 400 (`:40`). Not a regression test — it's a benchmark / cross-implementation sanity check.

Action: **move to `inst/scripts/compare-with-python.R`**, fix it up while we're at it:
- Read Python path from `RETICULATE_PYTHON` env var or `reticulate::py_discover_config()`; error clearly if neither resolves.
- Update the stale `num_tune` comment to match current default (400).
- Add a header comment: "Run manually when bumping `nuts-rs` or `nutpie` (Python) versions to confirm the wrappers still produce equivalent draws. Not part of the test suite."
- Reference it from `tests/README.md` so devs know where it went.

Why move rather than gate: the reticulate test isn't really a per-PR check — it's something you run once per `nuts-rs` bump. Keeping it under `tests/testthat/` (even gated) implies "this should pass" semantics that don't fit. `inst/scripts/` is the conventional R-package location for "useful but not test" scripts.

## Carve-out: `resolve_constrain_flags` taking names instead of a handle

Current signature: `resolve_constrain_flags(handle, pars, include)`.

To unit-test it without a BS handle, factor to:
```r
resolve_constrain_flags_impl <- function(block, block_tp, full, pars, include) { ... }
resolve_constrain_flags <- function(handle, pars, include) {
  resolve_constrain_flags_impl(
    block = block_prefixes(bs_block_names(handle)),
    block_tp = block_prefixes(bs_block_tp_names(handle)),
    full = block_prefixes(bs_full_names(handle)),
    pars, include
  )
}
```

Lets the new unit tests construct synthetic name vectors and exercise every flag-path branch in <1ms. Optional but makes the slim cleaner; recommend doing it as part of this PR since it's a 5-line refactor with material test-speed payoff.

## Discoverability

Compile-cache added a one-line note to `CLAUDE.md` for `NUTPIER_RUN_SLOW_TESTS`. This PR adds:
- **`tests/README.md`** (new): "Optional test gates" section listing every `NUTPIER_*` env var (`NUTPIER_RUN_SLOW_TESTS`, `NUTPIER_REQUIRE_MODELS`, `NUTPIER_DISABLE_COMPILE_CACHE`), what each runs/changes, and example invocations.
- **`README.md`** (or `CONTRIBUTING.md` if added later): brief "Testing" section linking to `tests/README.md`.

## Validation before merging

- [ ] `devtools::test()` warm wall time drops from ~25s (compile-cache baseline) to ~20s with the dropped sampler runs.
- [ ] `NUTPIER_REQUIRE_MODELS=1 devtools::test()` passes locally.
- [ ] `NUTPIER_RUN_SLOW_TESTS=1 devtools::test()` passes locally (one-off check; already covered by compile-cache).
- [ ] `devtools::check()` is clean.
- [ ] Diff coverage: skim the deleted tests and confirm each one is either redundant with a kept test or covered by a new unit test. Don't merge until that mapping is explicit.
- [ ] CI workflow updated to set `NUTPIER_REQUIRE_MODELS=1`.

## Out of scope

- Adding tests for the polish-pass items (seed validation, `cores` NA fallback, etc.) — those land in the polish PR against the slimmed structure.
- Restructuring `test-init.R` and `test-nutpieR.R` into different filenames. The existing organization is fine; the slim is mostly within-file.
- Snapshot tests (`testthat::expect_snapshot`) for the `print` methods — possibly worthwhile later, not this PR.

## Open questions for review

1. Whether to keep `sample_normal` exposed at all. It's exported in `extendr-wrappers.R` and tested as a smoke check, but it ships with the package binary. If we drop the tests, do we also drop the function? (Probably yes; it was a development-time scaffold, and the bernoulli test now serves the smoke role.)
2. Whether the `resolve_constrain_flags` carve-out belongs here or in the polish PR.
