# Polish pass PR

Branch: `polish-pass` off `main`. Independent of the (re-scoped) test-slim PR; either order works.

Scope: small fixes / docs polish before talking to Paul Bürkner about brms backend. Excludes anything brms-shaped and excludes stanc `--O1` (separate workstream).

(Compile-cache PR #15 has landed since this plan was drafted; the `bridgestan_version` extendr wrapper added there is on the same `@noRd` list as the older shims — see Item 4.)

## Items

### 1. Validate `seed` before unsigned cast
**Bug.** `nutpie_sample(model, data, seed = NA)` doesn't error today — it samples with a garbage huge seed because `NA_integer_ = i32::MIN` sign-extends to ~1.8e19 when `seed as u64` runs in Rust.

- `R/sample.R:117-121` — replace the "fill if NULL" block with:
  - if `is.null(seed)`: `seed <- sample.int(.Machine$integer.max, 1L)` (same as today)
  - else: `seed <- check_count(seed, "seed", min = 0L)` plus `<= .Machine$integer.max` upper bound (extend `check_count` with optional `max` arg, or add a `check_seed` wrapper).
- `src/rust/src/lib.rs:382-407` — add `if seed < 0 { return Err(...) }` to the existing defensive guard block. (R-side validates; Rust mirrors for direct callers.)
- New unit tests: `expect_error(nutpie_sample(..., seed = NA))`, `seed = -1`, `seed = 1.5`, `seed = 2^32`.

### 2. `cores = NULL` fallback when `detectCores()` returns `NA`
- `R/sample.R:132-134` — wrap in `if (is.na(cores)) cores <- 1L` after the `min()` call. Rare but real on some Linux containers.

### 3. README accuracy fixes
- `README.md:110` — "the sampler defaults to 800 warmup draws (vs 400 for diagonal)" is **not implemented** (`sample.R:103` defaults `num_warmup = 400L` regardless of `low_rank_modified_mass_matrix`). Either:
  - (a) Implement the bump: in `R/sample.R`, default `num_warmup` to 800 when `low_rank_modified_mass_matrix = TRUE` (matches Python nutpie's default), or
  - (b) Drop the line from README.
  - **Recommend (a)** — Python nutpie does this for a reason; low-rank needs more warmup to stabilize. Use `missing(num_warmup)` to detect "user didn't pass it" so explicit values still win.
- `README.md:117` — replace "Apache Arrow (zero-copy)" with "Apache Arrow with a single copy into R-allocated memory". Honest framing of the 1.4.0 perf work.
- New "Advanced usage" section before `## How it works` — short blocks for `pars`/`include`, `init = list(...)`, `nutpie_param_names()`. Half a screen total.

### 4. `@keywords internal` → `@noRd` in `extendr-wrappers.R`
Internal `.Call` shims still generate `man/*.Rd` pages today (`bs_open.Rd`, `sample_normal.Rd`, `bridgestan_version.Rd` from the compile-cache PR, etc.) which clutter the docs index.

- `R/extendr-wrappers.R` — change every `@keywords internal` on the wrappers to `@noRd`. Includes the original shims (`sample_normal`, `compile_stan_model`, `sample_stan`, `bs_*`) and the newer `bridgestan_version`.
- Run `devtools::document()`; delete the now-unused `man/*.Rd` files for the internal helpers.
- Note: `rextendr` regenerates this file. Verify the `@noRd` survives the next regen — if not, add a wrapper-script comment or upstream a config flag. Quick test: run `rextendr::document()` after the change and diff.

### 6. `add_tbb_to_path()` in `Once`
- `src/rust/src/model.rs:12-44` — wrap the entire body in `static TBB_INIT: Once = Once::new(); TBB_INIT.call_once(|| { ... });`. Currently runs `read_dir(~/.bridgestan)` + a process-wide `set_var("PATH", ...)` on every `bs_open`. Cheap individually but unnecessary, and the `set_var` race is theoretically observable.
- Import: `use std::sync::Once;`.

### 7. `r_err` preserves error chain
- `src/rust/src/lib.rs:25-27` — change `Error::Other(e.to_string())` to `Error::Other(format!("{e:#}"))`. anyhow's `#` alternate format prints the full cause chain; users hitting BridgeStan compile failures or Stan model load errors will see the leaf cause they need.
- Note: this changes user-visible error strings. `test-nutpieR.R:294-297` ("sampling with bad data gives R error, not crash") only matches `expect_error(...)` without a regex, so safe. Grep the tests for hardcoded error text before merging.

### 9. Relocate `%||%`
- `R/diagnostics.R:110` — currently defined at the end of `diagnostics.R`, used in `sample.R:182`, `diagnostics.R:30, 36`. Works only because R loads files alphabetically.
- Move to `R/utils.R` (new file, just this helper for now). Add `#' @noRd` above it.

## Test additions

In whatever shape the test suite takes after the slim PR:
- `check_seed`: NA, negative, fractional, > integer.max → expect_error.
- `cores = NULL` when `detectCores()` returns `NA`: monkey-patch via `with_mocked_bindings` or `local_mocked_bindings` (testthat 3.1+) to force NA, expect success with cores = 1.
- (Optional) `r_err` format: a contrived bad-Stan-code compile that surfaces a chain; assert the message contains expected leaf text.

## Out of scope (deliberately deferred)

- **Item 8 (handle caching on `nutpie_model`)**: touches public-ish surface (the `nutpie_model` object's fields), worth its own PR with attention to GC interaction with `ExternalPtr` finalizers.
- **Item 10 (split `lib.rs`)**: 942-line file is fine for now. Revisit when next touching progress reporting or diagnostics extraction.
- **Test slim**: PR 1, separate.

## Sequencing notes

- Independent of the (re-scoped) test-slim PR; either order works. Compile-cache PR #15 is already in.
- Single PR. ~150 lines diff estimated. README + doc regen accounts for some of that.
- Run `devtools::check()` locally; expect zero new NOTEs/WARNINGs.
- Verify on Windows specifically: the `Once` change in `add_tbb_to_path` and the seed validation are the platform-sensitive bits.
