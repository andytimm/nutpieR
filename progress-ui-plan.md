# Progress UI finalization plan

Finalizes the `feat/progress-ui` design after design review (2026-06-10).
Core pattern: **teach once, then abbreviate** — one-time hint lines above the
bar carry explanation; persistent bar tokens stay terse. Every live claim is
backed by exact data (no approximate detectors); precise/actionable claims
that need per-draw data live in the end-of-run summary.

## Decisions (locked)

- Modes: `auto` / `cli` / `text` / `none`. **progressr mode is removed**
  (unreleased, weakest of the options; the R-callback architecture keeps the
  door open for a future `progress = function(snapshot)` escape hatch).
- **cli moves Suggests → Imports** (tidyverse-core, zero hard deps). All
  `requireNamespace("cli")` conditionals and non-cli fallback branches die.
  Routing then depends only on environment: `auto` → `cli` when
  `interactive()` and not knitr; otherwise `text`. `refresh <= 0` → none.
- All progress output (bar, hints, text-mode chain lines, end summary) goes
  to **stderr** via `message()`/cli — one stream, correct interleaving,
  `suppressMessages()` silences everything. Text-mode chain lines switch
  from `cat()` to `message()`. Real warnings stay `warning()`.
- Hints fire **once** per run, on the earliest evidence that predicts the
  post-warmup behavior they describe.

## Final hint language (numbers interpolated at fire time)

```
! div: divergent transitions detected — try increasing `target_accept` or
  reparameterizing.

ℹ grad/draw: ~210 gradient evaluations per draw (tree depth ≈ 8) — usually a
  sign of difficult posterior geometry. If that's unexpected for your model,
  it's worth a sanity check.

ℹ spread: chain progress is uneven (slowest 23%, fastest 78%) — often one
  chain adapted a smaller step size or is in a harder region of the
  posterior. Adding to status line.
```

Emit via `cli::cli_alert_warning()` (div) / `cli::cli_alert_info()` (grad,
spread) so the `!`/`ℹ` symbols and their ASCII fallbacks come free.
Pre-format interpolated numbers with `sprintf()` before handing to cli (avoid
`{}` interpolation surprises).

## Triggers

| Hint | Fires when | Modes |
|---|---|---|
| div | first **post-warmup** divergence observed | cli, text |
| grad/draw | late-warmup-baseline exact average ≥ `GRAD_HINT_THRESHOLD` | cli, text |
| spread | over **started** chains: `max% − min% ≥ SPREAD_TRIGGER_POINTS` and median chain ≥ `SPREAD_MEDIAN_FLOOR` | cli only |

Details:

- **Post-warmup div counting (everywhere).** `{div}` token (cli *and* text),
  the hint, and red styling all count only post-warmup divergences, computed
  per chain from `divergent_draws` indices `>= num_warmup`. This fixes the
  current inconsistency where the live bar shows warmup divergences but the
  end table (sample-phase diagnostics) shows 0. *Implementation check:*
  verify nuts-rs `divergent_draws` index base (0- vs 1-, warmup-inclusive)
  against a forced-divergence run before wiring the comparison.
- **grad/draw exact average, late-warmup baseline.** When a chain's
  `finished_draws` first passes `LATE_WARMUP_FRACTION × num_warmup`, snapshot
  its `(total_num_steps, finished_draws)` as a baseline. Pooled average =
  `Σ(total − base_total) / Σ(finished − base_finished)` over baselined
  chains — exact and unbiased, no cap detection, no length-biased polling.
  The hint is informational (ℹ), not advisory; `max_treedepth` advice lives
  only in the end summary where %-at-cap is exact. Hint text's tree depth =
  `round(log2(avg + 1))` (`infer_tree_depth()` exists).
- **Spread sticky.** Once triggered, the `{spread}` token renders for the
  rest of the run (no flicker; collapses honestly to `spread 98-100%` at the
  end). Trigger state lives in the callback closure; the formatter takes an
  `active` flag. Computing over `started` chains only avoids false-firing
  when `cores < num_chains` queues chains at 0%.

## Tokens

- Default cli `chain_format`: `"{div} | {grad} | {spread}"`.
  `{spread}` is empty until triggered (empty-segment cleanup handles the
  pipes — must become multi-pass, see bugs).
- `{div}`: post-warmup count; `cli::col_red` when > 0.
- `{grad}`: `X.X grad/draw`; switches to the baseline-adjusted average once
  baselines exist (so bar and hint never disagree); `▲` prefix + yellow while
  average ≥ `GRAD_HINT_THRESHOLD` (`^` fallback when
  `!cli::is_utf8_output()`; the styling regex already matches `[▲^]`).
- `{spread}`: `spread 23-78%` — ASCII hyphen (text-safe), percent not draw
  counts (no collision with `pb_current/pb_total`).
- `{spark}` (opt-in, never default): **gap-from-leader sparkline** — adopt
  the encoding from the uncommitted `format_chain_spread()` experiment in
  the working tree (flat `▁▁▁▁` when chains are together; tall glyph = how
  far a chain trails the leader; 2% deadzone, saturates at 20%). Rename
  function → `format_chain_spark()`, token → `{spark}`; the `{spread}` name
  belongs to the percent-range token.
- `{draws}` (min–max range), `{lag}`, `{step}`: keep as opt-in tokens;
  `format_chain_draw_range()` switches en dash → hyphen.

**Absorbing the working-tree experiment** (uncommitted `R/progress.R` edit):
keep `format_chain_spread()`'s glyph encoding as `{spark}`; **revert** the
always-on bar-format change (restore `pb_current/pb_total` and the
`extra = list(phase = ...)` shape) — conditional surfacing via the status
field is the design, not permanent bar real estate.

## Mode matrix

| | cli | text | none |
|---|---|---|---|
| live display | single bar, status tokens | 1 line/chain every `refresh` draws, via `message()` | — |
| start banner | bar itself | `Sampling N chains, M draws each (W warmup)` from R | — |
| div + grad hints | yes | yes (plain one-time `message()` lines) | — |
| spread hint/token | yes | no (per-chain lines *are* the spread display) | — |
| end-of-run summary | yes | yes | no |
| interrupt handling | yes | yes | yes (Rust poll loop continues) |

Shared hint/trigger state (warned flags, baselines, spread-active) factors
into a small constructor used by both the cli and text callbacks.

## End-of-run summary (cli + text, identical content)

- Existing per-chain table (chain, draws, grad/draw, tdepth, step, div) —
  now cli-styled unconditionally.
- New: exact **%-at-cap** from per-draw diagnostics
  (`n_steps >= 2^max_treedepth − 1`, or `depth >= max_treedepth` when the
  depth column exists). When ≥ `CAP_SUMMARY_THRESHOLD`, print:
  `14% of draws hit the max_treedepth cap — consider increasing
  `max_treedepth`.` (Optionally fold a `%cap` column into the table.)
- Divergence advice line stays as is.

## Tunable constants (single block at top of `R/progress.R`)

```r
LATE_WARMUP_FRACTION  <- 0.75  # baseline grad/draw stats from here on
GRAD_HINT_THRESHOLD   <- 128   # avg grad/draw ≈ tree depth 7; tune in dogfooding
SPREAD_TRIGGER_POINTS <- 15    # percentage-point spread to trigger
SPREAD_MEDIAN_FLOOR   <- 0.10  # median chain must be past 10%
CAP_SUMMARY_THRESHOLD <- 0.05  # %-at-cap advice line gate
```

## Bug fixes folded in (from review)

1. Empty-token pipe cleanup in `format_status_tokens()` is single-pass —
   wrap in `while (grepl(...))`.
2. `as_progress_num()`: `is.na(x)` errors under `if()` for length > 1 input
   — guard with `x[[1]]`.
3. Delete dead `print_progress_summary()` (superseded by
   `print_sampling_diagnostic_summary()`).
4. knitr guard: `should_use_cli_progress()` also requires
   `!isTRUE(getOption("knitr.in.progress"))`.
5. Hardcoded `▲` bypasses encoding fallback — gate on
   `cli::is_utf8_output()`.

## File-by-file

- **`DESCRIPTION`**: cli Suggests → Imports; drop progressr from Suggests.
- **`R/progress.R`**: constants block; hint rewording + `cli_alert_*`
  symbols; post-warmup div counting; baseline machinery + shared hint-state
  constructor; `{spread}`/`{spark}` tokens (absorb experiment per above);
  text callback → `message()` + banner + hints; delete
  `make_progressr_callback()`, `in_with_progress()`,
  `print_progress_summary()`, all `requireNamespace("cli")` branches; %-at-cap
  in `print_sampling_diagnostic_summary()`; bugs 1–5.
- **`R/sample.R`**: drop `"progressr"` from `progress` choices and its
  branch; `resolve_progress_mode()` loses progressr; cli/text branches share
  the hint-state constructor; update roxygen for `progress` and
  `chain_format` (new tokens, post-warmup `{div}` semantics).
- **`src/rust/src/lib.rs`**: remove the now-dead `text_log` path
  (R always passes a callback unless mode = none): the rprintln chain lines,
  banner, `last_reported`, `chain_announced`/`announce_finished`; drop the
  unused `refresh` parameter from `sample_stan`/`run_sampler`. Keep the
  200ms poll + interrupt check (needed in all modes, including none).
  Regenerate `R/extendr-wrappers.R` via `rextendr::document()`.
- **`tests/testthat/test-progress.R`**: remove progressr tests; update mode
  resolution (no progressr; knitr option case); new tests — post-warmup div
  counting, grad baseline math + one-shot hint, spread trigger conditions
  (started-only, median floor, stickiness), `{spark}` rendering, text-mode
  `expect_message()` for chain lines + hints, %-at-cap summary with
  synthetic diagnostics, triple-empty-token pipe cleanup.
- **`NEWS.md`**: user-facing entry (modes, hints, tokens, cli Import).

## Implementation order (each step keeps tests green)

1. Surface shrink: remove progressr; cli → Imports; delete cli conditionals;
   knitr guard.
2. Rust: strip dead text path, drop `refresh` param, regenerate wrappers.
3. Stream unification: text mode → `message()`, R-side banner.
4. Post-warmup div semantics (token + hint + styling, cli and text) +
   reworded div hint.
5. grad/draw: baseline machinery, new ℹ hint, token accent switch,
   end-summary %-at-cap.
6. spread: percent-range token + sticky trigger + hint; `{spark}` rename of
   the experiment; default `chain_format` update; multi-pass cleanup.
7. Remaining small bugs, roxygen, NEWS.

## Verification

```r
NOT_CRAN=TRUE devtools::install(quick = TRUE)
devtools::test()
```

Plus eyeball runs: interactive cli (healthy model; forced-divergence model;
small `max_treedepth` model for the grad hint; `cores < num_chains` for the
spread false-positive guard), `Rscript` for text mode, and a knitr render.
