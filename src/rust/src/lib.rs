#![allow(non_snake_case)]

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray,
    StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use extendr_api::error::Result;
use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, DiagNutsSettings, LowRankNutsSettings,
    ProgressCallback, Sampler, SamplerWaitResult, Settings,
};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

extern "C" {
    static mut R_interrupts_pending: std::os::raw::c_int;
    static mut R_interrupts_suspended: std::os::raw::c_int;
    // Pump R's event loop. Front-ends (RStudio, Positron) buffer console output
    // produced during a blocking native call and only repaint when R processes
    // events, so without this the progress callback's output is invisible until
    // sample_stan() returns. See `pump_r_events`.
    fn R_ProcessEvents();
    // Write to R's output stream (stdout) — C-level, no SEXP allocation. Used
    // for progress rendering that avoids R's GC (#36). We use stdout (not
    // stderr) so GUI consoles like Positron/RStudio render it on a neutral base
    // with our ANSI accents honored — raw stderr gets painted a flat salmon by
    // those front-ends, swallowing the bar's coloring.
    fn Rprintf(format: *const std::os::raw::c_char, ...);
}

/// Flush front-end consoles mid-sampling by pumping R's event loop.
///
/// Interrupts are suspended across the call: on Unix `R_ProcessEvents` longjmps
/// (via `onintr`) when an interrupt is pending, which would skip the Rust
/// `Sampler` teardown and corrupt the in-flight result. Suspending lets it
/// flush output without jumping; the pending flag is left set and handled at the
/// top of the poll loop, which returns a clean error there instead.
fn pump_r_events() {
    unsafe {
        let prev = R_interrupts_suspended;
        R_interrupts_suspended = 1;
        R_ProcessEvents();
        R_interrupts_suspended = prev;
    }
}

/// Read and clear a pending R interrupt. Returns `true` if one was pending; the
/// caller should then return a clean `Err` so extendr raises a normal R error
/// at the call site ("Sampling interrupted.") rather than longjmp'ing past the
/// in-flight `Sampler` teardown.
fn interrupt_pending() -> bool {
    unsafe {
        if R_interrupts_pending != 0 {
            R_interrupts_pending = 0;
            true
        } else {
            false
        }
    }
}

/// Write a message to R's stdout (Rprintf) — C-level, no SEXP allocation.
/// Used for progress rendering during sampling to avoid triggering R's GC
/// while `tbbmalloc_proxy` is active (#36). Color support is decided R-side via
/// `cli::num_ansi_colors()` and passed down, so there is no `isatty` probe here
/// (it would read false in the RStudio/Positron consoles, which are not TTYs).
fn r_print(msg: &str) {
    // Rprintf is a printf-style C function; use "%s" to avoid interpreting
    // any '%' characters in `msg` as format specifiers.
    let fmt = b"%s\0";
    let c_msg = std::ffi::CString::new(msg).unwrap_or_default();
    unsafe {
        Rprintf(fmt.as_ptr() as *const std::os::raw::c_char, c_msg.as_ptr());
    }
}

mod model;

/// Convert any Display error to an extendr Error. Uses anyhow's alternate
/// Display format (`{:#}`) to preserve cause chains; a no-op for plain
/// `Display` impls that don't override the alternate flag.
fn r_err(e: impl std::fmt::Display) -> Error {
    Error::Other(format!("{e:#}"))
}

/// Validate a seed argument and convert to `u32`. Mirrors the R-side
/// `check_count(seed, max = .Machine$integer.max)` so direct callers
/// can't bypass it.
fn check_seed(seed: i32) -> Result<u32> {
    if seed < 0 {
        return Err(Error::Other(format!("seed must be >= 0, got {}", seed)));
    }
    Ok(seed as u32)
}

/// Unwrap an extendr `Result` at the FFI boundary, throwing a clean R error
/// on `Err` instead of going through extendr's panic-based conversion (which
/// prints `thread '<unnamed>' panicked...` to stderr before R catches it).
///
/// SAFETY note: `throw_r_error` longjmps and skips Rust destructors. Callers
/// must only invoke this from the outer `#[extendr]` boundary, after any
/// owned Rust state in the caller has been dropped — i.e. by delegating to
/// an `_impl` helper (or an immediately-invoked closure) that returns
/// `Result<T>` and calling `or_throw` on the returned `Result`.
fn or_throw<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => throw_r_error(e.to_string()),
    }
}

/// Return the linked BridgeStan crate version, e.g. "2.7.0". Used by the
/// inline-code compile cache key so a BridgeStan version bump invalidates
/// cached entries automatically.
/// @noRd
#[extendr]
fn bridgestan_version() -> String {
    bridgestan::VERSION.to_string()
}

/// Compile a Stan model to a shared library using BridgeStan.
/// Downloads BridgeStan sources if needed (first call is slow).
/// @param stan_file Path to the .stan file.
/// @param stanc_args Character vector of extra arguments for stanc compiler.
/// @param compile_args Character vector of extra arguments for make.
/// @return Path to the compiled shared library.
/// @noRd
#[extendr]
fn compile_stan_model(stan_file: &str, stanc_args: Strings, compile_args: Strings) -> String {
    or_throw(compile_stan_model_impl(stan_file, stanc_args, compile_args))
}

fn compile_stan_model_impl(
    stan_file: &str,
    stanc_args: Strings,
    compile_args: Strings,
) -> Result<String> {
    // Mirror the bridgestan crate's own cache-presence guard so the message
    // only fires on the genuine first-compile path. ~150s silent pause for
    // a 235 MB fetch was the single most reported dogfood surprise.
    if let Some(home) = dirs::home_dir() {
        let cache = home
            .join(".bridgestan")
            .join(format!("bridgestan-{}", bridgestan::VERSION));
        if !cache.exists() {
            rprintln!("Downloading BridgeStan sources (one-time, ~235 MB)...");
        }
    }
    let bs_path = bridgestan::download_bridgestan_src().map_err(r_err)?;
    let stan_path = PathBuf::from(stan_file);

    let stanc_vec: Vec<String> = if stanc_args.is_empty() {
        Vec::new()
    } else {
        stanc_args.iter().map(|s| s.to_string()).collect()
    };
    let stanc_refs: Vec<&str> = stanc_vec.iter().map(String::as_str).collect();
    let compile_vec: Vec<String> = if compile_args.is_empty() {
        Vec::new()
    } else {
        compile_args.iter().map(|s| s.to_string()).collect()
    };
    let compile_refs: Vec<&str> = compile_vec.iter().map(String::as_str).collect();

    let lib_path = bridgestan::compile_model(&bs_path, &stan_path, &stanc_refs, &compile_refs)
        .map_err(r_err)?;
    Ok(lib_path.to_string_lossy().into_owned())
}

#[derive(Clone, Default)]
struct ChainState {
    chain: usize,
    finished_draws: usize,
    total_draws: usize,
    divergences: usize,
    tuning: bool,
    started: bool,
    latest_num_steps: usize,
    total_num_steps: usize,
    step_size: f64,
    runtime: Duration,
    divergent_draws: Vec<usize>,
}

/// ANSI color codes for progress output. Using direct escape sequences avoids
/// any dependency on R's cli package — these go through Rprintf (stdout). Colors
/// are only emitted when R reports color support (`cli::num_ansi_colors()`,
/// passed down via the style string).
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_YELLOW: &str = "\x1b[33m";

/// Spinner frames for the progress bar animation.
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Grad/draw threshold above which we accent the value (matches R-side
/// GRAD_HINT_THRESHOLD).
const GRAD_HINT_THRESHOLD: f64 = 128.0;

/// Persistent state across progress polls for the Rprintf path. Mirrors the
/// R-side `new_progress_hints()` environment: late-warmup baselines for
/// grad/draw, one-shot hint flags, and the spread latch.
#[derive(Default)]
struct ProgressHints {
    /// Per-chain baseline: (total_num_steps, finished_draws) at the moment the
    /// chain crossed LATE_WARMUP_FRACTION of warmup. Keyed by chain index.
    grad_baselines: Vec<Option<(usize, usize)>>,
    warned_div: bool,
    warned_grad: bool,
    warned_spread: bool,
    spread_active: bool,
}

/// Fraction of warmup that must complete before we anchor the grad/draw
/// baseline (matches R-side `LATE_WARMUP_FRACTION`).
const LATE_WARMUP_FRACTION: f64 = 0.75;
/// Percentage-point spread that triggers the spread token (matches
/// `SPREAD_TRIGGER_POINTS`).
const SPREAD_TRIGGER_POINTS: f64 = 15.0;
/// Median chain must be past this fraction before spread fires (matches
/// `SPREAD_MEDIAN_FLOOR`).
const SPREAD_MEDIAN_FLOOR: f64 = 0.10;

impl ProgressHints {
    fn new() -> Self {
        Self::default()
    }

    /// Ensure the baselines vec is long enough for the number of chains.
    fn ensure_capacity(&mut self, n: usize) {
        while self.grad_baselines.len() < n {
            self.grad_baselines.push(None);
        }
    }

    /// Record each started chain's `(total_num_steps, finished_draws)` the first
    /// time it passes `LATE_WARMUP_FRACTION` of warmup. Returns the pooled
    /// baseline-adjusted grad/draw if any baselined chain has produced
    /// post-baseline draws.
    fn update_and_pooled_grad(&mut self, state: &[ChainState], num_warmup: usize) -> Option<f64> {
        self.ensure_capacity(state.len());
        let threshold = LATE_WARMUP_FRACTION * num_warmup as f64;

        for (i, c) in state.iter().enumerate() {
            if !c.started {
                continue;
            }
            if (c.finished_draws as f64) < threshold {
                continue;
            }
            if self.grad_baselines[i].is_none() {
                self.grad_baselines[i] = Some((c.total_num_steps, c.finished_draws));
            }
        }

        // Pooled grad/draw over baselined chains
        let mut steps_delta: f64 = 0.0;
        let mut draws_delta: f64 = 0.0;
        for (i, c) in state.iter().enumerate() {
            if !c.started {
                continue;
            }
            if let Some((base_total, base_finished)) = self.grad_baselines.get(i).copied().flatten() {
                // saturating_sub: the counters are monotonic in practice, but a
                // stale/regressed snapshot must not underflow-panic the sampler.
                steps_delta += c.total_num_steps.saturating_sub(base_total) as f64;
                draws_delta += c.finished_draws.saturating_sub(base_finished) as f64;
            }
        }
        if draws_delta > 0.0 {
            Some(steps_delta / draws_delta)
        } else {
            None
        }
    }

    /// Per-chain baseline-adjusted grad/draw, mirroring R's
    /// `chain_grad_per_draw`: use the late-warmup baseline when the chain has one
    /// and has drawn past it, else fall back to the raw lifetime average. `None`
    /// before the chain has finished any draw.
    fn chain_grad(&self, i: usize, c: &ChainState) -> Option<f64> {
        if let Some((base_total, base_finished)) = self.grad_baselines.get(i).copied().flatten() {
            let draws_delta = c.finished_draws.saturating_sub(base_finished);
            if draws_delta > 0 {
                return Some(
                    c.total_num_steps.saturating_sub(base_total) as f64 / draws_delta as f64,
                );
            }
        }
        if c.finished_draws > 0 {
            Some(c.total_num_steps as f64 / c.finished_draws as f64)
        } else {
            None
        }
    }
}

/// Emit a one-shot hint line. In bar mode the live bar line has no trailing
/// newline, so prefix `\n` to break the hint onto its own line — the last bar
/// frame freezes above it (like cli's `cli_progress_output`) and the next render
/// redraws the bar below. In text mode every line already ends with `\n`, so no
/// prefix is needed. `body` must include its own trailing newline.
fn emit_hint(bar_mode: bool, body: &str) {
    if bar_mode {
        r_print(&format!("\n{}", body));
    } else {
        r_print(body);
    }
}

/// Emit the one-shot post-warmup divergence hint (mirrors R's `maybe_div_hint`).
/// Fires at most once per run; shared by the bar and per-chain text paths so the
/// wording can't drift between them.
fn maybe_emit_div_hint(hints: &mut ProgressHints, total_divs: usize, bar_mode: bool) {
    if !hints.warned_div && total_divs > 0 {
        hints.warned_div = true;
        emit_hint(bar_mode, "⚠ div: divergent transitions detected — these can bias your results; try increasing `target_accept` or reparameterizing.\n");
    }
}

/// Emit the one-shot grad/draw hint (mirrors R's `maybe_grad_hint`). `pooled` is
/// the late-warmup baseline-adjusted pooled average; the hint fires once it
/// reaches `GRAD_HINT_THRESHOLD`.
fn maybe_emit_grad_hint(hints: &mut ProgressHints, pooled: Option<f64>, bar_mode: bool) {
    if hints.warned_grad {
        return;
    }
    if let Some(avg) = pooled {
        if avg >= GRAD_HINT_THRESHOLD {
            hints.warned_grad = true;
            let depth = (avg + 1.0).log2().round() as i32;
            emit_hint(bar_mode, &format!("ℹ grad/draw: ~{} gradient evaluations per draw (tree depth ~{}) — sampling is taking long trajectories; often a sign of difficult geometry or incomplete adaptation, worth checking if unexpected.\n",
                avg.round() as i32, depth));
        }
    }
}

/// Format a one-line progress summary for Rprintf rendering (#36).
///
/// This replaces the R callback path when `tbbmalloc_proxy` is active on macOS.
/// By writing via `Rprintf` (C FFI, no SEXP allocation), we avoid triggering
/// R's GC during sampling — the root cause of the `__TBB_malloc_safer_msize`
/// segfault.
///
/// Renders the single-line cli-style bar (the `progress = "cli"` macOS path).
/// The returned string carries its own leading `\r` so the caller can overwrite
/// the current terminal line. `progress = "text"` is rendered separately, per
/// chain, by [`format_text_progress_lines`].
fn format_progress_line(
    state: &[ChainState],
    format: &str,
    use_color: bool,
    width: i32,
    frame: usize,
    hints: &mut ProgressHints,
    num_warmup: usize,
) -> String {
    if state.is_empty() {
        return String::new();
    }

    let started: Vec<&ChainState> = state.iter().filter(|c| c.started).collect();
    if started.is_empty() {
        return String::new();
    }

    let total_finished: usize = started.iter().map(|c| c.finished_draws).sum();
    let total_draws: usize = started.iter().map(|c| c.total_draws).sum();
    let total_divs: usize = started.iter().map(|c| c.divergences).sum();
    let all_tuning = started.iter().all(|c| c.tuning);
    let any_tuning = started.iter().any(|c| c.tuning);
    let phase = if all_tuning {
        "warmup"
    } else if any_tuning {
        "mixed"
    } else {
        "sample"
    };

    let pct = if total_draws > 0 {
        (total_finished as f64 / total_draws as f64 * 100.0).round() as u32
    } else {
        0
    };

    let max_runtime = started
        .iter()
        .map(|c| c.runtime)
        .max()
        .unwrap_or_default();
    let elapsed = max_runtime.as_secs_f64();

    // ETA
    let eta_str = if total_finished > 0 && total_draws > 0 && elapsed > 0.1 {
        let rate = total_finished as f64 / elapsed;
        let remaining = (total_draws - total_finished) as f64 / rate;
        format_time(remaining)
    } else {
        "?".to_string()
    };

    let total_steps: usize = started.iter().map(|c| c.total_num_steps).sum();

    // Late-warmup baseline-adjusted grad/draw: discards the high-leapfrog
    // early-warmup transient so the reported average reflects the tuned sampler.
    // Falls back to the raw average until baselines are established.
    let baseline_grad = hints.update_and_pooled_grad(state, num_warmup);
    let avg_grad = baseline_grad.unwrap_or_else(|| {
        if total_finished > 0 {
            total_steps as f64 / total_finished as f64
        } else {
            0.0
        }
    });

    // --- One-shot hints (fire at most once per run; shared with text path) ---
    maybe_emit_div_hint(hints, total_divs, true);
    maybe_emit_grad_hint(hints, baseline_grad, true);

    // --- Spread (percent-range across running chains) ---
    // Compute per-chain fractions for started, unfinished chains
    let running: Vec<&ChainState> = started.iter()
        .filter(|c| c.total_draws > 0 && c.finished_draws < c.total_draws)
        .copied()
        .collect();

    let chain_fracs: Vec<f64> = running.iter()
        .map(|c| c.finished_draws as f64 / c.total_draws as f64)
        .collect();

    // Latch spread on once the trigger fires
    if !hints.spread_active && chain_fracs.len() >= 2 {
        let spread_pp = (chain_fracs.iter().cloned().fold(0.0f64, f64::max)
            - chain_fracs.iter().cloned().fold(1.0f64, f64::min)) * 100.0;
        let mut sorted = chain_fracs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        if spread_pp >= SPREAD_TRIGGER_POINTS && median >= SPREAD_MEDIAN_FLOOR {
            hints.spread_active = true;
            if !hints.warned_spread {
                hints.warned_spread = true;
                emit_hint(true, &format!("ℹ spread: chain progress is uneven (slowest {}%, fastest {}%) — often one chain adapted a smaller step size or is in a harder region of the posterior. Adding to status line.\n",
                    (chain_fracs.iter().cloned().fold(1.0f64, f64::min) * 100.0).round() as i32,
                    (chain_fracs.iter().cloned().fold(0.0f64, f64::max) * 100.0).round() as i32));
            }
        }
    }

    let spread_part = if hints.spread_active && chain_fracs.len() >= 2 {
        Some(format!("spread {}-{}%",
            (chain_fracs.iter().cloned().fold(1.0f64, f64::min) * 100.0).round() as i32,
            (chain_fracs.iter().cloned().fold(0.0f64, f64::max) * 100.0).round() as i32))
    } else {
        None
    };

    // --- Status tokens. The set and renderers mirror R's `format_status_token`
    // (R/progress.R); the active subset is chosen by `format` (the cli
    // `chain_format`, default `"{div} | {grad} | {spread}"`). ---

    let div_part = if total_divs > 0 {
        // Divergence count is red, matching R's `cli::col_red` styling.
        if use_color {
            format!("{}⚠ div: {}{}", ANSI_RED, total_divs, ANSI_RESET)
        } else {
            format!("⚠ div: {}", total_divs)
        }
    } else {
        "div: 0".to_string()
    };

    let grad_part = if avg_grad > 0.0 {
        // One decimal, matching R's `format_gradient_status` (`%.1f grad/draw`).
        let label = format!("{:.1} grad/draw", avg_grad);
        if avg_grad >= GRAD_HINT_THRESHOLD && use_color {
            format!("{}▲ {}{}", ANSI_YELLOW, label, ANSI_RESET)
        } else if avg_grad >= GRAD_HINT_THRESHOLD {
            format!("▲ {}", label)
        } else {
            label
        }
    } else {
        "- grad/draw".to_string()
    };

    let spread_part = spread_part.unwrap_or_default();

    // {draws}: per-chain finished range over a per-chain total, e.g.
    // "12,925-13,100/21k" (R's `format_chain_draw_range`).
    let draws_part = {
        let min_fin = started.iter().map(|c| c.finished_draws).min().unwrap_or(0);
        let max_fin = started.iter().map(|c| c.finished_draws).max().unwrap_or(0);
        let per_chain_total = started.iter().map(|c| c.total_draws).max().unwrap_or(0);
        if min_fin == max_fin {
            format!("{}/{}", fmt_thousands(min_fin), fmt_compact(per_chain_total))
        } else {
            format!("{}-{}/{}", fmt_thousands(min_fin), fmt_thousands(max_fin), fmt_compact(per_chain_total))
        }
    };

    // {spark}: gap-from-leader glyph per running chain (R's `format_chain_spark`).
    let spark_part = if running.len() > 1 {
        let max_total = running.iter().map(|c| c.total_draws).max().unwrap_or(0);
        if max_total > 0 {
            let max_fin = running.iter().map(|c| c.finished_draws).max().unwrap_or(0);
            const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            running
                .iter()
                .map(|c| {
                    let gap = max_fin.saturating_sub(c.finished_draws);
                    let ratio = gap as f64 / max_total as f64;
                    let frac = ((ratio - 0.02) / (0.20 - 0.02)).clamp(0.0, 1.0);
                    GLYPHS[((frac * 7.0).round() as usize).min(7)]
                })
                .collect::<String>()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // {lag}: which running chains trail the median by >10% (R's `format_chain_lag`).
    let lag_part = if running.len() > 1 {
        let mut fins: Vec<f64> = running.iter().map(|c| c.finished_draws as f64).collect();
        fins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = if fins.len() % 2 == 0 {
            (fins[fins.len() / 2 - 1] + fins[fins.len() / 2]) / 2.0
        } else {
            fins[fins.len() / 2]
        };
        if med > 0.0 {
            let labels: Vec<String> = running
                .iter()
                .filter(|c| (c.finished_draws as f64) < med * 0.9)
                .map(|c| format!("c{}", c.chain))
                .collect();
            if labels.is_empty() {
                String::new()
            } else {
                format!("{} slow", labels.join(","))
            }
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // {step}: smallest finite step size, bare `%.3g` (R's `format_min_step`).
    let step_part = {
        let min_step = state
            .iter()
            .map(|c| c.step_size)
            .filter(|s| s.is_finite() && *s > 0.0)
            .fold(f64::INFINITY, f64::min);
        if min_step.is_finite() {
            fmt_3g(min_step)
        } else {
            String::new()
        }
    };

    // {tdepth}: depth inferred from the max latest step count (R's
    // `format_treedepth_status` / `infer_tree_depth`).
    let tdepth_part = {
        let max_latest = started.iter().map(|c| c.latest_num_steps).max().unwrap_or(0);
        if max_latest > 0 {
            format!("tdepth: {}", (max_latest as f64).log2().floor() as i32)
        } else {
            "tdepth: -".to_string()
        }
    };

    let status = render_status_tokens(
        format,
        &[
            ("div", &div_part),
            ("grad", &grad_part),
            ("spread", &spread_part),
            ("draws", &draws_part),
            ("spark", &spark_part),
            ("lag", &lag_part),
            ("step", &step_part),
            ("tdepth", &tdepth_part),
        ],
    );

    let bar_width = 24usize;
    let filled = if total_draws > 0 {
        (bar_width * total_finished / total_draws).min(bar_width)
    } else {
        0
    };
    let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
    let spinner = SPINNER[frame % SPINNER.len()];

    // Layout mirrors the R cli bar's format string (R/progress.R):
    // `{spin} {phase} {pct} |{bar}| {current}/{total} {eta} | {status}`.
    // No elapsed counter in the prefix (cli has none); current/total are the
    // aggregate finished/total across started chains. The status tail is dropped
    // when the chosen `chain_format` renders nothing.
    let tail = if status.is_empty() {
        String::new()
    } else {
        format!(" | {}", status)
    };
    let line = format!(
        "\r{spinner} {phase} {pct:>3}% |{bar}| {current}/{total} ETA: {eta}{tail}",
        spinner = spinner,
        phase = phase,
        pct = pct,
        bar = bar,
        current = total_finished,
        total = total_draws,
        eta = eta_str,
        tail = tail,
    );

    // Clip to the console width so the `\r`-redrawn bar can't wrap. Keep one
    // column of margin to dodge terminals that auto-wrap on the final cell. The
    // leading `\r` (one byte) is preserved; only the visible body is truncated.
    if width > 1 {
        format!("\r{}", truncate_display(&line[1..], (width as usize) - 1))
    } else {
        line
    }
}

/// Render `progress = "text"` as per-chain lines, matching R's text callback
/// (`make_text_progress_callback`). One line per chain that has advanced at
/// least `refresh` draws since it last printed (or has just finished), gated
/// against `last_printed`. Emits the shared one-shot div/grad hints (the spread
/// hint is cli-only — in text mode the per-chain lines are themselves the spread
/// display). Returns the lines to print, in chain order.
fn format_text_progress_lines(
    state: &[ChainState],
    format: &str,
    refresh: usize,
    last_printed: &mut Vec<usize>,
    hints: &mut ProgressHints,
    num_warmup: usize,
) -> Vec<String> {
    if state.is_empty() {
        return Vec::new();
    }

    // One-shot div/grad hints, from the same baseline the cli bar uses. Text
    // lines already end with "\n", so no bar-mode newline framing is needed.
    let total_divs: usize = state.iter().map(|c| c.divergences).sum();
    maybe_emit_div_hint(hints, total_divs, false);
    let pooled = hints.update_and_pooled_grad(state, num_warmup);
    maybe_emit_grad_hint(hints, pooled, false);

    let max_runtime = state.iter().map(|c| c.runtime).max().unwrap_or_default();
    let elapsed_str = format_elapsed(max_runtime.as_secs_f64());

    while last_printed.len() < state.len() {
        last_printed.push(0);
    }

    let mut lines = Vec::new();
    for (i, c) in state.iter().enumerate() {
        if !c.started {
            continue;
        }
        let finished = c.finished_draws;
        let total = c.total_draws;
        let since_last = finished.saturating_sub(last_printed[i]);
        let is_finished = total > 0 && finished >= total;
        // Refresh gate, mirroring R: print every `refresh` draws, plus one final
        // line when the chain crosses its total.
        if since_last < refresh && !(is_finished && last_printed[i] < total) {
            continue;
        }
        last_printed[i] = finished;

        // Phase-relative counts: warmup draws then sampling draws.
        let phase_total = if c.tuning {
            num_warmup
        } else {
            total.saturating_sub(num_warmup)
        };
        let phase_finished = if c.tuning {
            finished.min(phase_total)
        } else {
            finished.saturating_sub(num_warmup).min(phase_total)
        };
        let pct = if phase_total > 0 {
            (100.0 * phase_finished as f64 / phase_total as f64).round() as i64
        } else {
            0
        };
        let phase_str = if c.tuning { "warmup" } else { "sample" };

        let div_token = if c.divergences > 0 {
            format!("⚠ div: {}", c.divergences)
        } else {
            "div: 0".to_string()
        };
        let grad_token = match hints.chain_grad(i, c) {
            Some(g) if g >= GRAD_HINT_THRESHOLD => format!("▲ {:.1} grad/draw", g),
            Some(g) => format!("{:.1} grad/draw", g),
            None => "- grad/draw".to_string(),
        };
        let tdepth_token = if c.latest_num_steps > 0 {
            format!("tdepth: {}", (c.latest_num_steps as f64).log2().floor() as i32)
        } else {
            "tdepth: -".to_string()
        };

        // Substitute the text tokens, mirroring R's text callback. Default
        // chain_format: "[{elapsed}] c{chain} {phase} {pct}  {draws}/{total} | {div} | {grad}".
        // No pipe-collapsing here — text tokens always render (no optional ones).
        let line = format
            .replace("{elapsed}", &elapsed_str)
            .replace("{chain}", &c.chain.to_string())
            .replace("{phase}", phase_str)
            .replace("{pct}", &format!("{}%", pct))
            .replace("{draws}", &fmt_thousands(phase_finished))
            .replace("{total}", &fmt_thousands(phase_total))
            .replace("{div}", &div_token)
            .replace("{grad}", &grad_token)
            .replace("{tdepth}", &tdepth_token);
        lines.push(line);
    }
    lines
}

/// Format an integer with thousands separators ("1,000"), matching R's
/// `format_draw_count`.
fn fmt_thousands(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// Compact draw count ("21k" for >= 1000, else the integer), matching R's
/// `format_draw_count_compact`. Used for the `{draws}` token's denominator.
fn fmt_compact(n: usize) -> String {
    if n >= 1000 {
        format!("{}k", n / 1000)
    } else {
        n.to_string()
    }
}

/// Three-significant-figure float ("0.041", "1.36", "126"), a close analogue of
/// R's `%.3g` for the `{step}` token. Trailing zeros are trimmed. (Does not
/// switch to exponential for very small magnitudes — step sizes never get there.)
fn fmt_3g(x: f64) -> String {
    if !x.is_finite() {
        return "NA".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let exp = x.abs().log10().floor() as i32;
    let decimals = (2 - exp).max(0) as usize;
    let s = format!("{:.*}", decimals, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Substitute `{token}` placeholders in a cli `chain_format` string, then collapse
/// the empty `" | "` segments left by tokens that rendered "" (e.g. an inactive
/// `{spread}`). Mirrors R's `format_status_tokens`: split on the pipe separator,
/// drop blank segments, rejoin with " | ". Token values never contain a pipe.
fn render_status_tokens(format: &str, tokens: &[(&str, &str)]) -> String {
    let mut result = format.to_string();
    for (name, value) in tokens {
        result = result.replace(&format!("{{{}}}", name), value);
    }
    result
        .split('|')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Truncate `s` to at most `max_cols` *visible* columns, ANSI-aware: escape
/// sequences (`ESC [ ... letter`) are copied whole and counted as zero width,
/// and a `ANSI_RESET` is appended if anything was dropped so an open color can't
/// bleed past the cut. Without this the bar — which has no trailing newline and
/// is redrawn with `\r` — wraps and corrupts when it overflows the console width
/// (long custom `chain_format`, narrow panes). Glyphs are counted one column
/// each (true for every glyph the bar emits — block/braille/triangle/ASCII).
fn truncate_display(s: &str, max_cols: usize) -> String {
    let mut out = String::with_capacity(s.len());
    let mut visible = 0usize;
    let mut truncated = false;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Copy the whole CSI escape sequence; it occupies no columns.
            out.push(ch);
            while let Some(&c) = chars.peek() {
                out.push(c);
                chars.next();
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        if visible >= max_cols {
            truncated = true;
            break;
        }
        out.push(ch);
        visible += 1;
    }
    if truncated {
        out.push_str(ANSI_RESET);
    }
    out
}

/// Compact elapsed-time string matching R's `format_progress_time`: "<0.1s",
/// "%.1fs" under a minute, then "%dm%02ds" / "%dh%02dm".
fn format_elapsed(seconds: f64) -> String {
    let seconds = if !seconds.is_finite() || seconds < 0.0 {
        0.0
    } else {
        seconds
    };
    if seconds > 0.0 && seconds < 0.1 {
        return "<0.1s".to_string();
    }
    if seconds < 60.0 {
        return format!("{:.1}s", seconds);
    }
    let total = seconds as u64;
    let mins = total / 60;
    let secs = (seconds % 60.0).round() as u64;
    if mins < 60 {
        format!("{}m{:02}s", mins, secs)
    } else {
        let hours = mins / 60;
        let m = mins % 60;
        format!("{}h{:02}m", hours, m)
    }
}

/// Format seconds as a compact time string for the bar's ETA (e.g. "45s",
/// "1m23s").
fn format_time(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "?".to_string();
    }
    if seconds < 1.0 {
        return "<1s".to_string();
    }
    let secs = seconds.round() as u64;
    if secs < 60 {
        format!("{}s", secs)
    } else {
        let mins = secs / 60;
        let s = secs % 60;
        if mins < 60 {
            format!("{}m{:02}s", mins, s)
        } else {
            let h = mins / 60;
            let m = mins % 60;
            format!("{}h{:02}m", h, m)
        }
    }
}

/// Sample from a Stan model using nuts-rs NUTS sampler.
/// Run the sampler with progress reporting. Generic over Settings type.
fn run_sampler<S: Settings>(
    stan_model: model::StanModel,
    settings: S,
    num_cores: i32,
    save_warmup: bool,
    progress_cb: Option<Function>,
    rprintf_progress: &str,
    chain_format: &str,
    console_width: i32,
    num_warmup: usize,
) -> Result<Vec<ArrowTrace>> {
    let mut progress_cb = progress_cb;
    let use_callback = progress_cb.is_some();
    let use_rprintf = !rprintf_progress.is_empty();

    let progress_state: Arc<Mutex<Vec<ChainState>>> = Arc::new(Mutex::new(Vec::new()));
    let state_clone = progress_state.clone();

    // Install the nuts-rs progress callback when either the R callback or the
    // Rprintf path is active — both need per-poll snapshots from the sampler.
    // With `progress = "none"` neither is set, so we skip the snapshot
    // bookkeeping — but the wait loop below still runs to service interrupts.
    let callback = if use_callback || use_rprintf {
        Some(ProgressCallback {
            callback: Box::new(move |_elapsed: Duration, progress: Box<[ChainProgress]>| {
                let snapshot: Vec<ChainState> = progress
                    .iter()
                    .enumerate()
                    .map(|(i, c)| ChainState {
                        chain: i + 1,
                        finished_draws: c.finished_draws,
                        total_draws: c.total_draws,
                        divergences: c.divergences,
                        tuning: c.tuning,
                        started: c.started,
                        latest_num_steps: c.latest_num_steps,
                        total_num_steps: c.total_num_steps,
                        step_size: c.step_size,
                        runtime: c.runtime,
                        divergent_draws: c.divergent_draws.clone(),
                    })
                    .collect();
                *state_clone.lock().unwrap() = snapshot;
            }),
            rate: Duration::from_millis(100),
        })
    } else {
        None
    };

    let mut arrow_config = ArrowConfig::default();
    arrow_config.store_warmup = save_warmup;
    let mut sampler_opt = Some(
        Sampler::new(
            stan_model,
            settings,
            arrow_config,
            num_cores as usize,
            callback,
        )
        .map_err(r_err)?,
    );

    // Parse the Rust-render style. The R side encodes everything it knows into
    // this one string (so the FFI signature stays put):
    //   "bar"        / "bar:color"
    //   "text:<n>"   / "text:<n>:color"
    // where <n> is the `refresh` draw count and the ":color" suffix is present
    // when R reports ANSI color support (`cli::num_ansi_colors()`). Empty means
    // the R callback path (non-macOS) or no progress.
    let mut parts = rprintf_progress.split(':');
    let kind = parts.next().unwrap_or("");
    let style_is_bar = kind == "bar";
    let style_is_text = kind == "text";
    let text_refresh: usize = if style_is_text {
        parts.next().and_then(|s| s.parse().ok()).unwrap_or(100).max(1)
    } else {
        100
    };

    // Throttle the cli bar to ~1 Hz. The poll loop still spins at 200 ms for
    // interrupt responsiveness and console flushing; the bar just re-renders once
    // per second. (Text mode is gated per-chain by `text_refresh` instead.)
    let cb_interval = Duration::from_millis(
        std::env::var("NUTPIE_CB_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000),
    );
    let mut last_render: Option<Instant> = None;
    let mut last_line = String::new();
    let mut frame: usize = 0;
    // Per-chain last-printed draw counts for the text path's refresh gate.
    let mut last_printed: Vec<usize> = Vec::new();
    let use_color = rprintf_progress.ends_with(":color");
    let mut hints = ProgressHints::new();

    let results = loop {
        let sampler = sampler_opt.take().unwrap();
        let wait_dur = Duration::from_millis(200);
        match sampler.wait_timeout(wait_dur) {
            SamplerWaitResult::Trace(traces) => break traces,
            SamplerWaitResult::Timeout(s) => {
                sampler_opt = Some(s);
                // Check interrupt BEFORE calling back into R, so extendr raises a
                // clean R error at the call site ("Sampling interrupted.") rather
                // than longjmp'ing past the assignment and leaving the user with a
                // confusing "object not found" on the next line.
                if interrupt_pending() {
                    return Err(Error::Other("Sampling interrupted.".into()));
                }
                // Repaint the front-end console each poll so progress streams
                // live instead of appearing all at once when sampling ends
                // (GitHub #34: RStudio buffers native-call output).
                pump_r_events();
                // pump_r_events() suspends interrupts, so R_ProcessEvents() can
                // latch a pending interrupt without longjmp'ing. Re-check before
                // the R progress callback runs, or it could service the flag by
                // longjmp'ing across these Rust frames.
                if interrupt_pending() {
                    return Err(Error::Other("Sampling interrupted.".into()));
                }
                let state_snapshot: Vec<ChainState> = {
                    let state = progress_state.lock().unwrap();
                    state.clone()
                };
                if state_snapshot.is_empty() {
                    continue;
                }

                // Three mutually exclusive render paths. The macOS Rust paths
                // (bar/text) write via Rprintf (stdout) and never touch the R
                // callback; the R callback path is the non-macOS default.
                if style_is_bar {
                    // cli bar: one overwriting line, throttled to ~1 Hz. The `\r`
                    // overwrite hides duplicates in a real terminal, but the
                    // last_line guard avoids noise in piped/captured output.
                    let due = last_render.map_or(true, |t| t.elapsed() >= cb_interval);
                    if due {
                        let line = format_progress_line(
                            &state_snapshot,
                            chain_format,
                            use_color,
                            console_width,
                            frame,
                            &mut hints,
                            num_warmup,
                        );
                        frame = frame.wrapping_add(1);
                        if !line.is_empty() && line != last_line {
                            r_print(&line);
                            pump_r_events();
                            last_line = line;
                        }
                        last_render = Some(Instant::now());
                    }
                } else if style_is_text {
                    // Per-chain text lines, gated by the refresh draw count. No
                    // time throttle — the refresh gate decides when each chain
                    // prints, matching R's text callback.
                    let lines = format_text_progress_lines(
                        &state_snapshot,
                        chain_format,
                        text_refresh,
                        &mut last_printed,
                        &mut hints,
                        num_warmup,
                    );
                    if !lines.is_empty() {
                        for line in &lines {
                            r_print(&format!("{}\n", line));
                        }
                        pump_r_events();
                    }
                } else if let Some(ref cb) = progress_cb {
                    // Non-macOS: hand the snapshot to the R progress callback every
                    // poll (cli rate-limits its own redraws). On failure, disable
                    // the callback for the rest of the run rather than aborting.
                    let snapshot_list = build_progress_snapshot(&state_snapshot);
                    let pairlist = Pairlist::from_pairs([("", Robj::from(snapshot_list))]);
                    match cb.call(pairlist) {
                        Ok(_) => {}
                        Err(e) => {
                            rprintln!(
                                "nutpieR: progress callback failed ({}); disabling further callbacks for this run.",
                                e
                            );
                            progress_cb = None;
                        }
                    }
                }
            }
            SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
        }
    };

    // Final progress render / callback.
    if use_rprintf {
        // Close off the bar's in-place line with a newline so the R end-of-run
        // summary starts fresh. Text mode already ends each line with "\n".
        if style_is_bar {
            r_print("\n");
        }
    } else if let Some(ref cb) = progress_cb {
        let state_snapshot: Vec<ChainState> = {
            let state = progress_state.lock().unwrap();
            state.clone()
        };
        if !state_snapshot.is_empty() {
            let snapshot_list = build_progress_snapshot(&state_snapshot);
            let pairlist = Pairlist::from_pairs([("", Robj::from(snapshot_list))]);
            let _ = cb.call(pairlist);
        }
    }

    Ok(results)
}

/// Build a `List` of per-chain named lists for the R progress callback.
/// Field set must match the R-side progress helpers in `R/progress.R`.
fn build_progress_snapshot(state: &[ChainState]) -> List {
    let entries: Vec<Robj> = state
        .iter()
        .map(|c| {
            list!(
                chain = c.chain as i32,
                finished_draws = c.finished_draws as i32,
                total_draws = c.total_draws as i32,
                divergences = c.divergences as i32,
                tuning = c.tuning,
                started = c.started,
                latest_num_steps = c.latest_num_steps as i32,
                total_num_steps = c.total_num_steps as f64,
                step_size = c.step_size,
                runtime = c.runtime.as_secs_f64(),
                divergent_draws = c
                    .divergent_draws
                    .iter()
                    .map(|x| *x as i32)
                    .collect::<Vec<i32>>()
            )
            .into_robj()
        })
        .collect();
    List::from_values(entries)
}

/// @param handle An `ExternalPtr<BSHandle>` from `bs_open()`.
/// @param num_draws Number of draws per chain after warmup.
/// @param num_warmup Number of warmup (tuning) draws per chain.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @param init_positions Optional list of numeric vectors (one per chain, or length 1 = broadcast).
/// @param jitter If TRUE, apply ±0.5 uniform jitter per coordinate.
/// @param save_warmup Whether to return warmup draws.
/// @param num_cores Number of CPU cores to use for parallel sampling.
/// @param store_divergences Whether to store detailed divergence information.
/// @param store_mass_matrix Whether to store the mass matrix at each draw.
/// @param store_unconstrained Whether to store the unconstrained position at each draw.
/// @param store_gradient Whether to store the gradient at each draw.
/// @param adaptation One of "diag" or "low_rank". The R wrapper accepts
///   "low-rank" as a Python-style alias and normalises it before calling.
/// @param max_treedepth Optional maximum tree depth for NUTS. NULL keeps the
///   nuts-rs default.
/// @param mindepth Optional minimum tree depth for NUTS.
/// @param target_accept Optional target acceptance probability.
/// @param max_energy_error Optional energy-error divergence threshold.
/// @param extra_doublings Optional number of extra tree doublings after a
///   turning point is reached.
/// @param mass_matrix_gamma Optional regularisation parameter for low-rank
///   mass matrix.
/// @param eigval_cutoff Optional eigenvalue cutoff for low-rank mass matrix.
/// @param keep_indices Optional 0-indexed integer vector of constrained
///   parameter columns to materialize. NULL means keep all. Indices are
///   resolved against the post-flag column layout selected by
///   `include_tp` / `include_gq`.
/// @param include_tp Whether bridgestan should compute transformed parameters
///   when expanding each draw. When the caller has filtered them out via
///   `pars`/`include`, set this to `FALSE` to skip the per-draw allocation
///   and Stan-side work.
/// @param include_gq Whether bridgestan should compute generated quantities
///   when expanding each draw. Setting this `FALSE` skips the GQ block
///   (including any `*_rng` calls) entirely. Must imply `include_tp = TRUE`
///   when `TRUE`, since GQ may reference TP.
/// @param progress_callback NULL or an R closure invoked on each poll wakeup
///   with one argument: a list of `num_chains` per-chain snapshots
///   (`chain`, `finished_draws`, `total_draws`, `divergences`, `tuning`,
///   `started`, `latest_num_steps`, `total_num_steps`, `step_size`, `runtime`,
///   `divergent_draws`). When supplied, the built-in per-chain text log is
///   suppressed; errors raised by the closure are warned once and further calls
///   suppressed for the run.
/// @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
///   sampler_config (JSON), and optionally warmup_draws and warmup_diagnostics.
/// @noRd
#[allow(clippy::too_many_arguments)]
#[extendr]
fn sample_stan(
    handle: ExternalPtr<model::BSHandle>,
    num_draws: i32,
    num_warmup: i32,
    num_chains: i32,
    seed: i32,
    init_positions: Robj,
    jitter: bool,
    save_warmup: bool,
    num_cores: i32,
    store_divergences: bool,
    store_mass_matrix: bool,
    store_unconstrained: bool,
    store_gradient: bool,
    adaptation: &str,
    max_treedepth: Robj,
    mindepth: Robj,
    target_accept: Robj,
    max_energy_error: Robj,
    extra_doublings: Robj,
    mass_matrix_gamma: Robj,
    eigval_cutoff: Robj,
    keep_indices: Robj,
    include_tp: bool,
    include_gq: bool,
    progress_callback: Robj,
    rprintf_progress: &str,
    chain_format: &str,
    console_width: i32,
) -> List {
    or_throw((|| -> Result<List> {
        // Defensive guards before unsigned casts. The R wrapper validates these
        // already; we re-check here so direct callers (or malformed inputs that
        // somehow slip past the R layer) can't turn negative ints into huge
        // u64/usize allocations.
        for (val, name) in [
            (num_draws, "num_draws"),
            (num_chains, "num_chains"),
            (num_cores, "num_cores"),
        ] {
            if val <= 0 {
                return Err(Error::Other(format!("{} must be >= 1, got {}", name, val)));
            }
        }
        if num_warmup < 0 {
            return Err(Error::Other(format!(
                "num_warmup must be >= 0, got {}",
                num_warmup
            )));
        }
        check_seed(seed)?;

        let max_treedepth_opt = opt_count(&max_treedepth, "max_treedepth", 1)?;
        let mindepth_opt = opt_count(&mindepth, "mindepth", 0)?;
        let target_accept_opt = opt_finite_in_open_unit(&target_accept, "target_accept")?;
        let max_energy_error_opt = opt_finite_positive_f64(&max_energy_error, "max_energy_error")?;
        let extra_doublings_opt = opt_count(&extra_doublings, "extra_doublings", 0)?;

        let init_positions_raw: Option<Vec<Vec<f64>>> = if init_positions.is_null() {
            None
        } else {
            let lst = init_positions.as_list().ok_or_else(|| {
                Error::Other("init_positions must be a list of numeric vectors".into())
            })?;
            let mut out = Vec::with_capacity(lst.len());
            for (i, (_, el)) in lst.iter().enumerate() {
                let v = el.as_real_vector().ok_or_else(|| {
                    Error::Other(format!(
                        "init_positions[[{}]] must be a numeric vector",
                        i + 1
                    ))
                })?;
                out.push(v);
            }
            Some(out)
        };

        let stan_model = model::StanModel::new(&handle)
            .with_init_positions(init_positions_raw, jitter)
            .and_then(|m| m.with_constrain_flags(&handle, include_tp, include_gq))
            .map_err(r_err)?;

        let ndim = stan_model.num_constrained();
        let all_param_names: &[String] = stan_model.constrained_param_names();

        // Resolve keep_indices: NULL → keep all, else use as-supplied. Indices
        // are 0-based and must be within [0, ndim).
        let keep_cols: Vec<usize> = if keep_indices.is_null() {
            (0..ndim).collect()
        } else {
            let v = keep_indices.as_integer_vector().ok_or_else(|| {
                Error::Other("keep_indices must be NULL or an integer vector".into())
            })?;
            let mut out = Vec::with_capacity(v.len());
            for &idx in v.iter() {
                if idx < 0 || (idx as usize) >= ndim {
                    return Err(Error::Other(format!(
                        "keep_indices contains out-of-range value {} (ndim={})",
                        idx, ndim
                    )));
                }
                out.push(idx as usize);
            }
            out
        };
        let kept_param_names: Vec<String> = keep_cols
            .iter()
            .map(|&i| all_param_names[i].clone())
            .collect();
        let expand_error_count = stan_model.expand_error_count_handle();

        let num_tune = num_warmup as usize;
        let n_draws_per_chain = num_draws as usize;

        macro_rules! configure_settings {
            ($settings:expr) => {{
                $settings.num_tune = num_warmup as u64;
                $settings.num_draws = num_draws as u64;
                $settings.num_chains = num_chains as usize;
                $settings.seed = seed as u64;
                if let Some(v) = max_treedepth_opt {
                    $settings.maxdepth = v as u64;
                }
                if let Some(v) = mindepth_opt {
                    $settings.mindepth = v as u64;
                }
                if let Some(v) = target_accept_opt {
                    $settings.adapt_options.step_size_settings.target_accept = v;
                }
                if let Some(v) = max_energy_error_opt {
                    $settings.max_energy_error = v;
                }
                if let Some(v) = extra_doublings_opt {
                    $settings.extra_doublings = v as u64;
                }
                $settings.store_divergences = store_divergences;
                $settings.store_unconstrained = store_unconstrained;
                $settings.store_gradient = store_gradient;
                $settings
                    .adapt_options
                    .mass_matrix_options
                    .store_mass_matrix = store_mass_matrix;
            }};
        }

        let progress_callback = if progress_callback.is_null() {
            None
        } else {
            Some(progress_callback.as_function().ok_or_else(|| {
                Error::Other("progress_callback must be NULL or a function".into())
            })?)
        };

        let (results, sampler_config_json) = match adaptation {
            "low_rank" => {
                let mut settings = LowRankNutsSettings::default();
                configure_settings!(settings);
                if let Some(v) = opt_finite_positive_f64(&mass_matrix_gamma, "mass_matrix_gamma")? {
                    settings.adapt_options.mass_matrix_options.gamma = v;
                }
                if let Some(v) = opt_finite_positive_f64(&eigval_cutoff, "eigval_cutoff")? {
                    settings.adapt_options.mass_matrix_options.eigval_cutoff = v;
                }
                run_with_settings(
                    stan_model,
                    settings,
                    num_cores,
                    save_warmup,
                    progress_callback.clone(),
                    rprintf_progress,
                    chain_format,
                    console_width,
                    num_warmup as usize,
                )?
            }
            "diag" => {
                let mut settings = DiagNutsSettings::default();
                configure_settings!(settings);
                run_with_settings(
                    stan_model,
                    settings,
                    num_cores,
                    save_warmup,
                    progress_callback.clone(),
                    rprintf_progress,
                    chain_format,
                    console_width,
                    num_warmup as usize,
                )?
            }
            other => {
                return Err(Error::Other(format!(
                    "adaptation must be one of \"diag\" or \"low_rank\", got \"{}\"",
                    other
                )));
            }
        };

        let post_warmup_skip = if save_warmup { num_tune } else { 0 };

        let draws_robj = build_draws_matrix(
            &results,
            &keep_cols,
            post_warmup_skip,
            n_draws_per_chain,
            &kept_param_names,
        )?;

        // Diagnostic columns to suppress at the R boundary. nuts-rs 0.17.4
        // populates `unconstrained_draw` and `gradient` unconditionally
        // regardless of `store_unconstrained` / `store_gradient` (the flags
        // exist but are not read in `chain.rs::extract_stats`). To match the
        // documented R-side semantics — and to avoid surfacing one
        // `ndim_unc`-wide list-of-vectors per draw by default — drop those
        // columns here unless the user opts in. When nuts-rs starts honouring
        // the flags, the columns will already be all-null and our existing
        // `any_non_null` filter will drop them.
        let mut drop_cols: Vec<&str> = Vec::new();
        if !store_unconstrained {
            drop_cols.push("unconstrained_draw");
        }
        if !store_gradient {
            drop_cols.push("gradient");
        }

        let diagnostics =
            extract_diagnostics(&results, post_warmup_skip, n_draws_per_chain, &drop_cols)?;

        let warmup_draws_robj: Robj = if save_warmup {
            build_draws_matrix(&results, &keep_cols, 0, num_tune, &kept_param_names)?
        } else {
            ().into_robj()
        };
        let warmup_diagnostics_robj: Robj = if save_warmup {
            extract_diagnostics(&results, 0, num_tune, &drop_cols)?.into_robj()
        } else {
            ().into_robj()
        };

        let n_expand_errors = expand_error_count.load(Ordering::Relaxed) as i32;

        Ok(list!(
            draws = draws_robj,
            num_warmup = num_warmup,
            num_chains = num_chains,
            diagnostics = diagnostics,
            warmup_draws = warmup_draws_robj,
            warmup_diagnostics = warmup_diagnostics_robj,
            expand_errors = n_expand_errors,
            sampler_config = sampler_config_json
        ))
    })())
}

/// Optional finite scalar (`NULL` -> None, REAL scalar -> Some). Caller does
/// any range checks. Mirrors the R-side `check_count` so direct FFI callers
/// can't bypass validation.
fn opt_finite_f64(robj: &Robj, name: &str) -> Result<Option<f64>> {
    if robj.is_null() {
        return Ok(None);
    }
    let v: f64 = robj
        .as_real()
        .ok_or_else(|| Error::Other(format!("`{}` must be NULL or a single numeric.", name)))?;
    if !v.is_finite() {
        return Err(Error::Other(format!("`{}` must be a finite number.", name)));
    }
    Ok(Some(v))
}

fn opt_finite_positive_f64(robj: &Robj, name: &str) -> Result<Option<f64>> {
    let v = opt_finite_f64(robj, name)?;
    if let Some(x) = v {
        if x <= 0.0 {
            return Err(Error::Other(format!("`{}` must be > 0, got {}.", name, x)));
        }
    }
    Ok(v)
}

fn opt_finite_in_open_unit(robj: &Robj, name: &str) -> Result<Option<f64>> {
    let v = opt_finite_f64(robj, name)?;
    if let Some(x) = v {
        if x <= 0.0 || x >= 1.0 {
            return Err(Error::Other(format!(
                "`{}` must be in (0, 1), got {}.",
                name, x
            )));
        }
    }
    Ok(v)
}

fn opt_count(robj: &Robj, name: &str, min: i32) -> Result<Option<i32>> {
    let Some(v) = opt_finite_f64(robj, name)? else {
        return Ok(None);
    };
    if v.fract() != 0.0 {
        return Err(Error::Other(format!(
            "`{}` must be a whole number, got {}.",
            name, v
        )));
    }
    if v < min as f64 || v > i32::MAX as f64 {
        return Err(Error::Other(format!(
            "`{}` must be in [{}, {}], got {}.",
            name,
            min,
            i32::MAX,
            v
        )));
    }
    Ok(Some(v as i32))
}

/// Run the sampler with `settings` and return the traces alongside a JSON
/// snapshot of the effective settings (surfaced via `attr(draws, "sampler_config")`).
fn run_with_settings<S: Settings + serde::Serialize>(
    stan_model: model::StanModel,
    settings: S,
    num_cores: i32,
    save_warmup: bool,
    progress_callback: Option<Function>,
    rprintf_progress: &str,
    chain_format: &str,
    console_width: i32,
    num_warmup: usize,
) -> Result<(Vec<ArrowTrace>, String)> {
    let json = serde_json::to_string(&settings)
        .map_err(|e| Error::Other(format!("failed to serialize sampler settings: {}", e)))?;
    let traces = run_sampler(
        stan_model,
        settings,
        num_cores,
        save_warmup,
        progress_callback,
        rprintf_progress,
        chain_format,
        console_width,
        num_warmup,
    )?;
    Ok((traces, json))
}

/// Build a draws matrix from Arrow traces.
///
/// Materializes only the columns listed in `keep_cols` (0-indexed, against
/// the full constrained parameter dimension), writing directly into an
/// R-allocated `Doubles` in column-major order. On wide models with
/// large transformed-parameter / generated-quantities blocks this avoids
/// allocating columns the user is going to drop anyway.
///
/// `skip` is the number of initial Arrow rows to skip, `n_draws` is how
/// many to extract. `param_names` must already be filtered to match
/// `keep_cols`.
fn build_draws_matrix(
    results: &[ArrowTrace],
    keep_cols: &[usize],
    skip: usize,
    n_draws: usize,
    param_names: &[String],
) -> Result<Robj> {
    let n_chains = results.len();
    let total_rows = n_draws * n_chains;
    let n_kept = keep_cols.len();

    let mut out = Doubles::new(total_rows * n_kept);
    let dest: &mut [Rfloat] = &mut out;

    for (chain_idx, trace) in results.iter().enumerate() {
        let batch = &trace.posterior;
        let col = batch
            .column_by_name("value")
            .ok_or_else(|| Error::Other("No 'value' column in posterior".into()))?;

        let list_arr = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .ok_or_else(|| Error::Other("'value' column is not LargeList".into()))?;

        for draw in 0..n_draws {
            let row = skip + draw;
            let inner = list_arr.value(row);
            let values = inner
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| Error::Other("inner array is not Float64".into()))?;
            let row_data = values.values();

            let dest_row = chain_idx * n_draws + draw;
            for (out_col, &src_col) in keep_cols.iter().enumerate() {
                dest[dest_row + out_col * total_rows] = Rfloat::from(row_data[src_col]);
            }
        }
    }

    let mut robj: Robj = out.into();
    robj.set_attrib("dim", [total_rows as i32, n_kept as i32].into_robj())
        .map_err(r_err)?;
    if !param_names.is_empty() {
        let colnames: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
        let dimnames = List::from_values(&[
            ().into_robj(), // rownames = NULL
            colnames.into_robj(),
        ]);
        robj.set_attrib("dimnames", dimnames).ok();
    }
    Ok(robj)
}

/// Convert an Arrow column window (across all chains) into an R object.
///
/// `Float*` columns always become R `double`. `Int*`/`UInt*` columns become R
/// `integer` when every non-null value fits in `(i32::MIN, i32::MAX]`
/// (`i32::MIN` is reserved as `NA_INTEGER`); otherwise they fall back to
/// `double`. `logp` is `Float64` in the nuts-rs schema, so it can never reach
/// the integer arm — no risk of silently truncating a misrouted logp.
///
/// Returns `None` for unsupported Arrow types so the caller can warn-and-skip.
fn column_to_robj(
    cols: &[&dyn Array],
    dtype: &DataType,
    skip: usize,
    n_draws: usize,
    carry_forward: bool,
) -> Option<Robj> {
    let total = cols.len() * n_draws;

    macro_rules! build_doubles {
        ($arr_ty:ty) => {{
            let mut out = Doubles::new(total);
            let dest: &mut [Rfloat] = &mut out;
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<$arr_ty>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    dest[i] = if arr.is_null(row) {
                        Rfloat::na()
                    } else {
                        Rfloat::from(arr.value(row) as f64)
                    };
                    i += 1;
                }
            }
            Some(out.into())
        }};
    }

    macro_rules! build_int_or_double {
        ($arr_ty:ty, $fits:expr) => {{
            let arrs: Vec<&$arr_ty> = cols
                .iter()
                .map(|c| c.as_any().downcast_ref::<$arr_ty>())
                .collect::<Option<Vec<_>>>()?;
            let fits_i32 = arrs.iter().all(|arr| {
                (skip..skip + n_draws).all(|row| arr.is_null(row) || $fits(arr.value(row)))
            });
            if fits_i32 {
                let mut out = Integers::new(total);
                let dest: &mut [Rint] = &mut out;
                let mut i = 0;
                for arr in &arrs {
                    for k in 0..n_draws {
                        let row = skip + k;
                        dest[i] = if arr.is_null(row) {
                            Rint::na()
                        } else {
                            Rint::from(arr.value(row) as i32)
                        };
                        i += 1;
                    }
                }
                Some(out.into())
            } else {
                build_doubles!($arr_ty)
            }
        }};
    }

    match dtype {
        DataType::Boolean => {
            let mut out = Logicals::new(total);
            let dest: &mut [Rbool] = &mut out;
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<BooleanArray>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    dest[i] = if arr.is_null(row) {
                        Rbool::na()
                    } else {
                        Rbool::from(arr.value(row))
                    };
                    i += 1;
                }
            }
            Some(out.into())
        }
        DataType::Utf8 => {
            let mut out = Strings::new_with_na(total);
            let mut i = 0;
            for c in cols {
                let arr = c.as_any().downcast_ref::<StringArray>()?;
                for k in 0..n_draws {
                    let row = skip + k;
                    if !arr.is_null(row) {
                        out.set_elt(i, Rstr::from(arr.value(row)));
                    }
                    i += 1;
                }
            }
            Some(out.into())
        }
        DataType::Float64 => build_doubles!(Float64Array),
        DataType::Float32 => build_doubles!(Float32Array),
        DataType::Int64 => build_int_or_double!(Int64Array, |v: i64| v > i32::MIN as i64
            && v <= i32::MAX as i64),
        DataType::UInt64 => build_int_or_double!(UInt64Array, |v: u64| v <= i32::MAX as u64),
        DataType::Int32 => build_int_or_double!(Int32Array, |v: i32| v > i32::MIN),
        DataType::UInt32 => build_int_or_double!(UInt32Array, |v: u32| v <= i32::MAX as u32),
        DataType::LargeList(inner) if matches!(inner.data_type(), DataType::Float64) => {
            // Uniform-width rows (mass_matrix_inv / _eigvals / _stds — all
            // ndim_unc wide) collapse to a 2-D matrix; mixed widths fall
            // back to a list-of-vectors.
            let arrs: Vec<&LargeListArray> = cols
                .iter()
                .map(|c| c.as_any().downcast_ref::<LargeListArray>())
                .collect::<Option<Vec<_>>>()?;

            let mut uniform_len: Option<usize> = None;
            let mut mixed = false;
            let scan_start = if carry_forward { 0 } else { skip };
            let scan_end = skip + n_draws;
            'outer: for arr in &arrs {
                for row in scan_start..scan_end {
                    if arr.is_null(row) {
                        continue;
                    }
                    let len = arr.value_length(row) as usize;
                    match uniform_len {
                        None => uniform_len = Some(len),
                        Some(prev) if prev == len => {}
                        Some(_) => {
                            mixed = true;
                            break 'outer;
                        }
                    }
                }
            }

            if !mixed {
                if let Some(inner_len) = uniform_len {
                    if inner_len > 0 {
                        let mut out = Doubles::new(total * inner_len);
                        let dest: &mut [Rfloat] = &mut out;
                        for (chain_idx, arr) in arrs.iter().enumerate() {
                            let inner_values = arr
                                .values()
                                .as_any()
                                .downcast_ref::<Float64Array>()?
                                .values();
                            let offsets = arr.value_offsets();
                            // Carry the most recent non-null row's offset
                            // forward; `inner_values` outlives the loop, so
                            // a usize is enough — no per-row Vec clone.
                            let mut last_start: Option<usize> = if carry_forward {
                                (0..skip)
                                    .rev()
                                    .find(|&r| !arr.is_null(r))
                                    .map(|r| offsets[r] as usize)
                            } else {
                                None
                            };
                            for k in 0..n_draws {
                                let row = skip + k;
                                let dest_row = chain_idx * n_draws + k;
                                let src_start = if arr.is_null(row) {
                                    last_start
                                } else {
                                    let s = offsets[row] as usize;
                                    if carry_forward {
                                        last_start = Some(s);
                                    }
                                    Some(s)
                                };
                                match src_start {
                                    Some(start) => {
                                        let src = &inner_values[start..start + inner_len];
                                        for (col, &val) in src.iter().enumerate() {
                                            dest[dest_row + col * total] = Rfloat::from(val);
                                        }
                                    }
                                    None => {
                                        for col in 0..inner_len {
                                            dest[dest_row + col * total] = Rfloat::na();
                                        }
                                    }
                                }
                            }
                        }
                        let mut robj: Robj = out.into();
                        robj.set_attrib("dim", [total as i32, inner_len as i32].into_robj())
                            .ok()?;
                        return Some(robj);
                    }
                }
            }

            let mut out: Vec<Robj> = Vec::with_capacity(total);
            for arr in &arrs {
                let inner_values = arr
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()?
                    .values();
                let offsets = arr.value_offsets();
                for k in 0..n_draws {
                    let row = skip + k;
                    if arr.is_null(row) {
                        out.push(().into_robj());
                        continue;
                    }
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    let slice = &inner_values[start..end];
                    if slice.is_empty() {
                        out.push(().into_robj());
                    } else {
                        let robj: Robj = Doubles::from_values(slice.iter().copied()).into();
                        out.push(robj);
                    }
                }
            }
            Some(List::from_values(out).into_robj())
        }
        _ => None,
    }
}

/// Extract diagnostic statistics from sample_stats RecordBatches.
///
/// Schema-driven: iterates `sample_stats.schema().fields()` and dispatches each
/// column through `column_to_robj`. Columns that are entirely null in the
/// requested window are dropped (covers e.g. `mass_matrix_inv` when
/// `store_mass_matrix = false`). Unsupported Arrow types are skipped with a
/// warning rather than failing the whole call.
fn extract_diagnostics(
    results: &[ArrowTrace],
    skip: usize,
    n_draws: usize,
    drop_cols: &[&str],
) -> Result<List> {
    if results.is_empty() {
        return Ok(List::from_values(Vec::<Robj>::new()));
    }

    let schema = results[0].sample_stats.schema();
    let mut names: Vec<String> = Vec::new();
    let mut values: Vec<Robj> = Vec::new();

    for (idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();

        if drop_cols.iter().any(|c| c == name) {
            continue;
        }

        let cols: Vec<&dyn Array> = results
            .iter()
            .map(|t| t.sample_stats.column(idx).as_ref())
            .collect();

        // The mass-matrix snapshots only land on update rows; treat them
        // as piecewise-constant and carry the most recent value forward
        // through the NA gaps. Other columns get the default null-is-NA
        // behaviour. Explicit list, not a prefix match, so a future
        // upstream `mass_matrix_*` column doesn't silently inherit
        // carry-forward semantics it may not warrant.
        let carry_forward = matches!(
            name.as_str(),
            "mass_matrix_inv" | "mass_matrix_eigvals" | "mass_matrix_stds"
        );
        let non_null_start = if carry_forward { 0 } else { skip };
        let any_non_null = cols
            .iter()
            .any(|c| (non_null_start..skip + n_draws).any(|row| row < c.len() && !c.is_null(row)));
        if !any_non_null {
            continue;
        }

        match column_to_robj(&cols, field.data_type(), skip, n_draws, carry_forward) {
            Some(robj) => {
                names.push(name.clone());
                values.push(robj);
            }
            None => {
                rprintln!(
                    "nutpieR: skipping diagnostic '{}' — unsupported Arrow type {:?}",
                    name,
                    field.data_type()
                );
            }
        }
    }

    let pairs: Vec<(&str, Robj)> = names.iter().map(String::as_str).zip(values).collect();
    Ok(List::from_pairs(pairs))
}

/// Open a BridgeStan model and return an `ExternalPtr<BSHandle>` that caches
/// parameter-name metadata. The handle may be used by any of the `bs_*`
/// accessor functions without re-opening the shared library.
/// @noRd
#[extendr]
fn bs_open(lib_path: &str, data_json: &str, seed: i32) -> Robj {
    or_throw((|| -> Result<Robj> {
        let seed_u32 = check_seed(seed)?;
        let handle = model::BSHandle::open(std::path::Path::new(lib_path), data_json, seed_u32)
            .map_err(r_err)?;
        Ok(ExternalPtr::new(handle).into_robj())
    })())
}

/// Block-level parameter names (no transformed parameters / generated
/// quantities), dot-indexed. Length equals `bs_ndim_block()`.
/// @noRd
#[extendr]
fn bs_block_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.block_names.clone()
}

/// Block-level + transformed-parameter names (no generated quantities),
/// dot-indexed. Length equals `param_num(true, false)`. Used by R-side
/// `pars` / `include` resolution to partition names into block / TP / GQ
/// without an extra round-trip into bridgestan.
/// @noRd
#[extendr]
fn bs_block_tp_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.block_tp_names.clone()
}

/// Full constrained parameter names (block + transformed parameters +
/// generated quantities), dot-indexed.
/// @noRd
#[extendr]
fn bs_full_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.full_names.clone()
}

/// Unconstrained parameter names, dot-indexed. Length equals `bs_ndim_unc()`.
/// @noRd
#[extendr]
fn bs_unc_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.unc_names.clone()
}

/// Number of unconstrained parameters.
/// @noRd
#[extendr]
fn bs_ndim_unc(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_unc as i32
}

/// Number of block-level constrained parameters (no TP, no GQ).
/// @noRd
#[extendr]
fn bs_ndim_block(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_block as i32
}

/// Map a flat block-level constrained vector (length `bs_ndim_block()`,
/// BridgeStan column-major / last-index-major order) to the unconstrained
/// space. No JSON parsing.
/// @noRd
#[extendr]
fn bs_param_unconstrain(handle: ExternalPtr<model::BSHandle>, theta: Vec<f64>) -> Vec<f64> {
    or_throw(bs_param_unconstrain_impl(handle, theta))
}

fn bs_param_unconstrain_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta: Vec<f64>,
) -> Result<Vec<f64>> {
    if theta.len() != handle.ndim_block {
        return Err(Error::Other(format!(
            "theta length {} does not match block-level parameter count {}",
            theta.len(),
            handle.ndim_block
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_unc];
    handle
        .model
        .param_unconstrain(&theta, &mut out)
        .map_err(r_err)?;
    Ok(out)
}

/// Map an unconstrained position to the full constrained scale (including
/// transformed parameters and generated quantities) using an already-opened
/// handle.
/// @noRd
#[extendr]
fn bs_param_constrain(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
    seed: i32,
) -> Vec<f64> {
    or_throw(bs_param_constrain_impl(handle, theta_unc, seed))
}

fn bs_param_constrain_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
    seed: i32,
) -> Result<Vec<f64>> {
    let seed_u32 = check_seed(seed)?;
    if theta_unc.len() != handle.ndim_unc {
        return Err(Error::Other(format!(
            "theta_unc length {} does not match unconstrained parameter count {}",
            theta_unc.len(),
            handle.ndim_unc
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_full];
    let mut rng = handle.model.new_rng(seed_u32).map_err(r_err)?;
    handle
        .model
        .param_constrain(&theta_unc, true, true, &mut out, Some(&mut rng))
        .map_err(r_err)?;
    Ok(out)
}

/// Map an unconstrained position to the block-level constrained scale only
/// (no transformed parameters, no generated quantities). No RNG is used and
/// no GQ code runs, so this cannot fail on GQ constraint violations — the
/// right primitive for resolving partial-init random fills.
/// @noRd
#[extendr]
fn bs_param_constrain_block(handle: ExternalPtr<model::BSHandle>, theta_unc: Vec<f64>) -> Vec<f64> {
    or_throw(bs_param_constrain_block_impl(handle, theta_unc))
}

fn bs_param_constrain_block_impl(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
) -> Result<Vec<f64>> {
    if theta_unc.len() != handle.ndim_unc {
        return Err(Error::Other(format!(
            "theta_unc length {} does not match unconstrained parameter count {}",
            theta_unc.len(),
            handle.ndim_unc
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_block];
    let no_rng: Option<&mut bridgestan::Rng<Arc<bridgestan::StanLibrary>>> = None;
    handle
        .model
        .param_constrain(&theta_unc, false, false, &mut out, no_rng)
        .map_err(r_err)?;
    Ok(out)
}

extendr_module! {
    mod nutpieR;
    fn bridgestan_version;
    fn compile_stan_model;
    fn sample_stan;
    fn bs_open;
    fn bs_block_names;
    fn bs_block_tp_names;
    fn bs_full_names;
    fn bs_unc_names;
    fn bs_ndim_unc;
    fn bs_ndim_block;
    fn bs_param_unconstrain;
    fn bs_param_constrain;
    fn bs_param_constrain_block;
}
