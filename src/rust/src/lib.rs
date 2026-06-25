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
    // Write to R's output stream (stdout) — C-level, no SEXP allocation.
    // Used for progress rendering that avoids R's GC (#36).
    fn Rprintf(format: *const std::os::raw::c_char, ...);
    // Write to R's error stream (stderr) — C-level, no SEXP allocation.
    fn REprintf(format: *const std::os::raw::c_char, ...);
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

/// Write a message to R's stderr (REprintf) — C-level, no SEXP allocation.
/// Used for progress rendering during sampling to avoid triggering R's GC
/// while `tbbmalloc_proxy` is active (#36).
fn r_eprint(msg: &str) {
    // REprintf is a printf-style C function; use "%s" to avoid interpreting
    // any '%' characters in `msg` as format specifiers.
    let fmt = b"%s\0";
    let c_msg = std::ffi::CString::new(msg).unwrap_or_default();
    unsafe {
        REprintf(
            fmt.as_ptr() as *const std::os::raw::c_char,
            c_msg.as_ptr(),
        );
    }
}

/// Write to R's stdout (Rprintf) — same rationale as `r_eprint`.
fn r_print(msg: &str) {
    let fmt = b"%s\0";
    let c_msg = std::ffi::CString::new(msg).unwrap_or_default();
    unsafe {
        Rprintf(fmt.as_ptr() as *const std::os::raw::c_char, c_msg.as_ptr());
    }
}

/// Detect whether R's output stream supports ANSI color escape sequences.
/// On macOS, R's stdout is a TTY when running interactively in Terminal or
/// RStudio's console. We use `isatty(1)` as the signal — this matches the
/// R-side `cli::num_ansi_colors()` heuristic well enough for the progress bar.
fn detect_r_color_support() -> bool {
    // Use isatty(1) via the C library (linked via R) to detect TTY.
    // This avoids adding a `libc` crate dependency.
    #[cfg(unix)]
    {
        extern "C" {
            fn isatty(fd: std::os::raw::c_int) -> std::os::raw::c_int;
        }
        unsafe { isatty(1) == 1 }
    }
    #[cfg(not(unix))]
    {
        false
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
/// any dependency on R's cli package — these go through Rprintf which writes
/// to stdout. Colors are only emitted when the terminal supports them.
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_DIM: &str = "\x1b[2m";

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
                steps_delta += (c.total_num_steps - base_total) as f64;
                draws_delta += (c.finished_draws - base_finished) as f64;
            }
        }
        if draws_delta > 0.0 {
            Some(steps_delta / draws_delta)
        } else {
            None
        }
    }
}

/// Format a one-line progress summary for Rprintf rendering (#36).
///
/// This replaces the R callback path when `tbbmalloc_proxy` is active on macOS.
/// By writing via `REprintf`/`Rprintf` (C FFI, no SEXP allocation), we avoid
/// triggering R's GC during sampling — the root cause of the
/// `__TBB_malloc_safer_msize` segfault.
///
/// Style: `"bar"` overwrites the line with `\r` and draws a Unicode progress
/// bar with ANSI colors; `"text"` emits a fresh line with `\n`.
fn format_progress_line(
    state: &[ChainState],
    style: &str,
    use_color: bool,
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

    // Per-chain draw range
    let min_finished = started.iter().map(|c| c.finished_draws).min().unwrap_or(0);
    let max_finished = started.iter().map(|c| c.finished_draws).max().unwrap_or(0);

    // Infer tree depth from max latest_num_steps
    let max_steps = started.iter().map(|c| c.latest_num_steps).max().unwrap_or(0);
    let tdepth = if max_steps > 0 {
        (max_steps as f64).log2().floor() as i32
    } else {
        0
    };

    // Min step size
    let min_step = started
        .iter()
        .map(|c| c.step_size)
        .filter(|&s| s > 0.0)
        .fold(f64::INFINITY, f64::min);

    let n_chains = started.len();

    // --- One-shot hints (fire at most once per run) ---

    // Divergence hint
    if !hints.warned_div && total_divs > 0 {
        hints.warned_div = true;
        r_eprint("⚠ div: divergent transitions detected — try increasing target_accept or reparameterizing.\n");
    }

    // Grad/draw hint
    if !hints.warned_grad && baseline_grad.is_some() && avg_grad >= GRAD_HINT_THRESHOLD {
        hints.warned_grad = true;
        let depth = (avg_grad + 1.0).log2().round() as i32;
        r_eprint(&format!("ℹ grad/draw: ~{} gradient evaluations per draw (tree depth ~{}) — sampling is taking long trajectories; often a sign of difficult geometry or incomplete adaptation, worth checking if unexpected.\n",
            avg_grad.round() as i32, depth));
    }

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
                r_eprint(&format!("ℹ spread: chain progress is uneven (slowest {}%, fastest {}%) — often one chain adapted a smaller step size or is in a harder region of the posterior. Adding to status line.\n",
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

    // --- Sparkline (gap-from-leader, one glyph per running chain) ---
    let spark_part = if running.len() > 1 && running.iter().all(|c| c.total_draws > 0) {
        let max_total = running.iter().map(|c| c.total_draws).max().unwrap_or(0);
        if max_total > 0 {
            let max_fin = running.iter().map(|c| c.finished_draws).max().unwrap_or(0);
            let glyphs = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            let s: String = running.iter().map(|c| {
                let gap = max_fin - c.finished_draws;
                let ratio = gap as f64 / max_total as f64;
                let lo = 0.02;
                let hi = 0.20;
                let frac = ((ratio - lo) / (hi - lo)).clamp(0.0, 1.0);
                let level = (frac * 7.0).round() as usize;
                glyphs[level.min(7)]
            }).collect();
            Some(s)
        } else {
            None
        }
    } else {
        None
    };

    // --- Chain lag indicator ---
    let lag_part = if running.len() > 1 {
        let finished: Vec<f64> = running.iter().map(|c| c.finished_draws as f64).collect();
        let mut sorted = finished.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        if med > 0.0 {
            let lag_idx: Vec<usize> = finished.iter().enumerate()
                .filter(|(_, &f)| f < med * 0.9)
                .map(|(i, _)| i)
                .collect();
            if !lag_idx.is_empty() {
                let labels: Vec<String> = lag_idx.iter()
                    .map(|&i| format!("c{}", running[i].chain))
                    .collect();
                Some(format!("{} slow", labels.join(",")))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Build the status portion: div | grad | spread | spark | lag | draws | tdepth | step
    let div_part = if total_divs > 0 {
        if use_color {
            format!("{}⚠ div: {}{}", ANSI_YELLOW, total_divs, ANSI_RESET)
        } else {
            format!("⚠ div: {}", total_divs)
        }
    } else {
        "div: 0".to_string()
    };

    let grad_part = if avg_grad > 0.0 {
        let label = format!("{:.0} grad/draw", avg_grad);
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

    let draws_part = if min_finished == max_finished {
        format!("{}/{}", min_finished, total_draws / n_chains)
    } else {
        format!("{}-{}/{}", min_finished, max_finished, total_draws / n_chains)
    };

    let tdepth_part = format!("tdepth: {}", tdepth);
    let step_part = if min_step.is_finite() && min_step > 0.0 {
        format!("step: {:.3}", min_step)
    } else {
        String::new()
    };

    // Assemble status tokens, filtering out None
    let mut tokens: Vec<String> = vec![div_part, grad_part];
    if let Some(ref s) = spread_part { tokens.push(s.clone()); }
    if let Some(ref s) = spark_part { tokens.push(s.clone()); }
    if let Some(ref s) = lag_part { tokens.push(s.clone()); }
    tokens.push(draws_part);
    tokens.push(tdepth_part);
    if !step_part.is_empty() { tokens.push(step_part); }
    let status = tokens.join(" | ");

    match style {
        "bar" => {
            let bar_width = 24usize;
            let filled = if total_draws > 0 {
                (bar_width * total_finished / total_draws).min(bar_width)
            } else {
                0
            };
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            let spinner = SPINNER[frame % SPINNER.len()];

            format!(
                "\r{spinner} {elapsed:.1}s {phase} {pct}% {bar} ETA {eta} | {status}",
                spinner = spinner,
                elapsed = elapsed,
                phase = phase,
                pct = pct,
                bar = bar,
                eta = eta_str,
                status = status,
            )
        }
        _ => {
            format!(
                "[{:.1}s] {} {}% | {}",
                elapsed, phase, pct, status,
            )
        }
    }
}

/// Format seconds as a compact time string (e.g. "1m23s", "45s", "3m").
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

    // Throttle Rprintf rendering to ~1 Hz. The poll loop still spins at 200 ms
    // for interrupt responsiveness and console flushing, but we only render a
    // progress line once per second to avoid flooding the console.
    let cb_interval = Duration::from_millis(
        std::env::var("NUTPIE_CB_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000),
    );
    let mut last_render: Option<Instant> = None;
    let mut last_line = String::new();
    let mut frame: usize = 0;
    // Detect color support: check if stdout is a TTY and R has not disabled it.
    // We use the same heuristic as cli: if R's stdout is a terminal, assume
    // color support. This is conservative — it may miss some terminals that
    // support color but aren't detected as TTYs.
    let use_color = rprintf_progress == "bar" && detect_r_color_support();
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

                // Throttle: only render progress once per `cb_interval`, not
                // on every 200 ms poll.
                let due = last_render.map_or(true, |t| t.elapsed() >= cb_interval);
                if due {
                    let line = format_progress_line(
                        &state_snapshot,
                        rprintf_progress,
                        use_color,
                        frame,
                        &mut hints,
                        num_warmup,
                    );
                    frame = frame.wrapping_add(1);
                    // Skip rendering if the line hasn't changed since last render.
                    // In bar mode the \r overwrites, so duplicates are invisible
                    // in a real terminal but produce noise in piped/captured output.
                    let changed = line != last_line;
                    if !line.is_empty() && changed {
                        if rprintf_progress == "bar" {
                            r_print(&line);
                        } else {
                            r_eprint(&format!("{}\n", line));
                        }
                        pump_r_events();
                        last_line = line;
                    }
                    last_render = Some(Instant::now());

                    // If an R callback is also set (non-macOS or explicit),
                    // run it too — but only when not on the Rprintf path.
                    if !use_rprintf {
                        if let Some(ref cb) = progress_cb {
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
                }
            }
            SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
        }
    };

    // Final progress render / callback.
    if use_rprintf {
        // Clear the progress bar line with a newline so the summary starts fresh.
        if rprintf_progress == "bar" {
            r_eprint("\n");
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
