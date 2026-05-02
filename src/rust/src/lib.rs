#![allow(non_snake_case)]

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeListArray,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use extendr_api::prelude::*;
use extendr_api::error::Result;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, DiagNutsSettings, LowRankNutsSettings,
    ProgressCallback, Sampler, SamplerWaitResult, Settings,
};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    let bs_path = bridgestan::download_bridgestan_src().map_err(r_err)?;
    let stan_path = PathBuf::from(stan_file);

    let stanc_vec: Vec<String> = if stanc_args.len() == 0 {
        Vec::new()
    } else {
        stanc_args.iter().map(|s| s.to_string()).collect()
    };
    let stanc_refs: Vec<&str> = stanc_vec.iter().map(String::as_str).collect();
    let compile_vec: Vec<String> = if compile_args.len() == 0 {
        Vec::new()
    } else {
        compile_args.iter().map(|s| s.to_string()).collect()
    };
    let compile_refs: Vec<&str> = compile_vec.iter().map(String::as_str).collect();

    let lib_path = bridgestan::compile_model(&bs_path, &stan_path, &stanc_refs, &compile_refs)
        .map_err(r_err)?;
    Ok(lib_path.to_string_lossy().into_owned())
}

/// Sample from a Stan model using nuts-rs NUTS sampler.
/// Run the sampler with progress reporting. Generic over Settings type.
fn run_sampler<S: Settings>(
    stan_model: model::StanModel,
    settings: S,
    num_chains: i32,
    num_draws: i32,
    num_warmup: i32,
    num_cores: i32,
    refresh: i32,
) -> Result<Vec<ArrowTrace>> {
    let show_progress = refresh > 0;
    let refresh = refresh.max(0) as usize;

    #[derive(Clone, Default)]
    struct ChainState {
        finished_draws: usize,
        total_draws: usize,
        divergences: usize,
        tuning: bool,
        step_size: f64,
    }

    let progress_state: Arc<Mutex<Vec<ChainState>>> =
        Arc::new(Mutex::new(Vec::new()));
    let state_clone = progress_state.clone();

    let callback = if show_progress {
        Some(ProgressCallback {
            callback: Box::new(
                move |_elapsed: Duration, progress: Box<[ChainProgress]>| {
                    let snapshot: Vec<ChainState> = progress
                        .iter()
                        .map(|c| ChainState {
                            finished_draws: c.finished_draws,
                            total_draws: c.total_draws,
                            divergences: c.divergences,
                            tuning: c.tuning,
                            step_size: c.step_size,
                        })
                        .collect();
                    *state_clone.lock().unwrap() = snapshot;
                },
            ),
            rate: Duration::from_millis(100),
        })
    } else {
        None
    };

    let mut sampler_opt = Some(
        Sampler::new(stan_model, settings, ArrowConfig::new(), num_cores as usize, callback)
            .map_err(r_err)?,
    );

    let start = Instant::now();
    let mut last_reported: Vec<usize> = vec![0; num_chains as usize];

    if show_progress {
        rprintln!(
            "Sampling {} chains, {} draws each ({} warmup)...\n",
            num_chains,
            num_draws,
            num_warmup
        );
    }

    let results = loop {
        let sampler = sampler_opt.take().unwrap();
        let wait_dur = if show_progress {
            Duration::from_millis(200)
        } else {
            Duration::from_secs(600)
        };
        match sampler.wait_timeout(wait_dur) {
            SamplerWaitResult::Trace(traces) => break traces,
            SamplerWaitResult::Timeout(s) => {
                sampler_opt = Some(s);
                if show_progress {
                    let state = progress_state.lock().unwrap();
                    if !state.is_empty() {
                        for (i, chain) in state.iter().enumerate() {
                            let draws_since = chain.finished_draws.saturating_sub(last_reported[i]);
                            if draws_since >= refresh {
                                let phase = if chain.tuning { "warmup" } else { "sample" };
                                let elapsed = start.elapsed().as_secs_f64();
                                if chain.divergences > 0 {
                                    rprintln!(
                                        "  Chain {} [{phase}] {}/{} draws ({} divergences, step size {:.3}) [{:.1}s]",
                                        i + 1,
                                        chain.finished_draws,
                                        chain.total_draws,
                                        chain.divergences,
                                        chain.step_size,
                                        elapsed
                                    );
                                } else {
                                    rprintln!(
                                        "  Chain {} [{phase}] {}/{} draws (step size {:.3}) [{:.1}s]",
                                        i + 1,
                                        chain.finished_draws,
                                        chain.total_draws,
                                        chain.step_size,
                                        elapsed
                                    );
                                }
                                last_reported[i] = chain.finished_draws;
                            }
                        }
                    }
                }
            }
            SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
        }
    };

    if show_progress {
        let elapsed = start.elapsed().as_secs_f64();
        rprintln!("\nSampling complete ({:.1}s)", elapsed);
    }

    Ok(results)
}

/// @param handle An `ExternalPtr<BSHandle>` from `bs_open()`.
/// @param num_draws Number of draws per chain after warmup.
/// @param num_warmup Number of warmup (tuning) draws per chain.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @param refresh Print progress every `refresh` draws per chain (0 = no progress).
/// @param init_positions Optional list of numeric vectors (one per chain, or length 1 = broadcast).
/// @param jitter If TRUE, apply ±0.5 uniform jitter per coordinate.
/// @param save_warmup Whether to return warmup draws.
/// @param num_cores Number of CPU cores to use for parallel sampling.
/// @param store_divergences Whether to store detailed divergence information.
/// @param store_mass_matrix Whether to store the mass matrix at each draw.
/// @param store_unconstrained Whether to store the unconstrained position at each draw.
/// @param store_gradient Whether to store the gradient at each draw.
/// @param adaptation One of "diag" or "low_rank".
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
/// @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
///   sampler_config (JSON), and optionally warmup_draws and warmup_diagnostics.
/// @noRd
#[extendr]
fn sample_stan(
    handle: ExternalPtr<model::BSHandle>,
    num_draws: i32,
    num_warmup: i32,
    num_chains: i32,
    seed: i32,
    refresh: i32,
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
            return Err(Error::Other(format!(
                "{} must be >= 1, got {}",
                name, val
            )));
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
    let target_accept_opt =
        opt_finite_in_open_unit(&target_accept, "target_accept")?;
    let max_energy_error_opt = opt_finite_positive_f64(&max_energy_error, "max_energy_error")?;
    let extra_doublings_opt = opt_count(&extra_doublings, "extra_doublings", 0)?;

    let init_positions_raw: Option<Vec<Vec<f64>>> = if init_positions.is_null() {
        None
    } else {
        let lst = init_positions
            .as_list()
            .ok_or_else(|| Error::Other("init_positions must be a list of numeric vectors".into()))?;
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
        let v = keep_indices
            .as_integer_vector()
            .ok_or_else(|| Error::Other("keep_indices must be NULL or an integer vector".into()))?;
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
    let kept_param_names: Vec<String> =
        keep_cols.iter().map(|&i| all_param_names[i].clone()).collect();
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
            $settings.adapt_options.mass_matrix_options.store_mass_matrix = store_mass_matrix;
        }};
    }

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
            run_with_settings(stan_model, settings, num_chains, num_draws, num_warmup, num_cores, refresh)?
        }
        "diag" => {
            let mut settings = DiagNutsSettings::default();
            configure_settings!(settings);
            run_with_settings(stan_model, settings, num_chains, num_draws, num_warmup, num_cores, refresh)?
        }
        other => {
            return Err(Error::Other(format!(
                "adaptation must be one of \"diag\" or \"low_rank\", got \"{}\"",
                other
            )));
        }
    };

    let draws_robj = build_draws_matrix(
        &results,
        &keep_cols,
        num_tune,
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

    let diagnostics = extract_diagnostics(&results, num_tune, n_draws_per_chain, &drop_cols)?;

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
        if !(x > 0.0) {
            return Err(Error::Other(format!("`{}` must be > 0, got {}.", name, x)));
        }
    }
    Ok(v)
}

fn opt_finite_in_open_unit(robj: &Robj, name: &str) -> Result<Option<f64>> {
    let v = opt_finite_f64(robj, name)?;
    if let Some(x) = v {
        if !(x > 0.0 && x < 1.0) {
            return Err(Error::Other(format!("`{}` must be in (0, 1), got {}.", name, x)));
        }
    }
    Ok(v)
}

fn opt_count(robj: &Robj, name: &str, min: i32) -> Result<Option<i32>> {
    let Some(v) = opt_finite_f64(robj, name)? else { return Ok(None); };
    if v.fract() != 0.0 {
        return Err(Error::Other(format!("`{}` must be a whole number, got {}.", name, v)));
    }
    if v < min as f64 || v > i32::MAX as f64 {
        return Err(Error::Other(format!(
            "`{}` must be in [{}, {}], got {}.",
            name, min, i32::MAX, v
        )));
    }
    Ok(Some(v as i32))
}

/// Run the sampler with `settings` and return the traces alongside a JSON
/// snapshot of the effective settings (surfaced via `attr(draws, "sampler_config")`).
fn run_with_settings<S: Settings + serde::Serialize>(
    stan_model: model::StanModel,
    settings: S,
    num_chains: i32,
    num_draws: i32,
    num_warmup: i32,
    num_cores: i32,
    refresh: i32,
) -> Result<(Vec<ArrowTrace>, String)> {
    let json = serde_json::to_string(&settings)
        .map_err(|e| Error::Other(format!("failed to serialize sampler settings: {}", e)))?;
    let traces = run_sampler(
        stan_model, settings, num_chains, num_draws, num_warmup, num_cores, refresh,
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
        DataType::Float64 => build_doubles!(Float64Array),
        DataType::Float32 => build_doubles!(Float32Array),
        DataType::Int64 => build_int_or_double!(
            Int64Array,
            |v: i64| v > i32::MIN as i64 && v <= i32::MAX as i64
        ),
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
                            let mut last: Option<Vec<f64>> = None;
                            if carry_forward && skip > 0 {
                                for row in 0..skip {
                                    if !arr.is_null(row) {
                                        let start = offsets[row] as usize;
                                        last = Some(inner_values[start..start + inner_len].to_vec());
                                    }
                                }
                            }
                            for k in 0..n_draws {
                                let row = skip + k;
                                let dest_row = chain_idx * n_draws + k;
                                let row_values: Option<&[f64]> = if arr.is_null(row) {
                                    last.as_deref()
                                } else {
                                    let start = offsets[row] as usize;
                                    last = Some(inner_values[start..start + inner_len].to_vec());
                                    last.as_deref()
                                };
                                match row_values {
                                    Some(vals) => {
                                        for (col, &val) in vals.iter().enumerate() {
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
                        robj.set_attrib(
                            "dim",
                            [total as i32, inner_len as i32].into_robj(),
                        )
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

        // Drop columns that are entirely null in the requested window
        // (e.g. mass_matrix_inv / divergence_* when their flags are off).
        let carry_forward = name.starts_with("mass_matrix");
        let non_null_start = if carry_forward { 0 } else { skip };
        let any_non_null = cols.iter().any(|c| {
            (non_null_start..skip + n_draws).any(|row| row < c.len() && !c.is_null(row))
        });
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
fn bs_param_unconstrain(
    handle: ExternalPtr<model::BSHandle>,
    theta: Vec<f64>,
) -> Vec<f64> {
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
fn bs_param_constrain_block(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
) -> Vec<f64> {
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
