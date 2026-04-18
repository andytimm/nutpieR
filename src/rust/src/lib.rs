#![allow(non_snake_case)]

use arrow::array::Array;
use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, CpuLogpFunc, CpuMath, DiagGradNutsSettings, HasDims,
    LowRankNutsSettings, Model, ProgressCallback, Sampler, SamplerWaitResult,
    Settings,
};
use rand::RngExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod model;

/// Convert any Display error to an extendr Error.
fn r_err(e: impl std::fmt::Display) -> Error {
    Error::Other(e.to_string())
}

// --- 10-d standard normal: logp = -0.5 * sum(x^2) ---

const NDIM: usize = 10;

#[derive(Debug)]
struct NormalLogp;

#[derive(Debug, thiserror::Error)]
#[error("normal logp error")]
struct NormalError;

impl nuts_rs::LogpError for NormalError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

impl HasDims for NormalLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        m.insert("dim".to_string(), NDIM as u64);
        m
    }
}

impl CpuLogpFunc for NormalLogp {
    type LogpError = NormalError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        NDIM
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let mut logp = 0.0;
        for i in 0..NDIM {
            logp -= 0.5 * position[i] * position[i];
            gradient[i] = -position[i];
        }
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, nuts_rs::CpuMathError> {
        Ok(array.to_vec())
    }
}

// --- Model factory: creates one NormalLogp per chain ---

struct NormalModel;

impl Model for NormalModel {
    type Math<'model> = CpuMath<NormalLogp>;

    fn math<R: rand::Rng + ?Sized>(
        &self,
        _rng: &mut R,
    ) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(NormalLogp))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        for p in position.iter_mut() {
            *p = rng.random_range(-2.0..2.0);
        }
        Ok(())
    }
}

/// Sample from a 10-d standard normal using nuts-rs.
/// Returns a matrix of draws (rows = draws*chains, cols = parameters).
/// @param num_draws Number of draws per chain after warmup.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @keywords internal
#[extendr]
fn sample_normal(num_draws: i32, num_chains: i32, seed: i32) -> Result<Robj> {
    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = 300;
    settings.num_draws = num_draws as u64;
    settings.num_chains = num_chains as usize;
    settings.seed = seed as u64;

    let model = NormalModel;
    let trace_config = ArrowConfig::new();

    let sampler = Sampler::new(model, settings, trace_config, num_chains as usize, None)
        .map_err(r_err)?;

    let results = match sampler.wait_timeout(Duration::from_secs(300)) {
        SamplerWaitResult::Trace(traces) => traces,
        SamplerWaitResult::Timeout(_) => return Err(Error::Other("Sampling timed out".into())),
        SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
    };

    let n_chains = results.len();
    let num_tune = settings.num_tune as usize;
    let n_draws_per_chain = num_draws as usize;
    let total_rows = n_draws_per_chain * n_chains;

    let mut data = vec![0.0f64; total_rows * NDIM];

    for (chain_idx, trace) in results.iter().enumerate() {
        let batch = &trace.posterior;
        let col = batch
            .column_by_name("value")
            .ok_or_else(|| Error::Other("No 'value' column in posterior".into()))?;

        let list_arr = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .ok_or_else(|| Error::Other("'value' column is not LargeList".into()))?;

        for draw in 0..n_draws_per_chain {
            let row = num_tune + draw;
            let inner = list_arr.value(row);
            let values = inner
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| Error::Other("inner array is not Float64".into()))?;

            for param in 0..NDIM {
                let row_idx = chain_idx * n_draws_per_chain + draw;
                data[row_idx + param * total_rows] = values.value(param);
            }
        }
    }

    let matrix = RMatrix::new_matrix(total_rows, NDIM, |r, c| data[r + c * total_rows]);
    Ok(matrix.into_robj())
}

/// Compile a Stan model to a shared library using BridgeStan.
/// Downloads BridgeStan sources if needed (first call is slow).
/// @param stan_file Path to the .stan file.
/// @param stanc_args Character vector of extra arguments for stanc compiler.
/// @param compile_args Character vector of extra arguments for make.
/// @return Path to the compiled shared library.
/// @keywords internal
#[extendr]
fn compile_stan_model(stan_file: &str, stanc_args: Strings, compile_args: Strings) -> Result<String> {
    let bs_path = bridgestan::download_bridgestan_src().map_err(r_err)?;
    let stan_path = PathBuf::from(stan_file);

    let stanc_vec: Vec<String> = stanc_args.iter().map(|s| s.to_string()).collect();
    let stanc_refs: Vec<&str> = stanc_vec.iter().map(String::as_str).collect();
    let compile_vec: Vec<String> = compile_args.iter().map(|s| s.to_string()).collect();
    let compile_refs: Vec<&str> = compile_vec.iter().map(String::as_str).collect();

    let lib_path =
        bridgestan::compile_model(&bs_path, &stan_path, &stanc_refs, &compile_refs).map_err(r_err)?;
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
/// @param max_treedepth Maximum tree depth for NUTS.
/// @param target_accept Target acceptance probability for step size adaptation.
/// @param refresh Print progress every `refresh` draws per chain (0 = no progress).
/// @param init_positions Optional list of numeric vectors (one per chain, or length 1 = broadcast).
/// @param jitter If TRUE, apply ±0.5 uniform jitter per coordinate.
/// @param save_warmup Whether to return warmup draws.
/// @param num_cores Number of CPU cores to use for parallel sampling.
/// @param store_divergences Whether to store detailed divergence information.
/// @param store_mass_matrix Whether to store the mass matrix at each draw.
/// @param low_rank Whether to use low-rank modified mass matrix adaptation.
/// @param mass_matrix_gamma Regularisation parameter for low-rank mass matrix (default 1e-5).
/// @param eigval_cutoff Eigenvalue cutoff for low-rank mass matrix (default 2.0).
/// @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
///   and optionally warmup_draws and warmup_diagnostics.
/// @keywords internal
#[extendr]
fn sample_stan(
    handle: ExternalPtr<model::BSHandle>,
    num_draws: i32,
    num_warmup: i32,
    num_chains: i32,
    seed: i32,
    max_treedepth: i32,
    target_accept: f64,
    refresh: i32,
    init_positions: Robj,
    jitter: bool,
    save_warmup: bool,
    num_cores: i32,
    store_divergences: bool,
    store_mass_matrix: bool,
    low_rank: bool,
    mass_matrix_gamma: f64,
    eigval_cutoff: f64,
) -> Result<List> {
    // Parse init_positions (NULL or list of numeric vectors) — per-chain path.
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
        .map_err(r_err)?;

    let ndim = stan_model.num_constrained();
    let param_names: Vec<String> = stan_model.constrained_param_names().to_vec();
    let expand_error_count = stan_model.expand_error_count_handle();

    let num_tune = num_warmup as usize;
    let n_draws_per_chain = num_draws as usize;

    // Branch on mass matrix type — both implement Settings but are different types.
    // Common fields are set via the configure_settings! macro to avoid duplication.
    macro_rules! configure_settings {
        ($settings:expr) => {{
            $settings.num_tune = num_warmup as u64;
            $settings.num_draws = num_draws as u64;
            $settings.num_chains = num_chains as usize;
            $settings.seed = seed as u64;
            $settings.maxdepth = max_treedepth as u64;
            $settings.adapt_options.step_size_settings.target_accept = target_accept;
            $settings.store_divergences = store_divergences;
            $settings.adapt_options.mass_matrix_options.store_mass_matrix = store_mass_matrix;
        }};
    }

    let results = if low_rank {
        let mut settings = LowRankNutsSettings::default();
        configure_settings!(settings);
        settings.adapt_options.mass_matrix_options.gamma = mass_matrix_gamma;
        settings.adapt_options.mass_matrix_options.eigval_cutoff = eigval_cutoff;
        run_sampler(stan_model, settings, num_chains, num_draws, num_warmup, num_cores, refresh)?
    } else {
        let mut settings = DiagGradNutsSettings::default();
        configure_settings!(settings);
        run_sampler(stan_model, settings, num_chains, num_draws, num_warmup, num_cores, refresh)?
    };

    // Build post-warmup draws matrix (column-major for R)
    let draws_robj = build_draws_matrix(&results, ndim, num_tune, n_draws_per_chain, &param_names)?;

    // Extract post-warmup diagnostics
    let diagnostics = extract_diagnostics(
        &results,
        num_tune,
        n_draws_per_chain,
        store_divergences,
        store_mass_matrix,
    );

    // Optionally extract warmup draws and diagnostics
    let warmup_draws_robj: Robj = if save_warmup {
        build_draws_matrix(&results, ndim, 0, num_tune, &param_names)?
    } else {
        ().into_robj()
    };
    let warmup_diagnostics_robj: Robj = if save_warmup {
        extract_diagnostics(&results, 0, num_tune, store_divergences, store_mass_matrix).into_robj()
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
        expand_errors = n_expand_errors
    ))
}

/// Build a draws matrix from Arrow traces.
/// `skip` is the number of initial rows to skip, `n_draws` is how many to extract.
fn build_draws_matrix(
    results: &[ArrowTrace],
    ndim: usize,
    skip: usize,
    n_draws: usize,
    param_names: &[String],
) -> Result<Robj> {
    let n_chains = results.len();
    let total_rows = n_draws * n_chains;
    let mut data = vec![0.0f64; total_rows * ndim];

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

            for param in 0..ndim {
                let row_idx = chain_idx * n_draws + draw;
                data[row_idx + param * total_rows] = values.value(param);
            }
        }
    }

    let matrix = RMatrix::new_matrix(total_rows, ndim, |r, c| data[r + c * total_rows]);
    let mut robj = matrix.into_robj();
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

/// Extract diagnostic statistics from sample_stats RecordBatches.
/// Returns a named R list with one vector per diagnostic field.
fn extract_diagnostics(
    results: &[ArrowTrace],
    skip: usize,
    n_draws: usize,
    include_divergence_detail: bool,
    include_mass_matrix: bool,
) -> List {
    let n_chains = results.len();
    let total = n_draws * n_chains;

    let mut diverging = vec![false; total];
    let mut depth = vec![0i32; total];
    let mut energy = vec![0.0f64; total];
    let mut energy_error = vec![0.0f64; total];
    let mut logp = vec![0.0f64; total];
    let mut n_steps = vec![0i32; total];
    let mut step_size_bar = vec![0.0f64; total];
    let mut mean_tree_accept = vec![0.0f64; total];

    // Pre-allocate optional vectors with R NULLs (non-divergent draws stay NULL)
    fn null_robj_vec(n: usize, include: bool) -> Vec<Robj> {
        if include { (0..n).map(|_| ().into_robj()).collect() } else { vec![] }
    }

    let mut div_start = null_robj_vec(total, include_divergence_detail);
    let mut div_end = null_robj_vec(total, include_divergence_detail);
    let mut div_momentum = null_robj_vec(total, include_divergence_detail);
    let mut div_start_grad = null_robj_vec(total, include_divergence_detail);
    let mut mass_matrix = null_robj_vec(total, include_mass_matrix);

    for (chain_idx, trace) in results.iter().enumerate() {
        let stats = &trace.sample_stats;
        let offset = chain_idx * n_draws;

        extract_bool(stats, "diverging", &mut diverging, offset, skip, n_draws);
        extract_u64_as_i32(stats, "depth", &mut depth, offset, skip, n_draws);
        extract_f64(stats, "energy", &mut energy, offset, skip, n_draws);
        extract_f64(stats, "energy_error", &mut energy_error, offset, skip, n_draws);
        extract_f64(stats, "logp", &mut logp, offset, skip, n_draws);
        extract_u64_as_i32(stats, "n_steps", &mut n_steps, offset, skip, n_draws);
        extract_f64(stats, "step_size_bar", &mut step_size_bar, offset, skip, n_draws);
        extract_f64(stats, "mean_tree_accept", &mut mean_tree_accept, offset, skip, n_draws);

        if include_divergence_detail {
            extract_large_list_f64(stats, "divergence_start", &mut div_start, offset, skip, n_draws);
            extract_large_list_f64(stats, "divergence_end", &mut div_end, offset, skip, n_draws);
            extract_large_list_f64(stats, "divergence_momentum", &mut div_momentum, offset, skip, n_draws);
            extract_large_list_f64(stats, "divergence_start_gradient", &mut div_start_grad, offset, skip, n_draws);
        }

        if include_mass_matrix {
            extract_large_list_f64(stats, "mass_matrix_inv", &mut mass_matrix, offset, skip, n_draws);
        }
    }

    fn optional_list(values: Vec<Robj>, include: bool) -> Robj {
        if include { List::from_values(values).into_robj() } else { ().into_robj() }
    }

    let div_start_robj = optional_list(div_start, include_divergence_detail);
    let div_end_robj = optional_list(div_end, include_divergence_detail);
    let div_momentum_robj = optional_list(div_momentum, include_divergence_detail);
    let div_start_grad_robj = optional_list(div_start_grad, include_divergence_detail);
    let mass_matrix_robj = optional_list(mass_matrix, include_mass_matrix);

    list!(
        diverging = diverging,
        depth = depth,
        energy = energy,
        energy_error = energy_error,
        logp = logp,
        n_steps = n_steps,
        step_size_bar = step_size_bar,
        mean_tree_accept = mean_tree_accept,
        divergence_start = div_start_robj,
        divergence_end = div_end_robj,
        divergence_momentum = div_momentum_robj,
        divergence_start_gradient = div_start_grad_robj,
        mass_matrix_inv = mass_matrix_robj
    )
}

fn extract_bool(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    out: &mut [bool],
    offset: usize,
    skip: usize,
    n: usize,
) {
    if let Some(col) = batch.column_by_name(name) {
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::BooleanArray>() {
            for i in 0..n {
                out[offset + i] = arr.value(skip + i);
            }
        }
    }
}

fn extract_u64_as_i32(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    out: &mut [i32],
    offset: usize,
    skip: usize,
    n: usize,
) {
    if let Some(col) = batch.column_by_name(name) {
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::UInt64Array>() {
            for i in 0..n {
                out[offset + i] = arr.value(skip + i) as i32;
            }
        }
    }
}

fn extract_f64(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    out: &mut [f64],
    offset: usize,
    skip: usize,
    n: usize,
) {
    if let Some(col) = batch.column_by_name(name) {
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
            for i in 0..n {
                out[offset + i] = arr.value(skip + i);
            }
        }
    }
}

/// Extract a LargeList(Float64) column into a Vec<Robj>.
/// Each element is either NULL (for null/empty list entries) or a numeric vector.
fn extract_large_list_f64(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
    out: &mut [Robj],
    offset: usize,
    skip: usize,
    n: usize,
) {
    if let Some(col) = batch.column_by_name(name) {
        if let Some(list_arr) = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
        {
            for i in 0..n {
                let row = skip + i;
                if list_arr.is_null(row) {
                    out[offset + i] = ().into_robj();
                } else {
                    let inner = list_arr.value(row);
                    if let Some(values) = inner
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                    {
                        if values.is_empty() {
                            out[offset + i] = ().into_robj();
                        } else {
                            let vec: Vec<f64> =
                                (0..values.len()).map(|j| values.value(j)).collect();
                            out[offset + i] = vec.into_robj();
                        }
                    }
                }
            }
        }
    }
}

/// Open a BridgeStan model and return an `ExternalPtr<BSHandle>` that caches
/// parameter-name metadata. The handle may be used by any of the `bs_*`
/// accessor functions without re-opening the shared library.
/// @keywords internal
#[extendr]
fn bs_open(lib_path: &str, data_json: &str, seed: i32) -> Result<ExternalPtr<model::BSHandle>> {
    let handle = model::BSHandle::open(std::path::Path::new(lib_path), data_json, seed as u32)
        .map_err(r_err)?;
    Ok(ExternalPtr::new(handle))
}

/// Block-level parameter names (no transformed parameters / generated
/// quantities), dot-indexed. Length equals `bs_ndim_block()`.
/// @keywords internal
#[extendr]
fn bs_block_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.block_names.clone()
}

/// Full constrained parameter names (block + transformed parameters +
/// generated quantities), dot-indexed.
/// @keywords internal
#[extendr]
fn bs_full_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.full_names.clone()
}

/// Unconstrained parameter names, dot-indexed. Length equals `bs_ndim_unc()`.
/// @keywords internal
#[extendr]
fn bs_unc_names(handle: ExternalPtr<model::BSHandle>) -> Vec<String> {
    handle.unc_names.clone()
}

/// Number of unconstrained parameters.
/// @keywords internal
#[extendr]
fn bs_ndim_unc(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_unc as i32
}

/// Number of block-level constrained parameters (no TP, no GQ).
/// @keywords internal
#[extendr]
fn bs_ndim_block(handle: ExternalPtr<model::BSHandle>) -> i32 {
    handle.ndim_block as i32
}

/// Map a flat block-level constrained vector (length `bs_ndim_block()`,
/// BridgeStan column-major / last-index-major order) to the unconstrained
/// space. No JSON parsing.
/// @keywords internal
#[extendr]
fn bs_param_unconstrain(
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
/// @keywords internal
#[extendr]
fn bs_param_constrain(
    handle: ExternalPtr<model::BSHandle>,
    theta_unc: Vec<f64>,
    seed: i32,
) -> Result<Vec<f64>> {
    if theta_unc.len() != handle.ndim_unc {
        return Err(Error::Other(format!(
            "theta_unc length {} does not match unconstrained parameter count {}",
            theta_unc.len(),
            handle.ndim_unc
        )));
    }
    let mut out = vec![0.0f64; handle.ndim_full];
    let mut rng = handle.model.new_rng(seed as u32).map_err(r_err)?;
    handle
        .model
        .param_constrain(&theta_unc, true, true, &mut out, Some(&mut rng))
        .map_err(r_err)?;
    Ok(out)
}

extendr_module! {
    mod nutpieR;
    fn sample_normal;
    fn compile_stan_model;
    fn sample_stan;
    fn bs_open;
    fn bs_block_names;
    fn bs_full_names;
    fn bs_unc_names;
    fn bs_ndim_unc;
    fn bs_ndim_block;
    fn bs_param_unconstrain;
    fn bs_param_constrain;
}
