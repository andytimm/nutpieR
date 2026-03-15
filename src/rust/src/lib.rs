use arrow::array::Array;
use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, ArrowTrace, ChainProgress, CpuLogpFunc, CpuMath, DiagGradNutsSettings, HasDims,
    Model, ProgressCallback, Sampler, SamplerWaitResult,
};
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use rand::RngExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

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
    let stanc_refs: Vec<&str> = stanc_vec.iter().map(|s| s.as_str()).collect();
    let compile_vec: Vec<String> = compile_args.iter().map(|s| s.to_string()).collect();
    let compile_refs: Vec<&str> = compile_vec.iter().map(|s| s.as_str()).collect();

    let lib_path =
        bridgestan::compile_model(&bs_path, &stan_path, &stanc_refs, &compile_refs).map_err(r_err)?;
    Ok(lib_path.to_string_lossy().into_owned())
}

/// Sample from a Stan model using nuts-rs NUTS sampler.
/// @param lib_path Path to the compiled Stan shared library.
/// @param data_json JSON string with model data (empty string for no data).
/// @param num_draws Number of draws per chain after warmup.
/// @param num_warmup Number of warmup (tuning) draws per chain.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @param max_treedepth Maximum tree depth for NUTS.
/// @param target_accept Target acceptance probability for step size adaptation.
/// @param show_progress Whether to show progress bars.
/// @param init_mean Optional numeric vector of initial values in unconstrained space.
/// @param save_warmup Whether to return warmup draws.
/// @param num_cores Number of CPU cores to use for parallel sampling.
/// @param store_divergences Whether to store detailed divergence information.
/// @param store_mass_matrix Whether to store the mass matrix at each draw.
/// @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
///   and optionally warmup_draws and warmup_diagnostics.
/// @keywords internal
#[extendr]
fn sample_stan(
    lib_path: &str,
    data_json: &str,
    num_draws: i32,
    num_warmup: i32,
    num_chains: i32,
    seed: i32,
    max_treedepth: i32,
    target_accept: f64,
    show_progress: bool,
    init_mean: Robj,
    save_warmup: bool,
    num_cores: i32,
    store_divergences: bool,
    store_mass_matrix: bool,
) -> Result<List> {
    // Parse init_mean from R (NULL or numeric vector)
    let init_mean_vec: Option<Vec<f64>> = if init_mean.is_null() {
        None
    } else {
        Some(
            init_mean
                .as_real_vector()
                .ok_or_else(|| Error::Other("init_mean must be a numeric vector".into()))?,
        )
    };

    let stan_model = model::StanModel::new(
        std::path::Path::new(lib_path),
        data_json,
        seed as u32,
        init_mean_vec,
    )
    .map_err(r_err)?;

    let ndim = stan_model.num_constrained();
    let param_names: Vec<String> = stan_model.constrained_param_names().to_vec();

    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = num_warmup as u64;
    settings.num_draws = num_draws as u64;
    settings.num_chains = num_chains as usize;
    settings.seed = seed as u64;
    settings.maxdepth = max_treedepth as u64;
    settings.adapt_options.step_size_settings.target_accept = target_accept;
    settings.store_divergences = store_divergences;
    settings.adapt_options.mass_matrix_options.store_mass_matrix = store_mass_matrix;

    let callback = if show_progress {
        Some(make_progress_callback())
    } else {
        None
    };

    let sampler =
        Sampler::new(stan_model, settings, ArrowConfig::new(), num_cores as usize, callback)
            .map_err(r_err)?;

    let results = match sampler.wait_timeout(Duration::from_secs(600)) {
        SamplerWaitResult::Trace(traces) => traces,
        SamplerWaitResult::Timeout(_) => return Err(Error::Other("Sampling timed out".into())),
        SamplerWaitResult::Err(e, _) => return Err(r_err(e)),
    };

    let num_tune = settings.num_tune as usize;
    let n_draws_per_chain = num_draws as usize;

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

    Ok(list!(
        draws = draws_robj,
        num_warmup = num_warmup,
        num_chains = num_chains,
        diagnostics = diagnostics,
        warmup_draws = warmup_draws_robj,
        warmup_diagnostics = warmup_diagnostics_robj
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

    // Optional divergence detail vectors
    let mut div_start: Vec<Robj> = if include_divergence_detail {
        (0..total).map(|_| ().into_robj()).collect()
    } else {
        vec![]
    };
    let mut div_end: Vec<Robj> = if include_divergence_detail {
        (0..total).map(|_| ().into_robj()).collect()
    } else {
        vec![]
    };
    let mut div_momentum: Vec<Robj> = if include_divergence_detail {
        (0..total).map(|_| ().into_robj()).collect()
    } else {
        vec![]
    };
    let mut div_start_grad: Vec<Robj> = if include_divergence_detail {
        (0..total).map(|_| ().into_robj()).collect()
    } else {
        vec![]
    };

    // Optional mass matrix vectors
    let mut mass_matrix: Vec<Robj> = if include_mass_matrix {
        (0..total).map(|_| ().into_robj()).collect()
    } else {
        vec![]
    };

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

    let div_start_robj: Robj = if include_divergence_detail {
        List::from_values(div_start).into_robj()
    } else {
        ().into_robj()
    };
    let div_end_robj: Robj = if include_divergence_detail {
        List::from_values(div_end).into_robj()
    } else {
        ().into_robj()
    };
    let div_momentum_robj: Robj = if include_divergence_detail {
        List::from_values(div_momentum).into_robj()
    } else {
        ().into_robj()
    };
    let div_start_grad_robj: Robj = if include_divergence_detail {
        List::from_values(div_start_grad).into_robj()
    } else {
        ().into_robj()
    };
    let mass_matrix_robj: Robj = if include_mass_matrix {
        List::from_values(mass_matrix).into_robj()
    } else {
        ().into_robj()
    };

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

/// Create an indicatif-based progress callback for sampling.
fn make_progress_callback() -> ProgressCallback {
    let multibar = MultiProgress::new();
    let mut bars: Vec<TerminalBar> = vec![];
    let mut finished = false;

    let header = multibar.add(ProgressBar::new(0));
    header.set_style(ProgressStyle::default_bar().template("{msg:.bold}").unwrap());
    header.set_message(format!(
        "  {:<35}   {:<10} {:<12} {:<11} {:<12} {:<10}",
        "Progress", "Draws", "Divergences", "Step size", "Grad evals", "Elapsed"
    ));
    header.tick();

    let separator = multibar
        .add(ProgressBar::new(0))
        .with_finish(ProgressFinish::Abandon);
    separator.set_style(ProgressStyle::default_bar().template("{msg}").unwrap());
    separator.set_message(format!(" {}", "─".repeat(95)));
    separator.tick();

    let callback = move |_elapsed: Duration, progress: Box<[ChainProgress]>| {
        if bars.is_empty() {
            for chain in progress.iter() {
                bars.push(TerminalBar::new(&multibar, chain.total_draws as u64));
            }
        }

        if finished {
            return;
        }

        for (bar, chain) in bars.iter_mut().zip(progress.iter()) {
            if !bar.is_finished() && chain.finished_draws == chain.total_draws {
                bar.pb.set_position(chain.total_draws as u64);
                bar.finish();
            }
        }

        if progress
            .iter()
            .all(|chain| chain.finished_draws == chain.total_draws)
        {
            finished = true;
            header.finish();
            separator.finish();
        }

        for (bar, chain) in bars.iter_mut().zip(progress.iter()) {
            if chain.divergences > 0 {
                bar.set_mode(ChainState::Divergences);
            }
            bar.update_position(chain);
        }
    };

    ProgressCallback {
        callback: Box::new(callback),
        rate: Duration::from_millis(100),
    }
}

#[derive(PartialEq, Eq)]
enum ChainState {
    Normal,
    Divergences,
    Finished,
}

struct TerminalBar {
    pb: ProgressBar,
    last_position: u64,
    mode: ChainState,
    segment_style: String,
}

impl TerminalBar {
    fn new(mb: &MultiProgress, draws: u64) -> Self {
        let segment_style = "━━╸  ".to_string();
        let pb = mb
            .add(ProgressBar::new(draws))
            .with_finish(ProgressFinish::Abandon);
        pb.set_style(
            ProgressStyle::with_template("  {bar:35.blue}   {pos:10} {msg} {elapsed:10}")
                .unwrap()
                .progress_chars(&segment_style),
        );
        Self {
            pb,
            last_position: 0,
            mode: ChainState::Normal,
            segment_style,
        }
    }

    fn set_mode(&mut self, mode: ChainState) {
        if self.mode != mode {
            let color = match mode {
                ChainState::Normal => "blue",
                ChainState::Divergences => "red",
                ChainState::Finished => "green",
            };
            self.pb.set_style(
                ProgressStyle::with_template(&format!(
                    "  {{bar:35.{color}}}   {{pos:10}} {{msg}} {{elapsed:10}}"
                ))
                .unwrap()
                .progress_chars(&self.segment_style),
            );
            self.mode = mode;
        }
    }

    fn is_finished(&self) -> bool {
        self.pb.is_finished()
    }

    fn finish(&mut self) {
        if self.mode != ChainState::Divergences {
            self.set_mode(ChainState::Finished);
        }
        self.pb.finish();
    }

    fn update_position(&mut self, chain: &ChainProgress) {
        let position = chain.finished_draws as u64;
        let delta = position.saturating_sub(self.last_position);
        if delta > 0 && !self.is_finished() {
            self.pb.set_position(position);
            self.pb.set_message(format!(
                "{:<12} {:<11.2} {:<12}",
                chain.divergences, chain.step_size, chain.latest_num_steps
            ));
            self.last_position = position;
        }
    }
}

extendr_module! {
    mod nutpieR;
    fn sample_normal;
    fn compile_stan_model;
    fn sample_stan;
}
