use extendr_api::prelude::*;
use nuts_rs::{
    ArrowConfig, CpuLogpFunc, CpuMath, DiagGradNutsSettings, HasDims, Model,
    Sampler, SamplerWaitResult,
};
use rand::RngExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

mod model;

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
/// @export
#[extendr]
fn sample_normal(num_draws: i32, num_chains: i32, seed: i32) -> Robj {
    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = 300;
    settings.num_draws = num_draws as u64;
    settings.num_chains = num_chains as usize;
    settings.seed = seed as u64;

    let model = NormalModel;
    let trace_config = ArrowConfig::new();

    let sampler = Sampler::new(model, settings, trace_config, num_chains as usize, None)
        .expect("Failed to create sampler");

    let results = match sampler.wait_timeout(Duration::from_secs(300)) {
        SamplerWaitResult::Trace(traces) => traces,
        SamplerWaitResult::Timeout(_) => panic!("Sampling timed out"),
        SamplerWaitResult::Err(e, _) => panic!("Sampling failed: {}", e),
    };

    // results is Vec<ArrowTrace>, one per chain
    // Each ArrowTrace.posterior has num_tune + num_draws rows
    // The "value" column is LargeList(Float64) with shape metadata
    let n_chains = results.len();
    let num_tune = settings.num_tune as usize;
    let n_draws_per_chain = num_draws as usize;
    let total_rows = n_draws_per_chain * n_chains;

    // Build column-major matrix for R (total_rows x NDIM)
    let mut data = vec![0.0f64; total_rows * NDIM];

    for (chain_idx, trace) in results.iter().enumerate() {
        let batch = &trace.posterior;
        let col = batch
            .column_by_name("value")
            .expect("No 'value' column in posterior");

        let list_arr = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .expect("'value' column is not LargeList");

        // Skip warmup draws (first num_tune rows)
        for draw in 0..n_draws_per_chain {
            let row = num_tune + draw;
            let inner = list_arr.value(row);
            let values = inner
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .expect("inner array is not Float64");

            for param in 0..NDIM {
                let row_idx = chain_idx * n_draws_per_chain + draw;
                data[row_idx + param * total_rows] = values.value(param);
            }
        }
    }

    let matrix = RMatrix::new_matrix(total_rows, NDIM, |r, c| data[r + c * total_rows]);
    matrix.into_robj()
}

/// Compile a Stan model to a shared library using BridgeStan.
/// Downloads BridgeStan sources if needed (first call is slow).
/// @param stan_file Path to the .stan file.
/// @return Path to the compiled shared library.
/// @export
#[extendr]
fn compile_stan_model(stan_file: &str) -> String {
    let bs_path = bridgestan::download_bridgestan_src()
        .expect("Failed to download BridgeStan sources");
    let stan_path = PathBuf::from(stan_file);
    let lib_path = bridgestan::compile_model(&bs_path, &stan_path, &[], &[])
        .expect("Failed to compile Stan model");
    lib_path.to_string_lossy().into_owned()
}

/// Sample from a Stan model using nuts-rs NUTS sampler.
/// @param lib_path Path to the compiled Stan shared library.
/// @param data_json JSON string with model data (empty string for no data).
/// @param num_draws Number of draws per chain after warmup.
/// @param num_chains Number of parallel chains.
/// @param seed Random seed.
/// @return A matrix of draws (rows = draws*chains, cols = parameters).
/// @export
#[extendr]
fn sample_stan(lib_path: &str, data_json: &str, num_draws: i32, num_chains: i32, seed: i32) -> Robj {
    let stan_model = model::StanModel::new(
        std::path::Path::new(lib_path),
        data_json,
        seed as u32,
    )
    .expect("Failed to load Stan model");

    let ndim = stan_model.ndim();

    let mut settings = DiagGradNutsSettings::default();
    settings.num_tune = 300;
    settings.num_draws = num_draws as u64;
    settings.num_chains = num_chains as usize;
    settings.seed = seed as u64;

    let sampler = Sampler::new(stan_model, settings, ArrowConfig::new(), num_chains as usize, None)
        .expect("Failed to create sampler");

    let results = match sampler.wait_timeout(Duration::from_secs(600)) {
        SamplerWaitResult::Trace(traces) => traces,
        SamplerWaitResult::Timeout(_) => panic!("Sampling timed out"),
        SamplerWaitResult::Err(e, _) => panic!("Sampling failed: {}", e),
    };

    let n_chains = results.len();
    let num_tune = settings.num_tune as usize;
    let n_draws_per_chain = num_draws as usize;
    let total_rows = n_draws_per_chain * n_chains;

    // Build column-major matrix for R (total_rows x ndim)
    let mut data = vec![0.0f64; total_rows * ndim];

    for (chain_idx, trace) in results.iter().enumerate() {
        let batch = &trace.posterior;
        let col = batch
            .column_by_name("value")
            .expect("No 'value' column in posterior");

        let list_arr = col
            .as_any()
            .downcast_ref::<arrow::array::LargeListArray>()
            .expect("'value' column is not LargeList");

        for draw in 0..n_draws_per_chain {
            let row = num_tune + draw;
            let inner = list_arr.value(row);
            let values = inner
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .expect("inner array is not Float64");

            for param in 0..ndim {
                let row_idx = chain_idx * n_draws_per_chain + draw;
                data[row_idx + param * total_rows] = values.value(param);
            }
        }
    }

    let matrix = RMatrix::new_matrix(total_rows, ndim, |r, c| data[r + c * total_rows]);
    matrix.into_robj()
}

extendr_module! {
    mod nutpieR;
    fn sample_normal;
    fn compile_stan_model;
    fn sample_stan;
}
