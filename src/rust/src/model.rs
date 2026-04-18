use nuts_rs::{CpuLogpFunc, CpuMath, CpuMathError, HasDims, Model};
use rand::RngExt;
use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Find the TBB DLL directory in the BridgeStan source tree and add it to PATH.
/// Stan models compiled with STAN_THREADS=true need tbb.dll at load time.
fn add_tbb_to_path() {
    // Try USERPROFILE first (Windows), then HOME (Unix), then dirs::home_dir()
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir());

    if let Some(home) = home {
        // Search for any bridgestan-* directory (not hardcoded version)
        let bs_base = home.join(".bridgestan");
        if let Ok(entries) = std::fs::read_dir(&bs_base) {
            for entry in entries.flatten() {
                let tbb_dir = entry
                    .path()
                    .join("stan")
                    .join("lib")
                    .join("stan_math")
                    .join("lib")
                    .join("tbb");
                if tbb_dir.exists() {
                    if let Ok(current_path) = std::env::var("PATH") {
                        let tbb_str = tbb_dir.to_string_lossy();
                        if !current_path.contains(&*tbb_str) {
                            std::env::set_var("PATH", format!("{};{}", tbb_str, current_path));
                        }
                    }
                    return;
                }
            }
        }
    }
}

/// Open a `bridgestan::Model` with the TBB PATH fixup applied.
/// Returns the model wrapped so the caller owns the StanLibrary Arc.
pub fn open_bs_model(
    lib_path: &Path,
    data_json: &str,
    seed: u32,
) -> anyhow::Result<bridgestan::Model<Arc<bridgestan::StanLibrary>>> {
    add_tbb_to_path();
    let lib = Arc::new(bridgestan::open_library(lib_path)?);
    let data = if data_json.is_empty() {
        None
    } else {
        Some(CString::new(data_json)?)
    };
    Ok(bridgestan::Model::new(lib, data.as_deref(), seed)?)
}

/// Opened BridgeStan model + cached parameter-name metadata.
///
/// Exposed to R as `ExternalPtr<BSHandle>`. The inner model is wrapped in
/// `Arc` so the sampler can hold a clone of the same Model without another
/// dlopen.
pub struct BSHandle {
    pub model: Arc<bridgestan::Model<Arc<bridgestan::StanLibrary>>>,
    pub block_names: Vec<String>,
    pub full_names: Vec<String>,
    pub unc_names: Vec<String>,
    pub ndim_unc: usize,
    pub ndim_block: usize,
    pub ndim_full: usize,
}

impl BSHandle {
    pub fn open(lib_path: &Path, data_json: &str, seed: u32) -> anyhow::Result<Self> {
        let mut model = open_bs_model(lib_path, data_json, seed)?;
        let block_names: Vec<String> = model
            .param_names(false, false)
            .split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        let full_names: Vec<String> = model
            .param_names(true, true)
            .split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        let unc_names: Vec<String> = model
            .param_unc_names()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        let ndim_unc = model.param_unc_num();
        let ndim_block = model.param_num(false, false);
        let ndim_full = model.param_num(true, true);
        Ok(BSHandle {
            model: Arc::new(model),
            block_names,
            full_names,
            unc_names,
            ndim_unc,
            ndim_block,
            ndim_full,
        })
    }
}

// --- Error type ---

#[derive(Debug, thiserror::Error)]
pub enum StanLogpError {
    #[error("BridgeStan error: {0}")]
    BridgeStan(#[from] bridgestan::BridgeStanError),
    #[error("Non-finite logp: {0}")]
    BadLogp(f64),
}

impl nuts_rs::LogpError for StanLogpError {
    fn is_recoverable(&self) -> bool {
        true // all errors become divergences, not panics
    }
}

// --- StanModel: factory for per-chain instances ---

// Per-chain id, set by Model::math() on the worker thread that owns the chain.
// nuts-rs's Sampler::start spawns one scoped task per chain; that task calls
// math() once, then init_position() (up to 500 times) on the same thread. We
// use math() as the chain-id assignment point. Rayon workers are reused across
// chains, so the value is overwritten by each new chain's first math() call —
// there is never a window where init_position runs without a current chain id
// set by the same task's earlier math() call.
thread_local! {
    static MY_CHAIN_ID: Cell<Option<usize>> = const { Cell::new(None) };
}

pub struct StanModel {
    inner: Arc<bridgestan::Model<Arc<bridgestan::StanLibrary>>>,
    ndim: usize,
    num_constrained: usize,
    constrained_param_names: Vec<String>,
    /// One position per chain. `len() == 1` means broadcast to all chains.
    /// Each vector has length `ndim` (unconstrained space).
    init_positions: Option<Vec<Vec<f64>>>,
    /// If true, apply ±0.5 uniform jitter (legacy `init_mean` behaviour).
    /// If false, chains start exactly at the provided position.
    jitter: bool,
    /// Assigns a unique chain id (0..num_chains) on each `math()` call.
    chain_counter: AtomicUsize,
    /// Count of draws where param_constrain failed (GQ filled with NaN).
    expand_errors: Arc<AtomicUsize>,
}

impl StanModel {
    pub fn new(handle: &BSHandle) -> Self {
        StanModel {
            inner: Arc::clone(&handle.model),
            ndim: handle.ndim_unc,
            num_constrained: handle.ndim_full,
            constrained_param_names: handle.full_names.clone(),
            init_positions: None,
            jitter: false,
            chain_counter: AtomicUsize::new(0),
            expand_errors: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn num_constrained(&self) -> usize {
        self.num_constrained
    }

    pub fn constrained_param_names(&self) -> &[String] {
        &self.constrained_param_names
    }

    /// Set per-chain init positions (unconstrained space).
    ///
    /// `positions` may have length 1 (broadcast) or `num_chains`. Each inner
    /// vector must have length `param_unc_num()`. If `jitter = true`,
    /// ±0.5 uniform jitter is added per-coordinate (matches legacy
    /// `init_mean` behaviour). If `jitter = false`, the position is used
    /// exactly.
    pub fn with_init_positions(
        mut self,
        positions: Option<Vec<Vec<f64>>>,
        jitter: bool,
    ) -> anyhow::Result<Self> {
        if let Some(ref ps) = positions {
            anyhow::ensure!(!ps.is_empty(), "init positions list must be non-empty");
            for (i, p) in ps.iter().enumerate() {
                anyhow::ensure!(
                    p.len() == self.ndim,
                    "init position #{} has length {}, expected {}",
                    i,
                    p.len(),
                    self.ndim
                );
            }
        }
        self.init_positions = positions;
        self.jitter = jitter;
        Ok(self)
    }

    /// Get a handle to the expand error counter (survives model consumption by sampler).
    pub fn expand_error_count_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.expand_errors)
    }
}

impl Model for StanModel {
    type Math<'model> = CpuMath<StanDensity<'model>>;

    fn math<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> anyhow::Result<Self::Math<'_>> {
        // Claim a chain id for this worker thread. See MY_CHAIN_ID doc comment.
        let chain_id = self.chain_counter.fetch_add(1, Ordering::SeqCst);
        MY_CHAIN_ID.with(|c| c.set(Some(chain_id)));

        let bs_rng = self.inner.new_rng(rng.next_u32())?;
        Ok(CpuMath::new(StanDensity {
            model: self,
            rng: bs_rng,
            expanded_buffer: vec![0f64; self.num_constrained],
            expand_errors: Arc::clone(&self.expand_errors),
        }))
    }

    fn init_position<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
        position: &mut [f64],
    ) -> anyhow::Result<()> {
        match &self.init_positions {
            Some(positions) => {
                let chain_id = MY_CHAIN_ID.with(|c| c.get()).unwrap_or(0);
                let idx = chain_id % positions.len();
                let src = &positions[idx];
                if self.jitter {
                    for (p, &m) in position.iter_mut().zip(src.iter()) {
                        *p = m + rng.random_range(-0.5..0.5);
                    }
                } else {
                    position.copy_from_slice(src);
                }
            }
            None => {
                for p in position.iter_mut() {
                    *p = rng.random_range(-2.0..2.0);
                }
            }
        }
        Ok(())
    }
}

// --- StanDensity: per-chain logp evaluator ---

pub struct StanDensity<'model> {
    model: &'model StanModel,
    rng: bridgestan::Rng<&'model bridgestan::StanLibrary>,
    expanded_buffer: Vec<f64>,
    expand_errors: Arc<AtomicUsize>,
}

impl<'model> HasDims for StanDensity<'model> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        m.insert("dim".to_string(), self.model.num_constrained as u64);
        m
    }
}

impl<'model> CpuLogpFunc for StanDensity<'model> {
    type LogpError = StanLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        self.model.ndim
    }

    fn logp(
        &mut self,
        position: &[f64],
        gradient: &mut [f64],
    ) -> std::result::Result<f64, Self::LogpError> {
        let lp = self
            .model
            .inner
            .log_density_gradient(position, true, true, gradient)?;
        if !lp.is_finite() {
            return Err(StanLogpError::BadLogp(lp));
        }
        Ok(lp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> std::result::Result<Self::ExpandedVector, CpuMathError> {
        match self.model.inner.param_constrain(
            array,
            true,
            true,
            &mut self.expanded_buffer,
            Some(&mut self.rng),
        ) {
            Ok(()) => Ok(self.expanded_buffer.clone()),
            Err(_) => {
                // param_constrain failed (e.g. generated quantities bounds violation).
                // The draw itself is valid — fill expanded vector with NaN and continue.
                self.expand_errors.fetch_add(1, Ordering::Relaxed);
                self.expanded_buffer.fill(f64::NAN);
                Ok(self.expanded_buffer.clone())
            }
        }
    }
}
