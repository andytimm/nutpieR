use nuts_rs::{CpuLogpFunc, CpuMath, CpuMathError, HasDims, Model};
use rand::RngExt;
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

pub struct StanModel {
    inner: Arc<bridgestan::Model<Arc<bridgestan::StanLibrary>>>,
    ndim: usize,
    num_constrained: usize,
    constrained_param_names: Vec<String>,
    init_mean: Option<Vec<f64>>,
    /// Count of draws where param_constrain failed (GQ filled with NaN).
    expand_errors: Arc<AtomicUsize>,
}

impl StanModel {
    pub fn new(
        lib_path: &Path,
        data_json: &str,
        seed: u32,
        init_mean: Option<Vec<f64>>,
    ) -> anyhow::Result<Self> {
        add_tbb_to_path();
        let lib = Arc::new(bridgestan::open_library(lib_path)?);
        let data = if data_json.is_empty() {
            None
        } else {
            Some(CString::new(data_json)?)
        };
        let model = bridgestan::Model::new(Arc::clone(&lib), data.as_deref(), seed)?;
        let ndim = model.param_unc_num();

        // Validate init_mean length
        if let Some(ref im) = init_mean {
            anyhow::ensure!(
                im.len() == ndim,
                "init_mean length ({}) does not match model dimension ({})",
                im.len(),
                ndim
            );
        }

        let num_constrained = model.param_num(true, true);
        let constrained_param_names: Vec<String> = model
            .param_names(true, true)
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let inner = Arc::new(model);
        Ok(StanModel {
            inner,
            ndim,
            num_constrained,
            constrained_param_names,
            init_mean,
            expand_errors: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn num_constrained(&self) -> usize {
        self.num_constrained
    }

    pub fn constrained_param_names(&self) -> &[String] {
        &self.constrained_param_names
    }

    /// Number of unconstrained parameters (for init_mean sizing).
    pub fn param_unc_num(&self) -> usize {
        self.ndim
    }

    /// Set init_mean after construction (enables scalar expansion in caller).
    pub fn with_init_mean(mut self, init_mean: Option<Vec<f64>>) -> anyhow::Result<Self> {
        if let Some(ref im) = init_mean {
            anyhow::ensure!(
                im.len() == self.ndim,
                "init_mean length ({}) does not match model dimension ({})",
                im.len(),
                self.ndim
            );
        }
        self.init_mean = init_mean;
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
        match &self.init_mean {
            Some(im) => {
                for (p, &m) in position.iter_mut().zip(im.iter()) {
                    *p = m + rng.random_range(-0.5..0.5);
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
