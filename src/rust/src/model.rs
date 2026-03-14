use nuts_rs::{CpuLogpFunc, CpuMath, CpuMathError, HasDims, Model};
use rand::RngExt;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::{Path, PathBuf};
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
    param_names: Vec<String>,
}

impl StanModel {
    pub fn new(lib_path: &Path, data_json: &str, seed: u32) -> anyhow::Result<Self> {
        add_tbb_to_path();
        // Debug: try loading with libloading directly to diagnose symbol issues
        {
            let test_lib = unsafe { libloading::Library::new(lib_path) };
            match test_lib {
                Ok(lib) => {
                    eprintln!("DEBUG: Library loaded OK with libloading");
                    let symbols = [
                        "bs_major_version", "bs_minor_version", "bs_patch_version",
                        "bs_model_construct", "bs_model_destruct", "bs_free_error_msg",
                        "bs_name", "bs_model_info", "bs_param_names", "bs_param_unc_names",
                        "bs_param_num", "bs_param_unc_num", "bs_param_constrain",
                        "bs_param_unconstrain", "bs_param_unconstrain_json",
                        "bs_log_density", "bs_log_density_gradient",
                        "bs_log_density_hessian", "bs_log_density_hessian_vector_product",
                        "bs_rng_construct", "bs_rng_destruct", "bs_set_print_callback",
                    ];
                    for sym in &symbols {
                        let mut name = sym.as_bytes().to_vec();
                        name.push(0);
                        let result: std::result::Result<libloading::Symbol<*const ()>, _> =
                            unsafe { lib.get(&name) };
                        match result {
                            Ok(_) => eprintln!("  OK: {}", sym),
                            Err(e) => eprintln!("  FAIL: {} -> {}", sym, e),
                        }
                    }
                }
                Err(e) => eprintln!("DEBUG: Library load FAILED: {}", e),
            }
        }
        let lib = Arc::new(bridgestan::open_library(lib_path)?);
        let data = if data_json.is_empty() {
            None
        } else {
            Some(CString::new(data_json)?)
        };
        let model = bridgestan::Model::new(Arc::clone(&lib), data.as_deref(), seed)?;
        let ndim = model.param_unc_num();
        let param_names = model
            .param_names(false, false)
            .split(',')
            .map(|s| s.to_string())
            .collect();
        let inner = Arc::new(model);
        Ok(StanModel {
            inner,
            ndim,
            param_names,
        })
    }

    pub fn ndim(&self) -> usize {
        self.ndim
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }
}

impl Model for StanModel {
    type Math<'model> = CpuMath<StanDensity<'model>>;

    fn math<R: rand::Rng + ?Sized>(
        &self,
        _rng: &mut R,
    ) -> anyhow::Result<Self::Math<'_>> {
        Ok(CpuMath::new(StanDensity {
            model: self,
        }))
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

// --- StanDensity: per-chain logp evaluator ---

pub struct StanDensity<'model> {
    model: &'model StanModel,
}

impl<'model> HasDims for StanDensity<'model> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        let mut m = HashMap::new();
        m.insert("dim".to_string(), self.model.ndim as u64);
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
        // MVP: return unconstrained params directly (no constrained transform)
        Ok(array.to_vec())
    }
}
