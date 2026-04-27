# nolint start

#' Return the linked BridgeStan crate version, e.g. "2.7.0". Used by the
#' inline-code compile cache key so a BridgeStan version bump invalidates
#' cached entries automatically.
#' @noRd
bridgestan_version <- function() .Call(wrap__bridgestan_version)

#' Compile a Stan model to a shared library using BridgeStan.
#' Downloads BridgeStan sources if needed (first call is slow).
#' @param stan_file Path to the .stan file.
#' @param stanc_args Character vector of extra arguments for stanc compiler.
#' @param compile_args Character vector of extra arguments for make.
#' @return Path to the compiled shared library.
#' @noRd
compile_stan_model <- function(stan_file, stanc_args, compile_args) .Call(wrap__compile_stan_model, stan_file, stanc_args, compile_args)

#' @param handle An `ExternalPtr<BSHandle>` from `bs_open()`.
#' @param num_draws Number of draws per chain after warmup.
#' @param num_warmup Number of warmup (tuning) draws per chain.
#' @param num_chains Number of parallel chains.
#' @param seed Random seed.
#' @param max_treedepth Maximum tree depth for NUTS.
#' @param target_accept Target acceptance probability for step size adaptation.
#' @param refresh Print progress every `refresh` draws per chain (0 = no progress).
#' @param init_positions Optional list of numeric vectors (one per chain, or length 1 = broadcast).
#' @param jitter If TRUE, apply ±0.5 uniform jitter per coordinate.
#' @param save_warmup Whether to return warmup draws.
#' @param num_cores Number of CPU cores to use for parallel sampling.
#' @param store_divergences Whether to store detailed divergence information.
#' @param store_mass_matrix Whether to store the mass matrix at each draw.
#' @param store_unconstrained Whether to store the unconstrained position at each draw.
#' @param store_gradient Whether to store the gradient at each draw.
#' @param low_rank Whether to use low-rank modified mass matrix adaptation.
#' @param mass_matrix_gamma Regularisation parameter for low-rank mass matrix (default 1e-5).
#' @param eigval_cutoff Eigenvalue cutoff for low-rank mass matrix (default 2.0).
#' @param keep_indices Optional 0-indexed integer vector of constrained
#'   parameter columns to materialize. NULL means keep all. Indices are
#'   resolved against the post-flag column layout selected by
#'   `include_tp` / `include_gq`.
#' @param include_tp Whether bridgestan should compute transformed parameters
#'   when expanding each draw. When the caller has filtered them out via
#'   `pars`/`include`, set this to `FALSE` to skip the per-draw allocation
#'   and Stan-side work.
#' @param include_gq Whether bridgestan should compute generated quantities
#'   when expanding each draw. Setting this `FALSE` skips the GQ block
#'   (including any `*_rng` calls) entirely. Must imply `include_tp = TRUE`
#'   when `TRUE`, since GQ may reference TP.
#' @return A named list with draws matrix, num_warmup, num_chains, diagnostics,
#'   and optionally warmup_draws and warmup_diagnostics.
#' @noRd
sample_stan <- function(handle, num_draws, num_warmup, num_chains, seed, max_treedepth, target_accept, refresh, init_positions, jitter, save_warmup, num_cores, store_divergences, store_mass_matrix, store_unconstrained, store_gradient, low_rank, mass_matrix_gamma, eigval_cutoff, keep_indices, include_tp, include_gq) .Call(wrap__sample_stan, handle, num_draws, num_warmup, num_chains, seed, max_treedepth, target_accept, refresh, init_positions, jitter, save_warmup, num_cores, store_divergences, store_mass_matrix, store_unconstrained, store_gradient, low_rank, mass_matrix_gamma, eigval_cutoff, keep_indices, include_tp, include_gq)

#' Open a BridgeStan model and return an `ExternalPtr<BSHandle>` that caches
#' parameter-name metadata. The handle may be used by any of the `bs_*`
#' accessor functions without re-opening the shared library.
#' @noRd
bs_open <- function(lib_path, data_json, seed) .Call(wrap__bs_open, lib_path, data_json, seed)

#' Block-level parameter names (no transformed parameters / generated
#' quantities), dot-indexed. Length equals `bs_ndim_block()`.
#' @noRd
bs_block_names <- function(handle) .Call(wrap__bs_block_names, handle)

#' Block-level + transformed-parameter names (no generated quantities),
#' dot-indexed. Length equals `param_num(true, false)`. Used by R-side
#' `pars` / `include` resolution to partition names into block / TP / GQ
#' without an extra round-trip into bridgestan.
#' @noRd
bs_block_tp_names <- function(handle) .Call(wrap__bs_block_tp_names, handle)

#' Full constrained parameter names (block + transformed parameters +
#' generated quantities), dot-indexed.
#' @noRd
bs_full_names <- function(handle) .Call(wrap__bs_full_names, handle)

#' Unconstrained parameter names, dot-indexed. Length equals `bs_ndim_unc()`.
#' @noRd
bs_unc_names <- function(handle) .Call(wrap__bs_unc_names, handle)

#' Number of unconstrained parameters.
#' @noRd
bs_ndim_unc <- function(handle) .Call(wrap__bs_ndim_unc, handle)

#' Number of block-level constrained parameters (no TP, no GQ).
#' @noRd
bs_ndim_block <- function(handle) .Call(wrap__bs_ndim_block, handle)

#' Map a flat block-level constrained vector (length `bs_ndim_block()`,
#' BridgeStan column-major / last-index-major order) to the unconstrained
#' space. No JSON parsing.
#' @noRd
bs_param_unconstrain <- function(handle, theta) .Call(wrap__bs_param_unconstrain, handle, theta)

#' Map an unconstrained position to the full constrained scale (including
#' transformed parameters and generated quantities) using an already-opened
#' handle.
#' @noRd
bs_param_constrain <- function(handle, theta_unc, seed) .Call(wrap__bs_param_constrain, handle, theta_unc, seed)

#' Map an unconstrained position to the block-level constrained scale only
#' (no transformed parameters, no generated quantities). No RNG is used and
#' no GQ code runs, so this cannot fail on GQ constraint violations — the
#' right primitive for resolving partial-init random fills.
#' @noRd
bs_param_constrain_block <- function(handle, theta_unc) .Call(wrap__bs_param_constrain_block, handle, theta_unc)


# nolint end
