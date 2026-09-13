#ifndef NUTPIER_DENSITY_KERNEL_V1_H
#define NUTPIER_DENSITY_KERNEL_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define NUTPIER_DENSITY_KERNEL_EXPORT __declspec(dllexport)
#else
#define NUTPIER_DENSITY_KERNEL_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental ABI for trusted density kernels running inside R.
 *
 * Calls and buffers
 * All functions use the C calling convention; doubles are IEEE-754 binary64.
 * Inputs are borrowed for the call only. Lengths count bytes/elements, not NUL.
 * Buffers do not alias. Callbacks must not call R or retain borrowed buffers
 * for background work. No exceptions or panics may cross this ABI.
 *
 * Ownership and threads
 * Constructors clean partial allocations on failure and publish no handle.
 * Bound data is immutable and concurrently readable. Factories may run
 * concurrently. Each workspace is created, evaluated and destroyed on one
 * thread. Bound/library destruction may run on another thread.
 * Successful bound and workspace handles may be NULL; destructors must accept
 * these successful NULL handles and never throw. All handles survive failed
 * calls. Recoverable errors leave scratch reusable.
 *
 * Evaluation and errors
 * Status: 0 success, 1 domain rejection (evaluate only), 2 fatal; others fatal.
 * Success fills finite logp and every gradient element. Failed outputs are ignored.
 * Write UTF-8 error text to the host buffer, at most error_capacity bytes
 * (none if zero). NUL terminate when capacity permits. The host scans at most
 * capacity bytes, decodes lossily and supplies a fallback when empty. Long
 * messages may be truncated; the host does not require a trailing NUL.
 *
 * Reference and data
 * Match BridgeStan propto=true, jacobian=true and its exact ordered
 * unconstrained coordinates. Layout is newline-separated BridgeStan unc names
 * without a trailing newline. Bind must reject dimension/layout mismatches.
 * JSON follows nutpie_attach_density_kernel documentation: jsonlite auto_unbox=TRUE,
 * digits=NA for lists; JSON strings/files unchanged; absent data becomes {}.
 * Parse values, not byte formatting. Fixed-data density kernels must verify meaningful
 * supplied values against embedded data.
 */
NUTPIER_DENSITY_KERNEL_EXPORT uint32_t nutpier_density_kernel_abi_version(void);
NUTPIER_DENSITY_KERNEL_EXPORT int32_t nutpier_density_kernel_bind(const char *json, size_t json_len,
    size_t ndim, const char *layout, size_t layout_len, void **bound,
    char *error, size_t error_capacity);
NUTPIER_DENSITY_KERNEL_EXPORT void nutpier_density_kernel_destroy(void *bound);
NUTPIER_DENSITY_KERNEL_EXPORT int32_t nutpier_density_kernel_workspace(void *bound, void **workspace,
    char *error, size_t error_capacity);
NUTPIER_DENSITY_KERNEL_EXPORT void nutpier_density_kernel_workspace_destroy(void *bound, void *workspace);
NUTPIER_DENSITY_KERNEL_EXPORT int32_t nutpier_density_kernel_evaluate(void *bound, void *workspace,
    const double *position, size_t ndim, double *logp, double *gradient,
    char *error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
