/* Illustrative runtime Gaussian kernel. This deliberately narrow parser handles
 * only flat numeric n/mu/sigma data, not general JSON. Use a maintained JSON
 * library for real producers. Compile with R CMD SHLIB; see run.R.
 */
#include "nutpier_kernel_v1.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
typedef struct { size_t n; double mu, sigma; } bound_t;
static int fail(char *out, size_t cap, const char *text) {
    if (cap) { size_t n = strlen(text); if (n >= cap) n = cap-1;
        memcpy(out, text, n); out[n] = 0; }
    return 2;
}
static void ws(const char **p, const char *end) {
    while (*p < end && isspace((unsigned char)**p)) ++*p;
}
/* Strict flat object parser for numeric n/mu/sigma; arbitrary key order and
 * whitespace, decimal/exponent spellings accepted. No pointer is retained.
 * Extra numeric fields are ignored. Other JSON shapes deliberately rejected.
 */
static int parse(const char *json, size_t len, bound_t *b) {
    char *copy = (char *)malloc(len+1), *tail;
    const char *p, *end; int seen = 0;
    if (!copy) return 0;
    memcpy(copy, json, len); copy[len] = 0; p = copy; end = copy+len;
    ws(&p,end); if (p == end || *p++ != '{') goto bad;
    for (;;) {
        const char *key; size_t k; double x;
        ws(&p,end); if (p == end || *p++ != '"') goto bad;
        key = p; while (p < end && *p != '"') ++p; k = (size_t)(p-key);
        if (p == end) goto bad;
        ++p; ws(&p,end); if (p == end || *p++ != ':') goto bad;
        ws(&p,end); x = strtod(p,&tail);
        if (tail == p || !isfinite(x)) goto bad;
        p = tail;
        if (k == 1 && !memcmp(key,"n",1)) {
            if ((seen&1) || x < 1 || x > 10000 || x != floor(x)) goto bad;
            b->n = (size_t)x; seen |= 1;
        } else if (k == 2 && !memcmp(key,"mu",2)) {
            if (seen&2) goto bad;
            b->mu=x; seen |= 2;
        } else if (k == 5 && !memcmp(key,"sigma",5)) {
            if ((seen&4) || x <= 0) goto bad;
            b->sigma=x; seen |= 4;
        }
        ws(&p,end); if (p == end) goto bad;
        if (*p == '}') { ++p; break; }
        if (*p++ != ',') goto bad;
    }
    ws(&p,end); if (p != end || seen != 7) goto bad;
    free(copy); return 1;
 bad: free(copy); return 0;
}
uint32_t nutpier_kernel_abi_version(void) { return 1; }
int32_t nutpier_kernel_bind(const char *json, size_t len, size_t ndim,
    const char *layout, size_t layout_len, void **out, char *err, size_t cap) {
    bound_t *b = (bound_t *)calloc(1,sizeof(bound_t));
    size_t i, used=0; char name[80];
    *out = NULL;
    if (!b) return fail(err,cap,"allocation failed");
    if (!parse(json,len,b)) { free(b); return fail(err,cap,"expected numeric n, mu, sigma"); }
    if (ndim != b->n) { free(b); return fail(err,cap,"dimension mismatch"); }
    for (i=0;i<ndim;++i) {
        int count=snprintf(name,sizeof(name),"%sx.%lu",i ? "\n" : "",(unsigned long)(i+1));
        if (count < 0 || used+(size_t)count > layout_len ||
            memcmp(layout+used,name,(size_t)count)) {
            free(b); return fail(err,cap,"ordered layout mismatch");
        }
        used += (size_t)count;
    }
    if (used != layout_len) { free(b); return fail(err,cap,"ordered layout mismatch"); }
    *out=b; return 0;
}
void nutpier_kernel_destroy(void *b) { free(b); }

/* No scratch is needed. NULL is a successful private workspace. */
int32_t nutpier_kernel_workspace(void *bound, void **out, char *err, size_t cap) {
    (void)bound; (void)err; (void)cap; *out = NULL; return 0;
}
void nutpier_kernel_workspace_destroy(void *bound, void *workspace) {
    (void)bound; (void)workspace;
}
int32_t nutpier_kernel_evaluate(void *bound, void *workspace, const double *q,
    size_t ndim, double *lp, double *gradient, char *err, size_t cap) {
    const bound_t *b = (const bound_t *)bound;
    (void)workspace;
    if (ndim != b->n) return fail(err,cap,"evaluation dimension mismatch");
    *lp = 0;
    for (size_t i=0; i<ndim; ++i) {
        double z = (q[i]-b->mu)/b->sigma;
        *lp -= 0.5*z*z;
        gradient[i] = -z/b->sigma;
    }
    /* propto=true drops normal constants because mu/sigma are data.
     * Unconstrained real parameters have an identity transform/Jacobian. */
    if (!isfinite(*lp)) { fail(err,cap,"Gaussian overflow"); return 1; }
    for (size_t i=0; i<ndim; ++i) {
        if (!isfinite(gradient[i])) { fail(err,cap,"Gaussian gradient overflow"); return 1; }
    }
    return 0;
}
