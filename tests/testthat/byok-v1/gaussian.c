/* Test producer, not a general JSON library or a kernel generator.
 * Build with R CMD SHLIB and -I/path/to/nutpieR/include. See README.md.
 */
#include "nutpier_kernel_v1.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#ifndef FIXTURE_MODE
#define FIXTURE_MODE 0
#endif
/* 0 runtime, 1 fixed, 2 stateless, 3 offset, 4 swapped gradient,
 * 5 partial output, 6 NaN, 7 domain, 8 fatal poisoned output,
 * 9 unknown status, 10 unterminated message, 11 workspace failure,
 * 12 bind failure, 13 bad version, 14 stateful q1/q2/q1 mismatch,
 * 15 positive parameter with Jacobian, 16 omitted Jacobian.
 */
typedef struct { size_t n; double mu, sigma; } bound_t;
typedef struct { size_t calls; } workspace_t;
static int fail(char *out, size_t cap, const char *text) {
    if (cap) { size_t n = strlen(text); if (n >= cap) n = cap-1;
        memcpy(out, text, n); out[n] = 0; }
    return 2;
}
static void ws(const char **p, const char *end) {
    while (*p < end && isspace((unsigned char)**p)) ++*p;
}
/* Parse flat numeric n/mu/sigma data in any key order, with whitespace and
 * decimal/exponent spellings. Retain no pointer into the input.
 * Ignore extra numeric fields; reject other JSON shapes.
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
uint32_t nutpier_kernel_abi_version(void) { return FIXTURE_MODE == 13 ? 999 : 1; }
int32_t nutpier_kernel_bind(const char *json, size_t len, size_t ndim,
    const char *layout, size_t layout_len, void **out, char *err, size_t cap) {
    bound_t *b = (bound_t *)calloc(1,sizeof(bound_t));
    size_t i, used=0; char name[80];
    *out = NULL;
    if (!b) return fail(err,cap,"allocation failed");
    if (!parse(json,len,b)) { free(b); return fail(err,cap,"expected numeric n, mu, sigma"); }
    if (FIXTURE_MODE == 12) { free(b); return fail(err,cap,"deliberate bind failure after allocation"); }
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
    if (FIXTURE_MODE == 1 && (b->n != 2 || b->mu != 1 || b->sigma != 2)) {
        free(b); return fail(err,cap,"fixed data mismatch");
    }
    *out=b; return 0;
}
void nutpier_kernel_destroy(void *b) { free(b); }
int32_t nutpier_kernel_workspace(void *b, void **out, char *err, size_t cap) {
    (void)b; *out=NULL;
    if (FIXTURE_MODE == 11) return fail(err,cap,"deliberate workspace failure");
    if (FIXTURE_MODE == 2) return 0;
    *out=calloc(1,sizeof(workspace_t));
    return *out ? 0 : fail(err,cap,"allocation failed");
}
void nutpier_kernel_workspace_destroy(void *b, void *w) { (void)b; free(w); }
int32_t nutpier_kernel_evaluate(void *bound, void *work, const double *q,
    size_t ndim, double *lp, double *g, char *err, size_t cap) {
    const bound_t *b=(const bound_t *)bound;
    workspace_t *w=(workspace_t *)work; size_t i;
    if (w) ++w->calls;
    if (ndim != b->n) return fail(err,cap,"evaluation dimension mismatch");
    if (FIXTURE_MODE == 7) { fail(err,cap,"deliberate domain rejection"); return 1; }
    if (FIXTURE_MODE == 8) { *lp=NAN; for(i=0;i<ndim;++i) g[i]=NAN;
        return fail(err,cap,"deliberate fatal error; outputs poisoned"); }
    if (FIXTURE_MODE == 9) { fail(err,cap,"unknown status"); return 99; }
    if (FIXTURE_MODE == 10) { memset(err,'X',cap); return 2; }
    *lp=0;
    for(i=0;i<ndim;++i) {
        double x=(FIXTURE_MODE >= 15) ? exp(q[i]) : q[i];
        double z=(x-b->mu)/b->sigma;
        *lp -= 0.5*z*z;
        if (!(FIXTURE_MODE == 5 && i == ndim-1))
            g[i]=-z/b->sigma * ((FIXTURE_MODE >= 15) ? x : 1);
        if (FIXTURE_MODE == 15) { *lp += q[i]; g[i] += 1; }
    }
    if (FIXTURE_MODE == 3) *lp += 10;
    if (FIXTURE_MODE == 4 && ndim >= 2) { double tmp=g[0]; g[0]=g[1]; g[1]=tmp; }
    if (FIXTURE_MODE == 6) *lp=NAN;
    if (FIXTURE_MODE == 14 && w) *lp += (double)w->calls;
    return 0;
}
