// Reproducer for GitHub #36: TBB tbbmalloc_proxy segfault
// ~200K parameters to trigger the allocator contention pattern
parameters {
  vector[200000] theta;
}
model {
  theta ~ normal(0, 1);
}
