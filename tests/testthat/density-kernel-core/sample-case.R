# Run in a child R process: returned fatal status may trigger a contained sampler panic.
library(nutpieR)
args <- commandArgs(trailingOnly = TRUE)
fixtures <- normalizePath(args[[1]], mustWork = TRUE)
work <- tempfile("density-kernel-sampling-")
dir.create(work)
source <- readLines(file.path(fixtures, "gaussian.c"))
needle <- "    if (w) ++w->calls;"
stopifnot(sum(source == needle) == 1L)
source[source == needle] <- paste(needle,
  '    const char *at = getenv("NUTPIER_CORE_FAIL_AT");',
  '    if (w && at && w->calls == (size_t)strtoul(at,NULL,10))',
  '        return fail(err,cap,"injected returned fatal");', sep="\n")
writeLines(source, file.path(work,"fault.c"))
old <- setwd(work)
Sys.setenv(PKG_CPPFLAGS=paste0('-I"',system.file("include",package="nutpieR"),'"'))
status <- system2(file.path(R.home("bin"),"R"),c("CMD","SHLIB","fault.c"),
  stdout="build.log",stderr="build.log",timeout=90)
if(status != 0L) stop(paste(readLines("build.log"),collapse="\n"))
ref <- nutpie_compile_model(file.path(fixtures,"gaussian.stan"))
bound <- nutpie_attach_density_kernel(ref,normalizePath(paste0("fault",.Platform$dynlib.ext)),
                             list(n=2L,mu=1,sigma=2))
error <- function(expr, pattern) {
  e <- tryCatch({force(expr);NULL},error=identity)
  stopifnot(inherits(e,"error"),grepl(pattern,conditionMessage(e),ignore.case=TRUE))
  cat("Caught:",conditionMessage(e),"\n")
}
stopifnot(environmentIsLocked(bound), bindingIsLocked("data_json",bound))
error(nutpie_sample(bound,data=list(),progress="none"),"rebind")
for (at in c(1L,2L,19L,100L,500L)) {
  Sys.setenv(NUTPIER_CORE_FAIL_AT=at)
  error(nutpie_sample(bound,num_draws=20L,num_warmup=200L,num_chains=2L,
    cores=2L,seed=42L,progress="none"),"injected returned fatal")
  Sys.unsetenv("NUTPIER_CORE_FAIL_AT")
  fit <- nutpie_sample(bound,num_draws=20L,num_warmup=200L,num_chains=2L,
    cores=2L,seed=42L,progress="none",save_warmup=TRUE,store_gradient=TRUE)
  stopifnot(inherits(fit,"draws_array"),all(is.finite(fit)),
            identical(dim(fit)[1:2],c(20L,2L)),dim(nutpie_warmup_draws(fit))[1]==200L)
}
# Zero-gradient initialization exhausts retries in the current sampler. It must
# discard the run, including partial/empty traces returned by abort().
error(nutpie_sample(bound,num_draws=20L,num_warmup=200L,num_chains=2L,
  seed=42L,progress="none",init=list(x=c(1,1))),"run discarded")
fit <- nutpie_sample(bound,num_draws=20L,num_warmup=200L,num_chains=2L,
  cores=1L,seed=42L,progress="none",adaptation="low_rank")
stopifnot(inherits(fit,"draws_array"))
dead <- unserialize(serialize(bound,NULL))
error(nutpie_validate_density_kernel(dead),"rebind")
error(nutpie_sample(dead,num_draws=10L,num_warmup=20L,progress="none"),"rebind")
rm(dead); gc()
stopifnot(nutpie_validate_density_kernel(bound)$status=="pass")
cat("density kernel sampling regressions passed\n")
setwd(old)
unlink(work,recursive=TRUE)
