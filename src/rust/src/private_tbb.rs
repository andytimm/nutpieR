//! macOS-only post-link isolation of BridgeStan's TBB libraries.
//! Models keep their original .so untouched; all models using identical TBB bytes
//! point at one shared, content-addressed bundle. Never publish a partial image.
#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::ffi::CStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static PUBLISH: Mutex<()> = Mutex::new(());
const LIBRARIES: [&str; 3] = [
    "libtbb.dylib",
    "libtbbmalloc.dylib",
    "libtbbmalloc_proxy.dylib",
];
const FORMAT: &[u8] = b"nutpieR-private-tbb-v3\0";
fn private_name(name: &str, version: &str) -> String {
    let kind = match name {
        "libtbb.dylib" => "t",
        "libtbbmalloc.dylib" => "m",
        _ => "p",
    };
    format!("n_{kind}{}.dylib", &version[..16])
}
fn private_dep(name: &str, version: &str) -> String {
    format!("@rpath/{}", private_name(name, version))
}

fn run(program: &str, args: &[&std::ffi::OsStr]) -> std::result::Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("{program}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
fn strarg(s: &str) -> &std::ffi::OsStr {
    std::ffi::OsStr::new(s)
}
fn dependencies(path: &Path) -> std::result::Result<Vec<String>, String> {
    let text = run("otool", &[strarg("-L"), path.as_os_str()])?;
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| {
            line.trim()
                .split_once(" (compatibility version ")
                .map(|(name, _)| name.to_owned())
        })
        .collect())
}
fn rpaths(path: &Path) -> std::result::Result<Vec<String>, String> {
    let text = run("otool", &[strarg("-l"), path.as_os_str()])?;
    let mut paths = Vec::new();
    let mut rpath = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("cmd ") {
            rpath = line == "cmd LC_RPATH";
        }
        if rpath && line.starts_with("path ") {
            if let Some((path, _)) = line[5..].rsplit_once(" (offset ") {
                paths.push(path.to_owned());
            }
            rpath = false;
        }
    }
    Ok(paths)
}
fn sign(path: &Path) -> std::result::Result<(), String> {
    run(
        "codesign",
        &[
            strarg("--force"),
            strarg("--sign"),
            strarg("-"),
            path.as_os_str(),
        ],
    )?;
    Ok(())
}
fn change(path: &Path, old: &str, new: &Path) -> std::result::Result<(), String> {
    run(
        "install_name_tool",
        &[
            strarg("-change"),
            strarg(old),
            new.as_os_str(),
            path.as_os_str(),
        ],
    )?;
    Ok(())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn temp_near(path: &Path) -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.with_file_name(format!(
        ".nutpieR-{}-{n}-{}",
        std::process::id(),
        path.file_name().unwrap_or_default().to_string_lossy()
    ))
}
fn ensure_adjacent_link(link: &Path, target: &Path) -> std::result::Result<(), String> {
    use std::os::unix::fs::symlink;
    let canonical_target =
        fs::canonicalize(target).map_err(|e| format!("{}: {e}", target.display()))?;
    if let Ok(existing) = fs::canonicalize(link) {
        if existing == canonical_target {
            return Ok(());
        }
        return Err(format!(
            "private TBB link {} points to a different library; remove it and recompile",
            link.display()
        ));
    }
    let staging = temp_near(link);
    symlink(&canonical_target, &staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    if let Err(e) = fs::rename(&staging, link) {
        let _ = fs::remove_file(&staging);
        return Err(format!("publish private TBB link {}: {e}", link.display()));
    }
    if fs::canonicalize(link).ok().as_ref() != Some(&canonical_target) {
        return Err(format!(
            "private TBB link {} changed concurrently",
            link.display()
        ));
    }
    Ok(())
}

fn tbb_name(dep: &str) -> Option<&'static str> {
    LIBRARIES
        .iter()
        .copied()
        .find(|name| dep.ends_with(&format!("/{name}")))
}
fn loaded_proxy_conflict(target: &Path) -> std::result::Result<(), String> {
    // dyld interns dylibs by install name. A second independently loaded malloc
    // proxy can register another global malloc zone; refuse before loading it.
    #[allow(deprecated)]
    unsafe {
        for i in 0..libc::_dyld_image_count() {
            let p = libc::_dyld_get_image_name(i);
            if p.is_null() {
                continue;
            }
            let name = CStr::from_ptr(p).to_string_lossy();
            if (name.contains("libtbbmalloc_proxy.dylib") || name.contains("/n_p"))
                && fs::canonicalize(Path::new(name.as_ref()))
                    .unwrap_or_else(|_| PathBuf::from(name.as_ref()))
                    != fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf())
            {
                return Err(format!("another TBB malloc proxy is already loaded ({}); restart R before loading this model", name));
            }
        }
    }
    Ok(())
}

pub(crate) fn package(model: &Path) -> std::result::Result<PathBuf, String> {
    let _guard = PUBLISH
        .lock()
        .map_err(|_| "private TBB packaging lock poisoned".to_string())?;
    if !model.is_file() {
        return Err(format!("model library not found: {}", model.display()));
    }
    // Never reinterpret an already-private artifact as source (its dependency
    // paths and possibly its original Stan rpath are no longer authoritative).
    if model
        .file_stem()
        .and_then(|n| n.to_str())
        .is_some_and(|s| s.contains("_nutpieR_private_"))
    {
        let deps = dependencies(model)?;
        if deps.iter().any(|d| d.starts_with("@rpath/libtbb")) {
            return Err(format!(
                "private TBB model still has public dependencies: {}; recompile the model",
                model.display()
            ));
        }
        let private_deps: Vec<_> = deps
            .iter()
            .filter_map(|dep| dep.strip_prefix("@rpath/n_"))
            .filter(|suffix| {
                suffix.ends_with(".dylib")
                    && matches!(suffix.as_bytes().first(), Some(b't' | b'm' | b'p'))
            })
            .collect();
        if !private_deps.is_empty() {
            // v2 artifacts carry the absolute shared-bundle rpath; v3 keeps
            // short @loader_path and symlinks to the shared bundle beside the model.
            let base = rpaths(model)?
                .into_iter()
                .find(|p| p == "@loader_path" || p.contains("/nutpieR/tbb/"))
                .ok_or("private TBB model missing private rpath; recompile the model")?;
            let directory = if base == "@loader_path" {
                model
                    .parent()
                    .ok_or("private TBB model has no parent directory")?
            } else {
                Path::new(&base)
            };
            for suffix in private_deps {
                let file = directory.join(format!("n_{suffix}"));
                if !file.is_file() {
                    // A cache cleaner may remove the shared bundle while a
                    // serialized model path survives. Recreate it from the raw
                    // sibling rather than leaving a permanent broken cache hit.
                    if let Some((stem, _)) = model
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.rsplit_once("_nutpieR_private_"))
                    {
                        let raw = model.with_file_name(format!("{stem}.so"));
                        if raw.is_file() {
                            drop(_guard);
                            return package(&raw);
                        }
                    }
                    return Err(format!(
                        "private TBB bundle was removed ({}); recompile with cache = FALSE or run nutpie_clear_cache() first",
                        file.display()
                    ));
                }
                if suffix.starts_with('p') {
                    loaded_proxy_conflict(&file)?;
                }
            }
        }
        return Ok(model.to_path_buf());
    }
    let deps = dependencies(model)?;
    // Do not silently skip isolation if a future Stan build changes TBB's
    // dylib name. An unknown TBB dependency could recreate the dyld collision.
    if let Some(dep) = deps.iter().find(|dep| {
        dep.rsplit('/')
            .next()
            .is_some_and(|name| name.starts_with("libtbb") && name.ends_with(".dylib"))
            && tbb_name(dep).is_none()
    }) {
        return Err(format!(
            "unsupported Stan TBB dependency {dep} in {}; update nutpieR's private TBB linker",
            model.display()
        ));
    }
    let tbb_deps: Vec<_> = deps
        .iter()
        .filter_map(|d| tbb_name(d).map(|name| (d, name)))
        .collect();
    if tbb_deps.is_empty() {
        return Ok(model.to_path_buf());
    }
    let bases = rpaths(model)?;
    // Follow the actual model load commands and TBB-to-TBB dependencies. A
    // TBB_LIBRARIES=tbb build has no allocator proxy; it must not require or
    // load one, even when the user explicitly disabled the proxy source patch.
    let mut selected: Vec<&str> = tbb_deps.iter().map(|(_, name)| *name).collect();
    selected.sort_unstable();
    selected.dedup();
    let mut sources = Vec::new();
    let mut i = 0;
    while i < selected.len() {
        let name = selected[i];
        // Resolve @rpath using this model's search paths rather than today's
        // BridgeStan installation, which may differ from the original build.
        let src = bases
            .iter()
            .filter_map(|base| {
                let base = base.replace("@loader_path", &model.parent()?.to_string_lossy());
                let path = Path::new(&base).join(name);
                path.is_file().then_some(path)
            })
            .next()
            .ok_or_else(|| {
                format!(
                    "cannot find required {name} in model rpaths ({}); recompile the model",
                    model.display()
                )
            })?;
        for dependency in dependencies(&src)? {
            if let Some(required) = tbb_name(&dependency) {
                if !selected.contains(&required) {
                    selected.push(required);
                }
            }
        }
        sources.push(src);
        i += 1;
    }
    let mut hash = Sha256::new();
    hash.update(FORMAT);
    let mut contents = Vec::new();
    for src in &sources {
        let bytes = fs::read(src).map_err(|e| format!("{}: {e}", src.display()))?;
        hash.update(src.file_name().unwrap().as_encoded_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
        contents.push(bytes);
    }
    let version = hex(&hash.finalize())[..24].to_string();
    if let Some(index) = selected.iter().position(|name| *name == LIBRARIES[2]) {
        if !contents[index]
            .windows(super::TBB_PATCH_MARKER.len())
            .any(|b| b == super::TBB_PATCH_MARKER.as_bytes())
        {
            return Err(format!("unsafe TBB malloc proxy at {}; recompile with cache = FALSE after patching BridgeStan's TBB source (unset NUTPIER_NO_TBB_PROXY_PATCH)", sources[index].display()));
        }
    }
    let home = dirs::home_dir().ok_or("cannot locate home directory for private TBB bundle")?;
    let root = home.join(".cache/nutpieR/tbb");
    let bundle = root.join(&version);
    if selected.contains(&LIBRARIES[2]) {
        loaded_proxy_conflict(&bundle.join(private_name(LIBRARIES[2], &version)))?;
    }
    fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
    if !bundle.is_dir() {
        let staging = temp_near(&bundle);
        let result = (|| {
            fs::create_dir(&staging).map_err(|e| e.to_string())?;
            for (index, (name, bytes)) in selected.iter().zip(contents.iter()).enumerate() {
                let file = staging.join(private_name(name, &version));
                fs::write(&file, bytes).map_err(|e| e.to_string())?;
                // Preserve executable mode when copying signed Mach-O images.
                fs::set_permissions(
                    &file,
                    fs::metadata(&sources[index])
                        .map_err(|e| e.to_string())?
                        .permissions(),
                )
                .map_err(|e| e.to_string())?;
                let id = private_dep(name, &version);
                run(
                    "install_name_tool",
                    &[strarg("-id"), strarg(&id), file.as_os_str()],
                )?;
                for dep in dependencies(&file)? {
                    if let Some(target) = tbb_name(&dep) {
                        change(&file, &dep, Path::new(&private_dep(target, &version)))?;
                    }
                }
                sign(&file)?;
            }
            Ok::<_, String>(())
        })();
        if let Err(e) = result {
            let _ = fs::remove_dir_all(&staging);
            return Err(e);
        }
        if let Err(e) = fs::rename(&staging, &bundle) {
            // A different process may have published the same content first.
            let _ = fs::remove_dir_all(&staging);
            if !bundle.is_dir() {
                return Err(format!("publish {}: {e}", bundle.display()));
            }
        }
    }
    let model_dir = model
        .parent()
        .ok_or("model library has no parent directory")?;
    for name in &selected {
        let private_file = private_name(name, &version);
        let target = bundle.join(&private_file);
        if !target.is_file() {
            return Err(format!(
                "incomplete private TBB bundle: {}",
                bundle.display()
            ));
        }
        // Keep the model's rpath short regardless of the user's HOME length.
        // The versioned install IDs and shared target still deduplicate TBB.
        ensure_adjacent_link(&model_dir.join(private_file), &target)?;
    }
    let model_bytes = fs::read(model).map_err(|e| e.to_string())?;
    let mut model_hash = Sha256::new();
    model_hash.update(FORMAT);
    model_hash.update(&model_bytes);
    model_hash.update(bundle.as_os_str().as_encoded_bytes());
    let model_version = hex(&model_hash.finalize())[..16].to_owned();
    let stem = model.file_stem().unwrap().to_string_lossy();
    let derived = model.with_file_name(format!("{stem}_nutpieR_private_{model_version}.so"));
    if !derived.exists() {
        let staging = temp_near(&derived);
        let result = (|| {
            fs::write(&staging, &model_bytes).map_err(|e| e.to_string())?;
            fs::set_permissions(
                &staging,
                fs::metadata(model)
                    .map_err(|e| e.to_string())?
                    .permissions(),
            )
            .map_err(|e| e.to_string())?;
            // Drop the original Stan rpath so public TBB is never in dyld's
            // search path. Replacing (rather than adding) also frees Mach-O
            // header space for renamed dependency load commands.
            let old_rpath = bases
                .iter()
                .find(|base| {
                    let expanded = base.replace(
                        "@loader_path",
                        &model.parent().unwrap_or(Path::new("")).to_string_lossy(),
                    );
                    selected
                        .iter()
                        .all(|name| Path::new(&expanded).join(name).is_file())
                })
                .ok_or("cannot locate original Stan TBB rpath")?;
            run(
                "install_name_tool",
                &[
                    strarg("-rpath"),
                    strarg(old_rpath),
                    strarg("@loader_path"),
                    staging.as_os_str(),
                ],
            )?;
            for (dep, name) in &tbb_deps {
                change(&staging, dep, Path::new(&private_dep(name, &version)))?;
            }
            // Distinguish the derived image from its still-valid raw sibling.
            // Changing a long original absolute ID to a short private ID also
            // frees header space on model builds without -headerpad.
            if let Some(id) = dependencies(&staging)?.first() {
                if id == &model.to_string_lossy() {
                    let new_id = format!("@rpath/n_model_{model_version}.so");
                    run(
                        "install_name_tool",
                        &[strarg("-id"), strarg(&new_id), staging.as_os_str()],
                    )?;
                }
            }
            sign(&staging)?;
            Ok::<_, String>(())
        })();
        if let Err(e) = result {
            let _ = fs::remove_file(&staging);
            return Err(e);
        }
        if let Err(e) = fs::rename(&staging, &derived) {
            let _ = fs::remove_file(&staging);
            if !derived.is_file() {
                return Err(format!("publish {}: {e}", derived.display()));
            }
        }
    }
    Ok(derived)
}
