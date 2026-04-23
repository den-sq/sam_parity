use std::path::{Path, PathBuf};

pub(crate) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(crate) fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(|value| {
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            repo_root().join(path)
        }
    })
}

pub(crate) fn env_path_string(name: &str) -> Option<String> {
    env_path(name).map(|path| path.to_string_lossy().into_owned())
}

pub(crate) fn bundle_root() -> PathBuf {
    env_path("SAM3_PARITY_BUNDLE_ROOT")
        .unwrap_or_else(|| repo_root().join("tests/reference-bundles"))
}

pub(crate) fn data_root() -> PathBuf {
    env_path("SAM3_PARITY_DATA_ROOT").unwrap_or_else(|| repo_root().join("tests/data"))
}

pub(crate) fn reference_bundle_dir(name: &str) -> PathBuf {
    bundle_root().join(name)
}

pub(crate) fn resolve_bundle_arg(path_or_name: &str) -> PathBuf {
    let path = PathBuf::from(path_or_name);
    if path.exists() || path.is_absolute() || path.components().count() > 1 {
        path
    } else {
        reference_bundle_dir(path_or_name)
    }
}

pub(crate) fn resolve_data_dir(name: &str) -> PathBuf {
    data_root().join(name)
}

pub(crate) fn example_asset(relative: &str) -> String {
    repo_root()
        .join(relative)
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn resolve_metadata_path(bundle_root: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        bundle_root.join(path)
    }
}
