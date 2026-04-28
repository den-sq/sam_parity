use std::fs;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::paths;

#[test]
fn full_parity_matrix_artifact_dirs_are_portable() -> Result<()> {
    let matrix_path = paths::repo_root().join("docs/video_tracker_strict_port_matrix.json");
    let matrix: Value = serde_json::from_str(&fs::read_to_string(&matrix_path).with_context(|| {
        format!(
            "failed to read video strict-port matrix from {}",
            matrix_path.display()
        )
    })?)?;
    let bundles = matrix
        .get("bundles")
        .and_then(Value::as_array)
        .context("matrix manifest is missing bundles array")?;
    for bundle in bundles {
        let name = bundle
            .get("name")
            .and_then(Value::as_str)
            .context("matrix bundle is missing name")?;
        let artifact_dir = bundle
            .get("artifact_dir")
            .and_then(Value::as_str)
            .with_context(|| format!("matrix bundle {name} is missing artifact_dir"))?;
        if artifact_dir.starts_with('/') || artifact_dir.contains("..") {
            bail!("matrix bundle {name} has non-portable artifact_dir: {artifact_dir}");
        }
        if name == "video_box_debug_default" {
            assert_eq!(artifact_dir, "reference_video_box_debug");
        }
    }
    Ok(())
}

#[test]
fn full_parity_generated_bundles_have_expected_layout_when_present() -> Result<()> {
    let root = paths::bundle_root();
    if !root.exists() {
        eprintln!(
            "skipping generated bundle layout checks because {} does not exist",
            root.display()
        );
        return Ok(());
    }

    let matrix_path = paths::repo_root().join("docs/video_tracker_strict_port_matrix.json");
    let matrix: Value = serde_json::from_str(&fs::read_to_string(&matrix_path)?)?;
    let bundles = matrix
        .get("bundles")
        .and_then(Value::as_array)
        .context("matrix manifest is missing bundles array")?;

    for bundle in bundles {
        let name = bundle
            .get("name")
            .and_then(Value::as_str)
            .context("matrix bundle is missing name")?;
        let artifact_dir = bundle
            .get("artifact_dir")
            .and_then(Value::as_str)
            .with_context(|| format!("matrix bundle {name} is missing artifact_dir"))?;
        let bundle_dir = root.join(artifact_dir);
        if !bundle_dir.exists() {
            eprintln!(
                "skipping {name} because generated bundle {} is absent",
                bundle_dir.display()
            );
            continue;
        }

        for relative in [
            "reference.json",
            "video_results.json",
            "frames",
            "masks",
            "masked_frames",
        ] {
            let path = bundle_dir.join(relative);
            if !path.exists() {
                bail!(
                    "generated bundle {artifact_dir} is missing expected artifact {relative}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn extracted_tracker_and_video_parity_sources_are_preserved() -> Result<()> {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for file_name in [
        "tracker_parity.rs",
        "video_parity.rs",
        "tracker_parity_support.rs",
        "video_parity_support.rs",
    ] {
        let path = manifest_dir.join("src").join(file_name);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if file_name.ends_with("_support.rs") {
            if text.trim().is_empty() {
                bail!("{} should contain scaffold helpers", path.display());
            }
            continue;
        }
        if !text.contains("#[test]") {
            bail!("{} no longer contains extracted test functions", path.display());
        }
    }
    Ok(())
}
