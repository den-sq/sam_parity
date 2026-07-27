use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::sam3::{
    Config, Sam3CheckpointSource, Sam3ImageModel, Sam3TrackerModel, TrackerFrameState,
    VisualBackboneOutput,
};
use clap::{Parser, ValueEnum};
use serde::Serialize;

const IMAGE_ENCODER_DESCRIPTION: &str = "image backbone and neck only";
const TRACKER_BASE_DESCRIPTION: &str =
    "SAM tracker head without prior history or new-memory encoding";
const TRACKER_WITH_HISTORY_DESCRIPTION: &str =
    "tracker head with one prior conditioning state, without new-memory encoding";
const TRACKER_FULL_DESCRIPTION: &str =
    "tracker head with one prior conditioning state and new-memory encoding";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputeDtype {
    F32,
    F16,
}

impl ComputeDtype {
    fn candle(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Partition one steady-state Candle SAM3 frame into coarse GPU stages")]
struct Args {
    /// Upstream-format SAM3 checkpoint.
    #[arg(long)]
    checkpoint: PathBuf,

    /// JPEG or PNG frame measured by the diagnostic.
    #[arg(long)]
    frame: PathBuf,

    /// Optional frame used to construct prior tracker memory (defaults to --frame).
    #[arg(long)]
    seed_frame: Option<PathBuf>,

    /// Machine-readable diagnostic output.
    #[arg(long)]
    output: PathBuf,

    /// Compute dtype used to load the image and tracker models.
    #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
    dtype: ComputeDtype,

    /// Number of untimed iterations before each stage.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Number of synchronized samples per stage.
    #[arg(long, default_value_t = 5)]
    samples: usize,

    /// Logical video length supplied to tracker memory selection.
    #[arg(long, default_value_t = 32)]
    num_frames: usize,

    /// Normalized positive seed point x coordinate.
    #[arg(long, default_value_t = 0.5)]
    point_x: f32,

    /// Normalized positive seed point y coordinate.
    #[arg(long, default_value_t = 0.5)]
    point_y: f32,

    /// Revision recorded in the output when the binary was not built with revision metadata.
    #[arg(long)]
    candle_revision: Option<String>,

    /// Run on CPU for development only. Performance comparisons require CUDA.
    #[arg(long)]
    cpu: bool,
}

#[derive(Debug, Serialize)]
struct StageStats {
    description: &'static str,
    samples_ms: Vec<f64>,
    sample_count: usize,
    mean_ms: f64,
    median_ms: f64,
    minimum_ms: f64,
    maximum_ms: f64,
}

#[derive(Debug, Serialize)]
struct Stages {
    image_encoder: StageStats,
    tracker_base: StageStats,
    tracker_with_history: StageStats,
    tracker_full: StageStats,
}

#[derive(Debug, Serialize)]
struct DerivedCosts {
    previous_memory_conditioning_increment_ms: f64,
    new_memory_encoder_increment_ms: f64,
    estimated_full_frame_ms: f64,
    estimated_full_frame_fps: f64,
}

#[derive(Debug, Serialize)]
struct DiagnosticOutput {
    schema_version: usize,
    framework: &'static str,
    status: &'static str,
    checkpoint: String,
    frame: String,
    seed_frame: String,
    candle_revision: Option<String>,
    device: String,
    dtype: String,
    warmup: usize,
    samples: usize,
    num_frames: usize,
    model_startup_seconds: f64,
    stages: Stages,
    derived: DerivedCosts,
    interpretation: BTreeMap<&'static str, &'static str>,
}

fn summarize_samples(description: &'static str, samples_ms: Vec<f64>) -> Result<StageStats> {
    if samples_ms.is_empty() {
        bail!("at least one timing sample is required");
    }
    let mut ordered = samples_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let sample_count = ordered.len();
    let median_ms = if sample_count % 2 == 0 {
        (ordered[sample_count / 2 - 1] + ordered[sample_count / 2]) / 2.0
    } else {
        ordered[sample_count / 2]
    };
    Ok(StageStats {
        description,
        mean_ms: samples_ms.iter().sum::<f64>() / sample_count as f64,
        median_ms,
        minimum_ms: ordered[0],
        maximum_ms: ordered[sample_count - 1],
        sample_count,
        samples_ms,
    })
}

fn measure_stage<T, F>(
    device: &Device,
    description: &'static str,
    warmup: usize,
    samples: usize,
    mut operation: F,
) -> Result<StageStats>
where
    F: FnMut() -> candle::Result<T>,
{
    for _ in 0..warmup {
        let value = operation()?;
        black_box(&value);
        device.synchronize()?;
        drop(value);
    }

    let mut samples_ms = Vec::with_capacity(samples);
    for _ in 0..samples {
        device.synchronize()?;
        let started = Instant::now();
        let value = operation()?;
        black_box(&value);
        device.synchronize()?;
        samples_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        drop(value);
    }
    summarize_samples(description, samples_ms)
}

fn tracker_visual_output(output: &VisualBackboneOutput) -> VisualBackboneOutput {
    match (&output.sam2_backbone_fpn, &output.sam2_pos_enc) {
        (Some(backbone_fpn), Some(vision_pos_enc)) => VisualBackboneOutput {
            backbone_fpn: backbone_fpn.clone(),
            vision_pos_enc: vision_pos_enc.clone(),
            sam2_backbone_fpn: output.sam2_backbone_fpn.clone(),
            sam2_pos_enc: output.sam2_pos_enc.clone(),
            tracker_sequences: output
                .tracker_sam2_sequences
                .clone()
                .or_else(|| output.tracker_sequences.clone()),
            tracker_sam2_sequences: output.tracker_sam2_sequences.clone(),
        },
        _ => output.clone(),
    }
}

fn preprocess_frame(frame_path: &Path, config: &Config, device: &Device) -> Result<Tensor> {
    let rgb = image::ImageReader::open(frame_path)?
        .decode()
        .context("failed to decode diagnostic frame")?
        .to_rgb8();
    let (width, height) = rgb.dimensions();
    let image = (Tensor::from_vec(
        rgb.into_raw(),
        (height as usize, width as usize, 3),
        &Device::Cpu,
    )?
    .permute((2, 0, 1))?
    .to_device(device)?
    .unsqueeze(0)?
    .to_dtype(DType::F32)?
    .upsample_bilinear2d(config.image.image_size, config.image.image_size, false)?
        / 255.0)?;
    let mean = Tensor::from_vec(config.image.image_mean.to_vec(), (1, 3, 1, 1), device)?;
    let std = Tensor::from_vec(config.image.image_std.to_vec(), (1, 3, 1, 1), device)?;
    Ok(image.broadcast_sub(&mean)?.broadcast_div(&std)?)
}

fn derived_costs(stages: &Stages) -> DerivedCosts {
    let estimated_full_frame_ms = stages.image_encoder.median_ms + stages.tracker_full.median_ms;
    DerivedCosts {
        previous_memory_conditioning_increment_ms: stages.tracker_with_history.median_ms
            - stages.tracker_base.median_ms,
        new_memory_encoder_increment_ms: stages.tracker_full.median_ms
            - stages.tracker_with_history.median_ms,
        estimated_full_frame_ms,
        estimated_full_frame_fps: 1000.0 / estimated_full_frame_ms,
    }
}

fn run(args: Args) -> Result<()> {
    if args.samples == 0 {
        bail!("--samples must be positive");
    }
    if args.num_frames < 2 {
        bail!("--num-frames must be at least two");
    }
    if !(0.0..=1.0).contains(&args.point_x) || !(0.0..=1.0).contains(&args.point_y) {
        bail!("--point-x and --point-y must be normalized to [0, 1]");
    }
    if !args.checkpoint.is_file() {
        bail!("checkpoint does not exist: {}", args.checkpoint.display());
    }
    if !args.frame.is_file() {
        bail!("frame does not exist: {}", args.frame.display());
    }
    let seed_frame = args.seed_frame.as_deref().unwrap_or(&args.frame);
    if !seed_frame.is_file() {
        bail!("seed frame does not exist: {}", seed_frame.display());
    }

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let dtype = args.dtype.candle();
    let config = Config::default();
    let checkpoint = Sam3CheckpointSource::upstream_pth(&args.checkpoint);

    let startup = Instant::now();
    let model = Sam3ImageModel::from_checkpoint_source(&config, &checkpoint, dtype, &device)?;
    let tracker = Sam3TrackerModel::from_checkpoint_source(&config, &checkpoint, dtype, &device)?;
    device.synchronize()?;
    let model_startup_seconds = startup.elapsed().as_secs_f64();

    let image = preprocess_frame(&args.frame, &config, &device)?;
    let visual = model.encode_image_features(&image)?;
    let tracker_visual = tracker_visual_output(&visual);
    let seed_image = preprocess_frame(seed_frame, &config, &device)?;
    let seed_visual = model.encode_image_features(&seed_image)?;
    let seed_tracker_visual = tracker_visual_output(&seed_visual);
    device.synchronize()?;

    let point_coords = Tensor::from_vec(
        vec![
            args.point_x * config.image.image_size as f32,
            args.point_y * config.image.image_size as f32,
        ],
        (1, 1, 2),
        &device,
    )?;
    let point_labels = Tensor::from_vec(vec![1.0f32], (1, 1), &device)?;
    let seed = tracker.track_frame(
        &seed_tracker_visual,
        0,
        args.num_frames,
        Some(&point_coords),
        Some(&point_labels),
        None,
        None,
        &BTreeMap::new(),
        true,
        false,
        false,
        true,
    )?;
    device.synchronize()?;
    let mut history = BTreeMap::<usize, TrackerFrameState>::new();
    history.insert(0, seed.state);
    let empty_history = BTreeMap::<usize, TrackerFrameState>::new();

    let image_encoder = measure_stage(
        &device,
        IMAGE_ENCODER_DESCRIPTION,
        args.warmup,
        args.samples,
        || model.encode_image_features(&image),
    )?;
    let tracker_base = measure_stage(
        &device,
        TRACKER_BASE_DESCRIPTION,
        args.warmup,
        args.samples,
        || {
            tracker.track_frame(
                &tracker_visual,
                1,
                args.num_frames,
                None,
                None,
                None,
                None,
                &empty_history,
                false,
                false,
                false,
                false,
            )
        },
    )?;
    let tracker_with_history = measure_stage(
        &device,
        TRACKER_WITH_HISTORY_DESCRIPTION,
        args.warmup,
        args.samples,
        || {
            tracker.track_frame(
                &tracker_visual,
                1,
                args.num_frames,
                None,
                None,
                None,
                None,
                &history,
                false,
                false,
                true,
                false,
            )
        },
    )?;
    let tracker_full = measure_stage(
        &device,
        TRACKER_FULL_DESCRIPTION,
        args.warmup,
        args.samples,
        || {
            tracker.track_frame(
                &tracker_visual,
                1,
                args.num_frames,
                None,
                None,
                None,
                None,
                &history,
                false,
                false,
                true,
                true,
            )
        },
    )?;
    let stages = Stages {
        image_encoder,
        tracker_base,
        tracker_with_history,
        tracker_full,
    };
    let derived = derived_costs(&stages);
    let candle_revision = args.candle_revision.or_else(|| {
        option_env!("CANDLE_SAM3_GIT_SHA")
            .map(str::to_owned)
            .filter(|value| !value.is_empty())
    });
    let interpretation = BTreeMap::from([
        (
            "tracker_base",
            "SAM tracker head without previous-frame history",
        ),
        (
            "previous_memory_conditioning_increment",
            "tracker_with_history median minus tracker_base median",
        ),
        (
            "new_memory_encoder_increment",
            "tracker_full median minus tracker_with_history median",
        ),
    ]);
    let output = DiagnosticOutput {
        schema_version: 1,
        framework: "candle",
        status: "completed",
        checkpoint: args.checkpoint.display().to_string(),
        frame: args.frame.display().to_string(),
        seed_frame: seed_frame.display().to_string(),
        candle_revision,
        device: format!("{device:?}"),
        dtype: format!("{dtype:?}"),
        warmup: args.warmup,
        samples: args.samples,
        num_frames: args.num_frames,
        model_startup_seconds,
        stages,
        derived,
        interpretation,
    };
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded = serde_json::to_string_pretty(&output)? + "\n";
    fs::write(&args.output, &encoded)?;
    print!("{encoded}");
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(median_ms: f64) -> StageStats {
        StageStats {
            description: "test",
            samples_ms: vec![median_ms],
            sample_count: 1,
            mean_ms: median_ms,
            median_ms,
            minimum_ms: median_ms,
            maximum_ms: median_ms,
        }
    }

    #[test]
    fn sample_summary_uses_statistical_median() {
        let stats = summarize_samples("test", vec![4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(stats.sample_count, 4);
        assert_eq!(stats.mean_ms, 2.5);
        assert_eq!(stats.median_ms, 2.5);
        assert_eq!(stats.minimum_ms, 1.0);
        assert_eq!(stats.maximum_ms, 4.0);
    }

    #[test]
    fn derived_costs_partition_frame() {
        let stages = Stages {
            image_encoder: stage(100.0),
            tracker_base: stage(40.0),
            tracker_with_history: stage(55.0),
            tracker_full: stage(70.0),
        };
        let derived = derived_costs(&stages);
        assert_eq!(derived.previous_memory_conditioning_increment_ms, 15.0);
        assert_eq!(derived.new_memory_encoder_increment_ms, 15.0);
        assert_eq!(derived.estimated_full_frame_ms, 170.0);
        assert!((derived.estimated_full_frame_fps - 1000.0 / 170.0).abs() < 1e-12);
    }
}
