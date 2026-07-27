#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("sam3_image_encoder_profile requires --features cuda")
}

#[cfg(feature = "cuda")]
mod cuda_profile {
    use std::hint::black_box;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{bail, Context, Result};
    use candle::{DType, Device, Tensor};
    use candle_transformers::models::sam3::{Config, Sam3CheckpointSource, Sam3ImageModel};
    use clap::{Parser, ValueEnum};
    use cudarc::driver::Profiler;

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
    #[command(
        about = "Capture exactly one warmed-up Candle SAM3 image encoder with CUDA profiler APIs"
    )]
    struct Args {
        /// Upstream-format SAM3 checkpoint.
        #[arg(long)]
        checkpoint: PathBuf,

        /// JPEG or PNG frame supplied to the image encoder.
        #[arg(long)]
        frame: PathBuf,

        /// Compute dtype used to load the image model.
        #[arg(long, value_enum, default_value_t = ComputeDtype::F32)]
        dtype: ComputeDtype,

        /// Number of complete encoder iterations before profiler collection starts.
        #[arg(long, default_value_t = 2)]
        warmup: usize,
    }

    fn preprocess_frame(frame_path: &Path, config: &Config, device: &Device) -> Result<Tensor> {
        let rgb = image::ImageReader::open(frame_path)?
            .decode()
            .context("failed to decode profiler frame")?
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

    fn run(args: Args) -> Result<()> {
        if !args.checkpoint.is_file() {
            bail!("checkpoint does not exist: {}", args.checkpoint.display());
        }
        if !args.frame.is_file() {
            bail!("frame does not exist: {}", args.frame.display());
        }

        let device = Device::new_cuda(0)?;
        let dtype = args.dtype.candle();
        let config = Config::default();
        let checkpoint = Sam3CheckpointSource::upstream_pth(&args.checkpoint);

        let startup = Instant::now();
        let model = Sam3ImageModel::from_checkpoint_source(&config, &checkpoint, dtype, &device)?;
        let image = preprocess_frame(&args.frame, &config, &device)?;
        device.synchronize()?;
        eprintln!(
            "model and input ready in {:.3}s; running {} warmup iteration(s)",
            startup.elapsed().as_secs_f64(),
            args.warmup
        );

        for _ in 0..args.warmup {
            let output = model.encode_image_features(&image)?;
            black_box(&output);
            device.synchronize()?;
            drop(output);
        }

        device.synchronize()?;
        eprintln!("starting CUDA profiler capture for one image encoder iteration");
        let profiler = Profiler::new().context("failed to start CUDA profiler capture")?;
        let started = Instant::now();
        let output = model.encode_image_features(&image)?;
        black_box(&output);
        device.synchronize()?;
        let captured_ms = started.elapsed().as_secs_f64() * 1000.0;
        drop(output);
        drop(profiler);

        println!(
            "{{\"status\":\"completed\",\"captured\":\"image_encoder\",\
             \"dtype\":\"{dtype:?}\",\"warmup\":{},\"captured_ms\":{captured_ms}}}",
            args.warmup
        );
        Ok(())
    }

    pub fn main() -> Result<()> {
        run(Args::parse())
    }
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    cuda_profile::main()
}
