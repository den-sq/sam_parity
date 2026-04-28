#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::sam3;
use candle_transformers::models::sam3::parity_support::*;
use image::GrayImage;
use serde_json::Value;

use crate::paths;

// Shared full-parity helper scaffold lives here. The extracted tracker/video
// parity sources are included inside harness modules so we can fill this in
// incrementally while keeping the original bodies intact.
//
// The attempted live include harness showed that the remaining external-Candle
// surface is fairly focused. To activate the parked scaffold, Candle likely
// needs feature-gated `sam3-parity-support` accessors/wrappers for:
//
// - `Sam3TrackerModel`:
//   - `prepare_high_res_features`
//   - `get_tpos_enc`
//   - `use_multimask`
//   - `use_mask_as_output`
//   - `forward_sam_heads`
//   - `prepare_memory_conditioned_features`
//   - `build_memory_conditioning_prompt`
//   - `memory_transformer.forward` (or an equivalent wrapper)
//   - compute dtype / hidden prompt dtype lookup
// - `Sam3VideoPredictor`:
//   - immutable/mutable access to `video_config`
//   - immutable/mutable access to a named session
// - `Sam3VideoSession`:
//   - immutable/mutable accessors for tracked objects, frame outputs, and
//     temporal-disambiguation metadata
// - `Sam3VideoTrackerCore`:
//   - `process_frame`
//   - `clear_non_cond_mem_around_input`
//   - `postprocess_output`
// - `ObjectFrameOutput`:
//   - public score extraction helper (`score_value`)
