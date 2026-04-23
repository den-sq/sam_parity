use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ParityBundleMetadata {
    #[serde(default = "default_bundle_version")]
    pub bundle_version: usize,
    #[serde(default)]
    pub image_path: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub effective_prompt: Option<String>,
    #[serde(default)]
    pub boxes_cxcywh: Vec<Vec<f32>>,
    #[serde(default)]
    pub box_labels: Vec<bool>,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub preprocess_mode: Option<String>,
    #[serde(default)]
    pub stage_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractiveReferenceStepMetadata {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub step_points_xy_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub step_point_labels: Vec<u32>,
    #[serde(default)]
    pub accumulated_points_xy_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub accumulated_point_labels: Vec<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InteractiveReferenceMetadata {
    #[serde(default = "default_bundle_version")]
    pub bundle_version: usize,
    #[serde(default)]
    pub image_path: String,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub preprocess_mode: Option<String>,
    #[serde(default)]
    pub replay_script_path: Option<String>,
    #[serde(default)]
    pub checkpoint_path: Option<String>,
    #[serde(default)]
    pub bpe_path: Option<String>,
    #[serde(default)]
    pub steps: Vec<InteractiveReferenceStepMetadata>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VideoExportMetadata {
    #[serde(default = "default_bundle_version")]
    pub bundle_version: usize,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub source_path: String,
    #[serde(default)]
    pub source_kind: String,
    #[serde(default)]
    pub session_frame_count: usize,
    #[serde(default)]
    pub exported_frame_count: usize,
    #[serde(default)]
    pub frame_stride: usize,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(default)]
    pub prompt_text: Option<String>,
    #[serde(default)]
    pub points_xy_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub point_labels: Vec<u32>,
    #[serde(default)]
    pub boxes_cxcywh_normalized: Vec<Vec<f32>>,
    #[serde(default)]
    pub box_labels: Vec<u32>,
    #[serde(default)]
    pub frames_dir: String,
    #[serde(default)]
    pub masks_dir: String,
    #[serde(default)]
    pub masked_frames_dir: String,
    #[serde(default)]
    pub results_path: String,
    #[serde(default)]
    pub debug_dir: Option<String>,
}

fn default_bundle_version() -> usize {
    1
}

