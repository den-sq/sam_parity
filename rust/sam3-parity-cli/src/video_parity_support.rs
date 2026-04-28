mod tests {
    use super::*;
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::{Path, PathBuf},
    };

    use candle::Tensor;
    use candle_nn::VarBuilder;
    use candle_transformers::models::sam3::parity_support::ParityTemporalDisambiguationFrameMetadata;
    use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

    const VIDEO_DEBUG_MASK_THRESHOLD: f32 = 0.5;

    fn tiny_segmentation_config() -> Config {
        Config {
            image: ImageConfig {
                image_size: 56,
                image_mean: [0.5, 0.5, 0.5],
                image_std: [0.5, 0.5, 0.5],
            },
            vision: VisionConfig {
                image_size: 56,
                pretrain_image_size: 28,
                patch_size: 14,
                embed_dim: 32,
                depth: 0,
                num_heads: 4,
                mlp_ratio: 4.0,
                window_size: 2,
                global_attn_blocks: vec![],
                use_abs_pos: true,
                tile_abs_pos: true,
                use_rope: true,
                use_interp_rope: true,
                rope_theta: 10_000.0,
                rope_pt_size: 24,
                retain_cls_token: false,
                ln_pre: false,
            },
            text: TextConfig {
                d_model: 32,
                width: 64,
                heads: 4,
                layers: 1,
                context_length: 4,
                vocab_size: 64,
            },
            neck: NeckConfig {
                d_model: 32,
                scale_factors: [4.0, 2.0, 1.0, 0.5],
                scalp: 1,
                add_sam2_neck: false,
            },
            geometry: GeometryConfig {
                d_model: 32,
                num_layers: 1,
                num_heads: 1,
                dim_feedforward: 64,
                roi_size: 2,
                add_cls: true,
                add_post_encode_proj: true,
            },
            encoder: EncoderConfig {
                d_model: 32,
                num_layers: 1,
                num_feature_levels: 1,
                num_heads: 1,
                dim_feedforward: 64,
                add_pooled_text_to_image: false,
                pool_text_with_mask: true,
            },
            decoder: DecoderConfig {
                d_model: 32,
                num_layers: 1,
                num_queries: 2,
                num_heads: 1,
                dim_feedforward: 64,
                presence_token: true,
                use_text_cross_attention: true,
                box_rpb_mode: "none".to_owned(),
                box_rpb_resolution: 56,
                box_rpb_stride: 14,
                clamp_presence_logit_max: 10.0,
            },
            segmentation: SegmentationConfig {
                enabled: true,
                hidden_dim: 32,
                upsampling_stages: 1,
                aux_masks: false,
                presence_head: false,
            },
        }
    }

    fn tiny_model(device: &Device) -> Result<Sam3ImageModel> {
        Sam3ImageModel::new(
            &tiny_segmentation_config(),
            VarBuilder::zeros(DType::F32, device),
        )
    }

    fn tiny_tracker(device: &Device) -> Result<Sam3TrackerModel> {
        let config = tiny_segmentation_config();
        let tracker_config = Sam3TrackerConfig::from_sam3_config(&config);
        Sam3TrackerModel::new(&tracker_config, VarBuilder::zeros(DType::F32, device))
    }

    fn sam3_test_checkpoint_path() -> Option<PathBuf> {
        let env_path = std::env::var_os("SAM3_TEST_CHECKPOINT")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("SAM3_TEST_CHECKPOINT_DIR").map(PathBuf::from));
        let mut candidates = Vec::new();
        if let Some(path) = env_path {
            candidates.push(path);
        }
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3"));
        candidates.push(PathBuf::from("/home/dnorthover/extcode/hf_sam3/sam3.pt"));
        candidates.into_iter().find_map(|path| {
            if path.is_dir() {
                let file = path.join("sam3.pt");
                file.exists().then_some(file)
            } else if path.exists() {
                Some(path)
            } else {
                None
            }
        })
    }

    fn tracker_config_with_reference_runtime_overrides(
        bundle: Option<&str>,
    ) -> Result<Sam3TrackerConfig> {
        let mut config = Sam3TrackerConfig::from_sam3_config(&Config::default());
        let Some(bundle) = bundle else {
            return Ok(config);
        };
        let manifest = load_reference_internal_manifest(bundle)?;
        let tracker_config = manifest["tracker_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing tracker_config".to_owned())
        })?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;

        if let Some(value) = tracker_config
            .get("use_memory_selection")
            .and_then(|value| value.as_bool())
        {
            config.use_memory_selection = value;
        }
        if let Some(value) = tracker_config
            .get("memory_temporal_stride_for_eval")
            .and_then(|value| value.as_u64())
        {
            config.memory_temporal_stride_for_eval = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_obj_ptrs_in_encoder")
            .and_then(|value| value.as_u64())
        {
            config.max_obj_ptrs_in_encoder = value as usize;
        }
        if let Some(value) = tracker_config
            .get("max_cond_frames_in_attn")
            .and_then(|value| value.as_u64())
        {
            config.max_cond_frames_in_attn = value as usize;
        }
        if let Some(value) = tracker_config
            .get("keep_first_cond_frame")
            .and_then(|value| value.as_bool())
        {
            config.keep_first_cond_frame = value;
        }
        if let Some(value) = tracker_config
            .get("trim_past_non_cond_mem_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.trim_past_non_cond_mem_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("offload_output_to_cpu_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.offload_output_to_cpu_for_eval = value;
        }
        if let Some(value) = tracker_config
            .get("forward_backbone_per_frame_for_eval")
            .and_then(|value| value.as_bool())
        {
            config.predictor.forward_backbone_per_frame_for_eval = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_around_input")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_around_input = value;
        }
        if let Some(value) = predictor_config
            .get("clear_non_cond_mem_for_multi_obj")
            .and_then(|value| value.as_bool())
        {
            config.predictor.clear_non_cond_mem_for_multi_obj = value;
        }
        if let Some(value) = predictor_config
            .get("always_start_from_first_ann_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.always_start_from_first_ann_frame = value;
        }
        if let Some(value) = predictor_config
            .get("iter_use_prev_mask_pred")
            .and_then(|value| value.as_bool())
        {
            config.predictor.iter_use_prev_mask_pred = value;
        }
        if let Some(value) = predictor_config
            .get("add_all_frames_to_correct_as_cond")
            .and_then(|value| value.as_bool())
        {
            config.predictor.add_all_frames_to_correct_as_cond = value;
        }
        if let Some(value) = predictor_config
            .get("use_prev_mem_frame")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_prev_mem_frame = value;
        }
        if let Some(value) = predictor_config
            .get("use_stateless_refinement")
            .and_then(|value| value.as_bool())
        {
            config.predictor.use_stateless_refinement = value;
        }
        if let Some(value) = predictor_config
            .get("refinement_detector_cond_frame_removal_window")
            .and_then(|value| value.as_u64())
        {
            config
                .predictor
                .refinement_detector_cond_frame_removal_window = value as usize;
        }
        Ok(config)
    }

    fn load_runtime_models_from_checkpoint(
        bundle: Option<&str>,
    ) -> Result<Option<(Sam3ImageModel, Sam3TrackerModel, Device)>> {
        let Some(checkpoint_path) = sam3_test_checkpoint_path() else {
            return Ok(None);
        };
        let device = Device::Cpu;
        let config = Config::default();
        let checkpoint = sam3::Sam3CheckpointSource::upstream_pth(checkpoint_path);
        let model =
            Sam3ImageModel::from_checkpoint_source(&config, &checkpoint, DType::F32, &device)?;
        let tracker_config = tracker_config_with_reference_runtime_overrides(bundle)?;
        let tracker = Sam3TrackerModel::new(
            &tracker_config,
            checkpoint.load_tracker_var_builder(DType::F32, &device)?,
        )?;
        Ok(Some((model, tracker, device)))
    }

    fn sam3_test_tokenizer_path() -> Option<PathBuf> {
        let checkpoint_path = sam3_test_checkpoint_path()?;
        let tokenizer = checkpoint_path.parent()?.join("tokenizer.json");
        tokenizer.exists().then_some(tokenizer)
    }

    fn reference_bundle_dir(name: &str) -> PathBuf {
        paths::bundle_root().join(name)
    }

    fn reference_input_frames_dir(name: &str) -> PathBuf {
        let bundle_dir = reference_bundle_dir(name);
        let tracker_frames = bundle_dir.join("tracker_input_frames");
        if tracker_frames.exists() {
            tracker_frames
        } else {
            bundle_dir.join("frames")
        }
    }

    fn load_reference_frame_output(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = &objects[0];
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing boxes_xyxy", frame_idx))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!("reference frame {} missing score", frame_idx))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!("reference frame {} missing mask_path", frame_idx))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_object_frame_output(
        bundle: &str,
        frame_idx: usize,
        obj_id: u32,
    ) -> Result<(Vec<f32>, f32, PathBuf)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        let frame = frames
            .iter()
            .find(|frame| frame["frame_idx"].as_u64() == Some(frame_idx as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference video results missing frame {}",
                    frame_idx
                ))
            })?;
        let objects = frame["objects"].as_array().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} missing objects array",
                frame_idx
            ))
        })?;
        let object = objects
            .iter()
            .find(|object| object["obj_id"].as_u64() == Some(obj_id as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} missing obj_id {}",
                    frame_idx, obj_id
                ))
            })?;
        let boxes = object["boxes_xyxy"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing boxes_xyxy",
                    frame_idx, obj_id
                ))
            })?
            .iter()
            .map(|value| value.as_f64().unwrap_or(0.0) as f32)
            .collect::<Vec<_>>();
        let score = object["scores"]
            .as_array()
            .and_then(|scores| scores.first())
            .and_then(|value| value.as_f64())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference frame {} obj_id {} missing score",
                    frame_idx, obj_id
                ))
            })? as f32;
        let mask_path = object["mask_path"].as_str().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference frame {} obj_id {} missing mask_path",
                frame_idx, obj_id
            ))
        })?;
        Ok((boxes, score, bundle_dir.join(mask_path)))
    }

    fn load_reference_frame_indices(bundle: &str) -> Result<Vec<usize>> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("video_results.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let frames = match &value {
            serde_json::Value::Array(frames) => frames,
            serde_json::Value::Object(_) => value["frames"].as_array().ok_or_else(|| {
                candle::Error::Msg("reference video results missing frames array".to_owned())
            })?,
            _ => {
                candle::bail!("reference video results must be an array or object with frames")
            }
        };
        Ok(frames
            .iter()
            .filter(|frame| {
                frame["objects"]
                    .as_array()
                    .map(|objects| !objects.is_empty())
                    .unwrap_or(false)
            })
            .filter_map(|frame| frame["frame_idx"].as_u64())
            .map(|frame_idx| frame_idx as usize)
            .collect())
    }

    fn load_reference_frame0_output(bundle: &str) -> Result<(Vec<f32>, f32, PathBuf)> {
        load_reference_frame_output(bundle, 0)
    }

    fn load_reference_box_prompt(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let boxes = if let Some(boxes) = value["boxes_cxcywh_normalized"]
            .as_array()
            .and_then(|boxes| boxes.first())
            .and_then(|first| first.as_array())
        {
            boxes.clone()
        } else {
            value["scenario"]["actions"]
                .as_array()
                .and_then(|actions| {
                    actions.iter().find(|action| {
                        action["type"].as_str() == Some("add_prompt")
                            && action["boxes_xywh"].as_array().is_some()
                    })
                })
                .and_then(|action| action["boxes_xywh"].as_array())
                .and_then(|boxes| boxes.first())
                .and_then(|first| first.as_array())
                .cloned()
                .ok_or_else(|| {
                    candle::Error::Msg(
                        "reference bundle missing box prompt in boxes_cxcywh_normalized or scenario actions"
                            .to_owned(),
                    )
                })?
        };
        let from_scenario_xywh = value["boxes_cxcywh_normalized"]
            .as_array()
            .map(|boxes| boxes.is_empty())
            .unwrap_or(true);
        let (x0_or_cx, y0_or_cy, w, h) = (
            boxes[0].as_f64().unwrap_or(0.0) as f32,
            boxes[1].as_f64().unwrap_or(0.0) as f32,
            boxes[2].as_f64().unwrap_or(0.0) as f32,
            boxes[3].as_f64().unwrap_or(0.0) as f32,
        );
        if from_scenario_xywh {
            Ok((x0_or_cx + w * 0.5, y0_or_cy + h * 0.5, w, h))
        } else {
            Ok((x0_or_cx, y0_or_cy, w, h))
        }
    }

    fn load_reference_mask_prompt_box_xyxy(bundle: &str) -> Result<(f32, f32, f32, f32)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let mask = actions[0]["mask"]["box_xyxy"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference mask scenario missing box_xyxy".to_owned())
        })?;
        Ok((
            mask[0].as_f64().unwrap_or(0.0) as f32,
            mask[1].as_f64().unwrap_or(0.0) as f32,
            mask[2].as_f64().unwrap_or(0.0) as f32,
            mask[3].as_f64().unwrap_or(0.0) as f32,
        ))
    }

    fn load_reference_point_prompt(bundle: &str) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        load_reference_point_prompt_on_frame(bundle, 0)
    }

    fn load_reference_point_prompt_on_frame(
        bundle: &str,
        frame_idx: usize,
    ) -> Result<(Vec<(f32, f32)>, Vec<u32>)> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        let add_prompt = actions
            .iter()
            .find(|action| {
                action["type"].as_str() == Some("add_prompt")
                    && action["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference bundle missing add_prompt action for frame {}",
                    frame_idx
                ))
            })?;
        let points = add_prompt["points_xy_normalized"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg(
                    "reference point scenario missing points_xy_normalized".to_owned(),
                )
            })?
            .iter()
            .map(|point| {
                let point = point.as_array().ok_or_else(|| {
                    candle::Error::Msg(
                        "reference point scenario contains a malformed point".to_owned(),
                    )
                })?;
                Ok((
                    point[0].as_f64().unwrap_or(0.0) as f32,
                    point[1].as_f64().unwrap_or(0.0) as f32,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let labels = add_prompt["point_labels"]
            .as_array()
            .ok_or_else(|| {
                candle::Error::Msg("reference point scenario missing point_labels".to_owned())
            })?
            .iter()
            .map(|value| value.as_u64().unwrap_or(0) as u32)
            .collect::<Vec<_>>();
        Ok((points, labels))
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ReferenceScenarioPointPromptAction {
        frame_idx: usize,
        obj_id: u32,
        points: Vec<(f32, f32)>,
        point_labels: Vec<u32>,
    }

    fn load_reference_point_prompt_actions(
        bundle: &str,
    ) -> Result<Vec<ReferenceScenarioPointPromptAction>> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        actions
            .iter()
            .filter(|action| {
                action["type"].as_str() == Some("add_prompt")
                    && action["points_xy_normalized"].as_array().is_some()
            })
            .map(|action| {
                let frame_idx = action["frame_idx"].as_u64().ok_or_else(|| {
                    candle::Error::Msg(
                        "reference point scenario action missing frame_idx".to_owned(),
                    )
                })? as usize;
                let obj_id = action["obj_id"].as_u64().ok_or_else(|| {
                    candle::Error::Msg("reference point scenario action missing obj_id".to_owned())
                })? as u32;
                let points = action["points_xy_normalized"]
                    .as_array()
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "reference point scenario missing points_xy_normalized".to_owned(),
                        )
                    })?
                    .iter()
                    .map(|point| {
                        let point = point.as_array().ok_or_else(|| {
                            candle::Error::Msg(
                                "reference point scenario contains a malformed point".to_owned(),
                            )
                        })?;
                        Ok((
                            point[0].as_f64().unwrap_or(0.0) as f32,
                            point[1].as_f64().unwrap_or(0.0) as f32,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let point_labels = action["point_labels"]
                    .as_array()
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "reference point scenario missing point_labels".to_owned(),
                        )
                    })?
                    .iter()
                    .map(|value| value.as_u64().unwrap_or(0) as u32)
                    .collect::<Vec<_>>();
                Ok(ReferenceScenarioPointPromptAction {
                    frame_idx,
                    obj_id,
                    points,
                    point_labels,
                })
            })
            .collect()
    }

    fn load_reference_remove_object_actions(bundle: &str) -> Result<Vec<(usize, u32)>> {
        let bundle_dir = reference_bundle_dir(bundle);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(bundle_dir.join("reference.json"))?)
                .map_err(|err| candle::Error::Msg(err.to_string()))?;
        let actions = value["scenario"]["actions"].as_array().ok_or_else(|| {
            candle::Error::Msg("reference bundle missing scenario actions".to_owned())
        })?;
        actions
            .iter()
            .filter(|action| action["type"].as_str() == Some("remove_object"))
            .map(|action| {
                let frame_idx = action["frame_idx"].as_u64().unwrap_or(0) as usize;
                let obj_id = action["obj_id"].as_u64().ok_or_else(|| {
                    candle::Error::Msg(
                        "reference remove_object scenario action missing obj_id".to_owned(),
                    )
                })? as u32;
                Ok((frame_idx, obj_id))
            })
            .collect()
    }

    fn load_reference_internal_manifest(bundle: &str) -> Result<serde_json::Value> {
        let bundle_dir = reference_bundle_dir(bundle);
        serde_json::from_slice(&fs::read(bundle_dir.join("debug/internal_manifest.json"))?)
            .map_err(|err| candle::Error::Msg(err.to_string()))
    }

    fn apply_reference_predictor_runtime_overrides(
        predictor: &mut Sam3VideoPredictor<'_>,
        bundle: &str,
    ) -> Result<()> {
        let manifest = load_reference_internal_manifest(bundle)?;
        let predictor_config = manifest["predictor_config"].as_object().ok_or_else(|| {
            candle::Error::Msg("reference manifest missing predictor_config".to_owned())
        })?;
        if let Some(fill_hole_area) = predictor_config
            .get("fill_hole_area")
            .and_then(|value| value.as_u64())
        {
            predictor.parity_video_config_mut().fill_hole_area = fill_hole_area as usize;
        }
        if let Some(max_point_num) = predictor_config
            .get("max_point_num_in_prompt_enc")
            .and_then(|value| value.as_u64())
        {
            predictor.parity_video_config_mut().max_point_num_in_prompt_enc =
                max_point_num as usize;
        }
        if let Some(non_overlap_masks_for_output) = predictor_config
            .get("non_overlap_masks_for_output")
            .and_then(|value| value.as_bool())
        {
            predictor.parity_video_config_mut().non_overlap_masks_for_output =
                non_overlap_masks_for_output;
        }
        Ok(())
    }

    fn load_reference_internal_tensor(bundle: &str, key: &str) -> Result<Tensor> {
        use candle::safetensors::Load;

        let bundle_dir = reference_bundle_dir(bundle);
        let path = bundle_dir.join("debug/internal_fixtures.safetensors");
        let tensors =
            unsafe { candle::safetensors::MmapedSafetensors::new(&path) }.map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to mmap reference fixtures {}: {err}",
                    path.display()
                ))
            })?;
        tensors
            .get(key)
            .map_err(|err| {
                candle::Error::Msg(format!(
                    "failed to read tensor `{key}` from reference fixtures {}: {err}",
                    path.display()
                ))
            })?
            .load(&Device::Cpu)
    }

    fn load_reference_internal_record(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<serde_json::Value> {
        let records = load_reference_internal_records(bundle, stage, frame_idx)?;
        records.into_iter().next().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference manifest missing {stage} record for frame {frame_idx}"
            ))
        })
    }

    fn load_reference_internal_records(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let manifest = load_reference_internal_manifest(bundle)?;
        Ok(manifest["records"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference manifest missing records".to_owned()))?
            .iter()
            .filter(|record| {
                record["stage"].as_str() == Some(stage)
                    && record["frame_idx"].as_u64() == Some(frame_idx as u64)
            })
            .cloned()
            .collect())
    }

    fn load_reference_internal_record_matching<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_internal_record_matching_last<F>(
        bundle: &str,
        stage: &str,
        frame_idx: usize,
        predicate: F,
    ) -> Result<serde_json::Value>
    where
        F: Fn(&serde_json::Value) -> bool,
    {
        load_reference_internal_records(bundle, stage, frame_idx)?
            .into_iter()
            .rev()
            .find(predicate)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference manifest missing last matching {stage} record for frame {frame_idx}"
                ))
            })
    }

    fn load_reference_track_step_frame_output(
        bundle: &str,
        frame_idx: usize,
        video_size: ImageSize,
    ) -> Result<(Vec<f32>, f32, Tensor)> {
        let record = load_reference_internal_record(bundle, "track_step", frame_idx)?;
        let tensor_keys = record["tensor_keys"].as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "reference track_step frame {frame_idx} missing tensor_keys"
            ))
        })?;
        let high_res_key = tensor_keys
            .get("track_step_output.pred_masks_high_res")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing pred_masks_high_res key"
                ))
            })?;
        let object_score_key = tensor_keys
            .get("track_step_output.object_score_logits")
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "reference track_step frame {frame_idx} missing object_score_logits key"
                ))
            })?;
        let mask_logits = load_reference_internal_tensor(bundle, high_res_key)?;
        let resized_logits = resize_mask_logits_to_video(&mask_logits, video_size)?;
        let masks = candle_nn::ops::sigmoid(&resized_logits)?;
        let boxes = mask_to_normalized_xyxy(&masks)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let presence_score =
            candle_nn::ops::sigmoid(&load_reference_internal_tensor(bundle, object_score_key)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .next()
                .unwrap_or(0.0);
        Ok((boxes, presence_score, masks))
    }

    fn resize_mask_logits_to_video(mask_logits: &Tensor, video_size: ImageSize) -> Result<Tensor> {
        let mask_logits = match mask_logits.rank() {
            2 => mask_logits.unsqueeze(0)?.unsqueeze(0)?,
            3 => mask_logits.unsqueeze(0)?,
            4 => mask_logits.clone(),
            rank => candle::bail!("expected mask logits rank 2, 3, or 4, got {}", rank),
        };
        mask_logits.upsample_bilinear2d(video_size.height, video_size.width, false)
    }

    fn mask_to_normalized_xyxy(mask: &Tensor) -> Result<Tensor> {
        let mask = match mask.rank() {
            4 => mask.i((0, 0))?,
            3 => mask.i(0)?,
            2 => mask.clone(),
            rank => candle::bail!("expected mask rank 2, 3, or 4, got {}", rank),
        };
        let (height, width) = mask.dims2()?;
        if height == 0 || width == 0 {
            return Tensor::zeros((1, 4), DType::F32, mask.device());
        }
        let binary = mask.ge(0.5f32)?.to_dtype(DType::F32)?;
        let row_any = binary.max(candle::D::Minus1)?;
        let col_any = binary.max(candle::D::Minus2)?;
        if row_any.max_all()?.to_scalar::<f32>()? <= 0.0 {
            return Tensor::zeros((1, 4), DType::F32, mask.device());
        }
        let width_scale = width.max(1) as f64;
        let height_scale = height.max(1) as f64;
        let min_x = col_any
            .argmax(0)?
            .to_dtype(DType::F32)?
            .reshape((1,))?
            .affine(1.0 / width_scale, 0.0)?;
        let min_y = row_any
            .argmax(0)?
            .to_dtype(DType::F32)?
            .reshape((1,))?
            .affine(1.0 / height_scale, 0.0)?;
        let max_x = col_any
            .flip(&[0])?
            .argmax(0)?
            .to_dtype(DType::F32)?
            .reshape((1,))?
            .affine(-1.0 / width_scale, 1.0)?;
        let max_y = row_any
            .flip(&[0])?
            .argmax(0)?
            .to_dtype(DType::F32)?
            .reshape((1,))?
            .affine(-1.0 / height_scale, 1.0)?;
        Tensor::stack(&[&min_x, &min_y, &max_x, &max_y], 0)?.reshape((1, 4))
    }

    fn load_reference_run_single_temporal_metadata_last_per_frame(
        bundle: &str,
    ) -> Result<BTreeMap<usize, ParityTemporalDisambiguationFrameMetadata>> {
        let manifest = load_reference_internal_manifest(bundle)?;
        let records = manifest["records"]
            .as_array()
            .ok_or_else(|| candle::Error::Msg("reference manifest missing records".to_owned()))?;
        let mut metadata_by_frame = BTreeMap::new();
        for record in records.iter().filter(|record| {
            record["stage"].as_str() == Some("run_single_frame_inference")
                && record["frame_idx"].as_u64().is_some()
        }) {
            let frame_idx = record["frame_idx"].as_u64().unwrap_or(0) as usize;
            let metadata = &record["metadata"];
            let read_ids = |key: &str| {
                metadata[key]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .map(|value| value.as_u64().unwrap_or(0) as u32)
                            .collect::<BTreeSet<_>>()
                    })
                    .unwrap_or_default()
            };
            metadata_by_frame.insert(
                frame_idx,
                ParityTemporalDisambiguationFrameMetadata {
                    removed_obj_ids: read_ids("removed_obj_ids"),
                    suppressed_obj_ids: read_ids("suppressed_obj_ids"),
                    unconfirmed_obj_ids: read_ids("unconfirmed_obj_ids"),
                    matched_obj_ids: BTreeSet::new(),
                    unmatched_obj_ids: BTreeSet::new(),
                },
            );
        }
        Ok(metadata_by_frame)
    }

    fn json_usize_vec(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
        value[key]
            .as_array()
            .ok_or_else(|| candle::Error::Msg(format!("missing `{key}` array")))?
            .iter()
            .map(|entry| {
                entry.as_u64().map(|value| value as usize).ok_or_else(|| {
                    candle::Error::Msg(format!("malformed `{key}` entry in reference metadata"))
                })
            })
            .collect()
    }

    fn tensor_to_mask_probs_2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        let tensor = match tensor.rank() {
            2 => tensor.clone(),
            3 => tensor.i(0)?,
            4 => tensor.i((0, 0))?,
            rank => candle::bail!("expected mask tensor rank 2/3/4, got {rank}"),
        };
        tensor.to_dtype(DType::F32)?.to_vec2::<f32>()
    }

    fn assert_tensor_close(
        label: &str,
        actual: &Tensor,
        expected: &Tensor,
        atol: f32,
    ) -> Result<()> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "{label} shape mismatch: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        let max_abs_diff = actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        if max_abs_diff > atol {
            candle::bail!("{label} max abs diff {max_abs_diff:.6} exceeded tolerance {atol:.6}");
        }
        Ok(())
    }

    fn tensor_max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        if actual.shape() != expected.shape() {
            candle::bail!(
                "shape mismatch when computing max abs diff: actual {:?}, expected {:?}",
                actual.shape().dims(),
                expected.shape().dims()
            );
        }
        let actual = actual.to_dtype(DType::F32)?;
        let expected = expected.to_dtype(DType::F32)?;
        actual
            .broadcast_sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()
    }

    fn binary_mask_iou(actual: &Tensor, expected_path: &Path) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = image::open(expected_path)
            .map_err(|err| candle::Error::Msg(err.to_string()))?
            .to_luma8();
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (y, row) in actual.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let actual_fg = *value >= 0.5;
                let expected_fg = expected.get_pixel(x as u32, y as u32)[0] >= 128;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn binary_mask_iou_tensor(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = tensor_to_mask_probs_2d(actual)?;
        let expected = tensor_to_mask_probs_2d(expected)?;
        if actual.len() != expected.len()
            || actual.first().map(Vec::len).unwrap_or(0)
                != expected.first().map(Vec::len).unwrap_or(0)
        {
            candle::bail!(
                "mask size mismatch when computing IoU from tensors: actual={}x{}, expected={}x{}",
                actual.len(),
                actual.first().map(Vec::len).unwrap_or(0),
                expected.len(),
                expected.first().map(Vec::len).unwrap_or(0)
            );
        }
        let mut intersection = 0usize;
        let mut union = 0usize;
        for (actual_row, expected_row) in actual.iter().zip(expected.iter()) {
            for (actual_value, expected_value) in actual_row.iter().zip(expected_row.iter()) {
                let actual_fg = *actual_value >= 0.5;
                let expected_fg = *expected_value >= 0.5;
                if actual_fg && expected_fg {
                    intersection += 1;
                }
                if actual_fg || expected_fg {
                    union += 1;
                }
            }
        }
        Ok(if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        })
    }

    fn assert_boxes_close(actual: &[f32], expected: &[f32], atol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= atol,
                "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
            );
        }
    }

    fn box_mismatch_message(actual: &[f32], expected: &[f32], atol: f32) -> Option<String> {
        if actual.len() != expected.len() {
            return Some(format!(
                "box length mismatch: actual={}, expected={}",
                actual.len(),
                expected.len()
            ));
        }
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            if (actual - expected).abs() > atol {
                return Some(format!(
                    "box component {idx} mismatch: actual={actual}, expected={expected}, atol={atol}"
                ));
            }
        }
        None
    }

    fn mask_tensor_to_binary_image(mask: &Tensor) -> Result<GrayImage> {
        let mask_probs = tensor_to_mask_probs_2d(mask)?;
        let height = mask_probs.len() as u32;
        let width = mask_probs.first().map(Vec::len).unwrap_or(0) as u32;
        let mut image = GrayImage::new(width, height);
        for (y, row) in mask_probs.iter().enumerate() {
            for (x, value) in row.iter().enumerate() {
                let pixel = if *value >= 0.5 { 255u8 } else { 0u8 };
                image.put_pixel(x as u32, y as u32, Luma([pixel]));
            }
        }
        Ok(image)
    }

    fn save_binary_mask_png(path: &Path, mask: &Tensor) -> Result<()> {
        mask_tensor_to_binary_image(mask)?
            .save(path)
            .map_err(|err| candle::Error::Msg(format!("failed to save {}: {err}", path.display())))
    }

    fn maybe_tensor_shape(tensor: Option<&Tensor>) -> Option<Vec<usize>> {
        tensor.map(|tensor| tensor.shape().dims().to_vec())
    }

    fn maybe_single_tensor_value(tensor: Option<&Tensor>) -> Result<Option<f32>> {
        match tensor {
            Some(tensor) => Ok(Some(
                tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .next()
                    .unwrap_or(0.0),
            )),
            None => Ok(None),
        }
    }

    fn dump_correction_failure_context(
        bundle: &str,
        actual8: &ObjectFrameOutput,
        actual9: &ObjectFrameOutput,
        expected_boxes8: &[f32],
        expected_score8: f32,
        expected_mask_path8: &Path,
        expected_boxes9: &[f32],
        expected_score9: f32,
        expected_mask_path9: &Path,
        frame8_state: &TrackerFrameState,
        correction_track_step: &serde_json::Value,
        correction_forward: &serde_json::Value,
        prepare_record: &serde_json::Value,
        failures: &[String],
        mask_iou8: f32,
        mask_iou9: f32,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir =
            PathBuf::from("/tmp/sam3_test_failures").join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;

        save_binary_mask_png(&out_dir.join("actual_frame8_mask.png"), &actual8.masks)?;
        save_binary_mask_png(&out_dir.join("actual_frame9_mask.png"), &actual9.masks)?;
        fs::copy(
            expected_mask_path8,
            out_dir.join("expected_frame8_mask.png"),
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path8.display()
            ))
        })?;
        fs::copy(
            expected_mask_path9,
            out_dir.join("expected_frame9_mask.png"),
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to copy {}: {err}",
                expected_mask_path9.display()
            ))
        })?;

        let summary = serde_json::json!({
            "bundle": bundle,
            "failures": failures,
            "frame8": {
                "actual_boxes_xyxy": actual8.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes8,
                "actual_score": actual8.parity_score_value()?,
                "expected_score": expected_score8,
                "actual_presence_score": maybe_single_tensor_value(actual8.presence_scores.as_ref())?,
                "memory_frame_indices": actual8.memory_frame_indices,
                "mask_iou": mask_iou8,
            },
            "frame9": {
                "actual_boxes_xyxy": actual9.boxes_xyxy.flatten_all()?.to_vec1::<f32>()?,
                "expected_boxes_xyxy": expected_boxes9,
                "actual_score": actual9.parity_score_value()?,
                "expected_score": expected_score9,
                "actual_presence_score": maybe_single_tensor_value(actual9.presence_scores.as_ref())?,
                "memory_frame_indices": actual9.memory_frame_indices,
                "mask_iou": mask_iou9,
            },
            "frame8_state": {
                "is_cond_frame": frame8_state.is_cond_frame,
                "maskmem_features_present": frame8_state.maskmem_features.is_some(),
                "maskmem_features_shape": maybe_tensor_shape(frame8_state.maskmem_features.as_ref()),
                "maskmem_pos_enc_present": frame8_state.maskmem_pos_enc.is_some(),
                "maskmem_pos_enc_shape": maybe_tensor_shape(frame8_state.maskmem_pos_enc.as_ref()),
                "object_score_logits": frame8_state.object_score_logits.flatten_all()?.to_vec1::<f32>()?,
            },
            "reference_internal_records": {
                "correction_track_step": correction_track_step,
                "correction_forward_sam_heads": correction_forward,
                "frame9_prepare_memory_conditioned_features": prepare_record,
            }
        });
        fs::write(
            out_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write correction failure summary in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn dump_simple_correction_failure_json(
        bundle: &str,
        phase: &str,
        details: &serde_json::Value,
    ) -> Result<PathBuf> {
        use std::fs;
        use std::time::{SystemTime, UNIX_EPOCH};

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| candle::Error::Msg(format!("time went backwards: {err}")))?
            .as_millis();
        let out_dir =
            PathBuf::from("/tmp/sam3_test_failures").join(format!("{}_{}", bundle, stamp));
        fs::create_dir_all(&out_dir).map_err(|err| {
            candle::Error::Msg(format!(
                "failed to create correction failure directory {}: {err}",
                out_dir.display()
            ))
        })?;
        fs::write(
            out_dir.join(format!("{phase}.json")),
            serde_json::to_vec_pretty(details)
                .map_err(|err| candle::Error::Msg(format!("failed to serialize summary: {err}")))?,
        )
        .map_err(|err| {
            candle::Error::Msg(format!(
                "failed to write simple correction failure dump in {}: {err}",
                out_dir.display()
            ))
        })?;
        Ok(out_dir)
    }

    fn normalized_box_xyxy_to_mask_tensor(
        box_xyxy: (f32, f32, f32, f32),
        size: ImageSize,
        device: &Device,
    ) -> Result<Tensor> {
        let clamp = |value: f32| value.clamp(0.0, 1.0);
        let x0 = (clamp(box_xyxy.0) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y0 = (clamp(box_xyxy.1) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let x1 = (clamp(box_xyxy.2) * (size.width.saturating_sub(1)) as f32).round() as usize;
        let y1 = (clamp(box_xyxy.3) * (size.height.saturating_sub(1)) as f32).round() as usize;
        let mut data = vec![0f32; size.height * size.width];
        if x0 <= x1 && y0 <= y1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    data[y * size.width + x] = 1.0;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, size.height, size.width), device)
    }

    include!("video_parity.rs");
}
