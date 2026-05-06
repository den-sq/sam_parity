# Strict Port Step 2–8 Bundle Coverage

Actionable follow-up is tracked in GitHub issues. This document is a coverage
snapshot only.

Current matrix source:
- `docs/video_tracker_strict_port_matrix.json`

Current materialized matrix-equivalent bundles on disk:
- `video_box_debug_default` -> `tests/reference-bundles/reference_video_box_debug`
- `video_box_debug_temporal_disambiguation` -> `tests/reference-bundles/reference_video_box_debug_temporal_disambiguation`
- `video_postprocess_unconfirmed_box_debug` -> `tests/reference-bundles/reference_video_postprocess_unconfirmed_box_debug`
- `video_point_debug_single_click` -> `tests/reference-bundles/reference_video_point_debug_single_click`
- `video_point_debug_multi_click` -> `tests/reference-bundles/reference_video_point_debug_multi_click`
- `video_point_debug_all_points` -> `tests/reference-bundles/reference_video_point_debug_all_points`
- `video_mask_debug` -> `tests/reference-bundles/reference_video_mask_debug`
- `video_correction_click_debug` -> `tests/reference-bundles/reference_video_correction_click_debug`
- `video_correction_click_no_prev_mask_pred_debug` -> `tests/reference-bundles/reference_video_correction_click_no_prev_mask_pred_debug`
- `video_correction_click_prev_mem_debug` -> `tests/reference-bundles/reference_video_correction_click_prev_mem_debug`
- `video_correction_click_stateless_refinement_debug` -> `tests/reference-bundles/reference_video_correction_click_stateless_refinement_debug`
- `video_correction_click_no_clear_mem_debug` -> `tests/reference-bundles/reference_video_correction_click_no_clear_mem_debug`
- `video_correction_click_not_all_frames_cond_debug` -> `tests/reference-bundles/reference_video_correction_click_not_all_frames_cond_debug`
- `video_start_from_first_ann_debug` -> `tests/reference-bundles/reference_video_start_from_first_ann_debug`
- `video_multi_object_debug` -> `tests/reference-bundles/reference_video_multi_object_debug`
- `video_multi_object_clear_mem_debug` -> `tests/reference-bundles/reference_video_multi_object_clear_mem_debug`
- `video_output_non_overlap_debug` -> `tests/reference-bundles/reference_video_output_non_overlap_debug`
- `video_mem_non_overlap_debug` -> `tests/reference-bundles/reference_video_mem_non_overlap_debug`
- `video_long_history_stride1_debug` -> `tests/reference-bundles/reference_video_long_history_stride1_debug`
- `video_long_history_obj_ptr_overflow_debug` -> `tests/reference-bundles/reference_video_long_history_obj_ptr_overflow_debug`
- `video_long_history_stride_gt1_debug` -> `tests/reference-bundles/reference_video_long_history_stride_gt1_debug`
- `video_long_history_keep_first_cond_debug` -> `tests/reference-bundles/reference_video_long_history_keep_first_cond_debug`
- `video_long_history_temporal_disambiguation_debug` -> `tests/reference-bundles/reference_video_long_history_temporal_disambiguation_debug`
- `video_long_history_trim_mem_debug` -> `tests/reference-bundles/reference_video_long_history_trim_mem_debug`
- `video_fill_hole_disabled_debug` -> `tests/reference-bundles/reference_video_fill_hole_disabled_debug`
- `video_fill_hole_enabled_debug` -> `tests/reference-bundles/reference_video_fill_hole_enabled_debug`
- `video_offload_output_cpu_debug` -> `tests/reference-bundles/reference_video_offload_output_cpu_debug`
- `video_reverse_propagation_debug` -> `tests/reference-bundles/reference_video_reverse_propagation_debug`
- `video_multimask_disabled_tracking_debug` -> `tests/reference-bundles/reference_video_multimask_disabled_tracking_debug`
- `video_multimask_disabled_sam_debug` -> `tests/reference-bundles/reference_video_multimask_disabled_sam_debug`
- `video_postprocess_hidden_obj_debug` -> `tests/reference-bundles/reference_video_postprocess_hidden_obj_debug`
- `video_suppressed_obj_ids_text_bed_debug` -> `tests/reference-bundles/reference_video_suppressed_obj_ids_text_bed_debug`

Interpretation of the `export_reference.py builds bundle?` column:
- `Yes` means the current exporter supports the scenario action set and override surface needed to construct that matrix row.
- It does **not** mean the bundle has already been materialized successfully on disk. Materialization status is reported separately in the `Sufficient bundle with export data exists?` column.

Legacy non-matrix bundle not counted toward strict-port coverage:
- `tests/reference-bundles/reference_tracker_box_debug`

Flash Attention note:
- the confirmed `reference_video_box_debug` export was generated with Flash Attention disabled because the available GPU is pre-Ampere. This is acceptable for strict parity against the current upstream export path, but it does not validate a Flash-Attention-enabled upstream runtime.

BF16 precision note:
- strict parity is now conditioned on identical backend precision support between Candle and upstream PyTorch.
- if upstream uses BF16 for a path but Candle does not have a working BF16 backend for the same path on the current runtime target, the residual difference must be tracked as a backend-support limitation rather than a strict-port behavioral failure.
- current confirmed example:
  - SAM3 vision `patch_embed` on the CUDA tracker path matches upstream exactly under PyTorch CUDA autocast BF16
  - the same path does not currently have a working BF16 implementation in Candle on this machine/runtime

## Step 2: Builder / Config Parity

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `build_tracker(apply_temporal_disambiguation=False)` | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Includes `tracker_config`, `predictor_config`, internal fixtures |
| `build_tracker(apply_temporal_disambiguation=True)` | `video_box_debug_temporal_disambiguation` | Yes: `reference_video_box_debug_temporal_disambiguation` | Yes | Confirms `use_memory_selection=True` branch |
| `maskmem_backbone` nested contract | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | Covered through exported runtime config plus internal tensor shapes |
| `transformer` nested contract | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | Covered through exported runtime config plus internal tensor shapes |
| prompt encoder contract | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | Prompt encoder dimensions and tensor shapes exported |
| SAM mask decoder contract | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | Decoder settings plus internal tensor shapes exported |
| `sam_mask_decoder_extra_args` | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | `dynamic_multimask_*` fields exported in `tracker_config` |
| constructed parameter / tensor shapes from tracker init and `_build_sam_heads` | `video_box_debug_default`, `video_box_debug_temporal_disambiguation` | Yes | Yes | Step 2 fixture-backed tests now consume these bundles |

## Step 3: Prompt-Frame Seed / Prompt Modes

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `start_session` / init-state video-inference path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Session metadata, source, configs, and frame 0 internals present |
| `add_prompt` box path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Box prompt response and frame 0 internals exported |
| `add_prompt` single-click point path | `video_point_debug_single_click` | Yes: `reference_video_point_debug_single_click` | Yes | Materialized via the tracker-engine path with internal `track_step` / SAM-head stages exported |
| `add_prompt` multi-click point path | `video_point_debug_multi_click` | Yes: `reference_video_point_debug_multi_click` | Yes | Materialized via the tracker-engine path with internal `track_step` / SAM-head stages exported |
| point prompt without truncation (`max_point_num_in_prompt_enc<=0`) | `video_point_debug_all_points` | Yes: `reference_video_point_debug_all_points` | Yes | Materialized via the tracker-engine path with internal `track_step` / SAM-head stages exported |
| `add_mask_prompt` / direct mask input path | `video_mask_debug` | Yes: `reference_video_mask_debug` | Yes | Materialized tracker-engine mask row |
| `_get_visual_prompt` initial visual box branch | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `get_visual_prompt` stage present after confirmation rerun |
| non-visual prompt path | `video_point_debug_single_click`, `video_mask_debug` | Yes | Yes | Point and direct-mask prompt bundles are both materialized |
| `_tracker_add_new_objects` detector-to-tracker seed handoff | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `tracker_add_new_objects_input` and `_post_preflight` stages present |
| `_use_mask_as_output` path | `video_box_debug_default`, `video_mask_debug` | Yes: `reference_video_box_debug`, `reference_video_mask_debug` | Yes | The dedicated mask bundle is materialized and exports `use_mask_as_output`, but Candle still misses the direct-mask `obj_ptr` parity check in `tracker_use_mask_as_output_matches_direct_mask_fixture_values` / `tracker_track_frame_matches_mask_prompt_fixture_values` |

## Step 4: SAM-Head Execution

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `get_tpos_enc` / object-pointer temporal encoding inputs | `video_box_debug_default`, `video_long_history_stride1_debug`, `video_long_history_obj_ptr_overflow_debug` | Yes | Yes | The default, stride-1 long-history, and overflow bundles all export `selected_object_pointer_*` metadata and temporal encodings; the stride-1 and overflow parity checks both pass |
| `use_multimask` enabled path | `video_point_debug_single_click` | Yes: `reference_video_point_debug_single_click` | Yes | Materialized tracker-engine point bundle |
| `use_multimask` disabled by point-count gating | `video_point_debug_multi_click` | Yes: `reference_video_point_debug_multi_click` | Yes | Materialized multi-click tracker-engine bundle |
| `_forward_sam_heads` standard path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `forward_sam_heads` stage present |
| raw prompt encoder sparse / dense embeddings | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `sam_prompt_encoder` stage now exports them |
| raw mask decoder multimask logits / IoUs / tokens | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `sam_mask_decoder` stage now exports multimasks, IoUs, tokens, object score logits |
| `multimask_output_for_tracking=False` | `video_multimask_disabled_tracking_debug` | Yes: `reference_video_multimask_disabled_tracking_debug` | Yes | Materialized tracker-engine point bundle with tracking multimask disabled |
| `multimask_output_in_sam=False` | `video_multimask_disabled_sam_debug` | Yes: `reference_video_multimask_disabled_sam_debug` | Yes | Materialized tracker-engine point bundle with SAM multimask disabled |
| dynamic multimask via stability fallback | `video_point_debug_single_click` | Yes: `reference_video_point_debug_single_click` | Yes | Materialized tracker-engine point bundle |
| multimask token / object-pointer selection branch | `video_point_debug_single_click` | Yes: `reference_video_point_debug_single_click` | Yes | Materialized tracker-engine point bundle |

## Step 5: Memory Conditioning

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `prepare_memory_conditioned_features` default short-history path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Stage present |
| object-pointer temporal encoding | `video_box_debug_default`, `video_long_history_stride1_debug`, `video_long_history_obj_ptr_overflow_debug` | Yes | Yes | Long-history temporal offsets and encoder-cap overflow/truncation are both reference-backed and currently green in Candle parity tests |
| conditioning / memory frame selection default path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Selected frame indices and sources exported |
| `use_memory_selection=False` | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Confirmed in tracker config and selection metadata |
| `use_memory_selection=True` default short-history path | `video_box_debug_temporal_disambiguation` | Yes: `reference_video_box_debug_temporal_disambiguation` | Yes | Temporal-disambiguation bundle exists, but does not stress long-history selection |
| `use_memory_selection=True` long-history / `frame_filter` path | `video_long_history_temporal_disambiguation_debug` | Yes: `reference_video_long_history_temporal_disambiguation_debug` | Yes | Materialized long-history temporal-disambiguation bundle |
| `keep_first_cond_frame=True` | `video_long_history_keep_first_cond_debug` | Yes: `reference_video_long_history_keep_first_cond_debug` | Yes | Materialized long-history keep-first-conditioning bundle |
| `memory_temporal_stride_for_eval=1` overflow path | `video_long_history_stride1_debug` | Yes: `reference_video_long_history_stride1_debug` | Yes | Materialized long-history stride-1 bundle |
| `memory_temporal_stride_for_eval>1` | `video_long_history_stride_gt1_debug` | Yes: `reference_video_long_history_stride_gt1_debug` | Yes | Materialized long-history stride>1 bundle |
| `max_cond_frames_in_attn` overflow / truncation | `video_long_history_stride1_debug` | Yes: `reference_video_long_history_stride1_debug` | Yes | Materialized long-history stride-1 bundle |
| `max_obj_ptrs_in_encoder` overflow / truncation | `video_long_history_obj_ptr_overflow_debug` | Yes: `reference_video_long_history_obj_ptr_overflow_debug` | Yes | Dedicated overflow row now proves frame-29 selection exceeds the 16-cap and truncates the oldest non-conditioning pointer frame |

## Step 6: Memory Writing / State Updates

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `encode_new_memory` default path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `encode_new_memory` stage present |
| `run_mem_encoder=False` prompt-frame path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Visible in `track_step` metadata on seed frame |
| `run_mem_encoder=True` propagation / preflight path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `run_memory_encoder` stage present |
| `propagate_in_video_preflight` | `video_mem_non_overlap_debug`, `video_long_history_trim_mem_debug`, `video_mask_debug`, `video_correction_click_debug` | Yes for tracker-engine paths via `reference_video_mem_non_overlap_debug` and `reference_video_long_history_trim_mem_debug` | Yes | Exporter records `propagate_in_video_preflight`; tracker-engine Step 6 bundles now exercise it, and Candle `video.rs` now preflights missing prompt-frame memory before non-prompt propagation |
| session-side state updates after preflight | `video_mem_non_overlap_debug`, `video_long_history_trim_mem_debug`, `video_mask_debug`, `video_correction_click_debug` | Yes for tracker-engine paths via `reference_video_mem_non_overlap_debug` and `reference_video_long_history_trim_mem_debug` | Yes | Preflight metadata now captures output-frame keys, per-object slices, and tracking-start transitions on real bundles; Candle now persists preflighted tracker states plus propagated outputs back into the session runtime |
| `non_overlap_masks_for_mem_enc=True` | `video_mem_non_overlap_debug` | Yes: `reference_video_mem_non_overlap_debug` | Yes | Materialized two-object tracker-engine bundle |
| `trim_past_non_cond_mem_for_eval=True` | `video_long_history_trim_mem_debug` | Yes: `reference_video_long_history_trim_mem_debug` | Yes | Materialized long-history tracker-engine bundle |
| `offload_output_to_cpu_for_eval=True` | `video_offload_output_cpu_debug` | Yes: `reference_video_offload_output_cpu_debug` | Yes | Materialized with an exporter-side contract shim that restores missing `maskmem_features` / `maskmem_pos_enc` keys as `None` on the prompt-frame offload path, plus explicit offload metadata in `track_step` / `run_single_frame_inference`; the row now uses the known-good single-click tracker prompt so the observable bundle also contains propagated objects |

## Step 7: Predictor / Orchestration

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `_det_track_one_frame` | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Stage present |
| `_run_single_frame_inference` | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Stage present |
| forward `propagate_in_video` orchestration | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | Observable outputs plus internal stages exist |
| `build_tracker_output` refinement/output assembly path | `video_correction_click_debug` | Yes: `reference_video_correction_click_debug` | Yes | Materialized bundle includes propagated pre-correction outputs plus frame-8 correction `build_tracker_output` records |
| correction click default behavior | `video_correction_click_debug` | Yes: `reference_video_correction_click_debug` | Yes | Materialized bundle now covers frame-8 click refinement and post-correction frame-9 propagation |
| `iter_use_prev_mask_pred=False` | `video_correction_click_no_prev_mask_pred_debug` | Yes: `reference_video_correction_click_no_prev_mask_pred_debug` | Yes | Materialized correction bundle; frame-8 refinement omits the previous low-res mask prompt as expected |
| `use_prev_mem_frame=True` | `video_correction_click_prev_mem_debug` | Yes: `reference_video_correction_click_prev_mem_debug` | Yes | Materialized correction bundle; frame-8 refinement uses previous memory conditioning |
| `use_stateless_refinement=True` | `video_correction_click_stateless_refinement_debug` | Yes: `reference_video_correction_click_stateless_refinement_debug` | Yes | Materialized correction bundle; Candle runtime now honors the bundle-side stateless-refinement flag for frame-8 refinement |
| `refinement_detector_cond_frame_removal_window` override | `video_correction_click_stateless_refinement_debug` | Yes: `reference_video_correction_click_stateless_refinement_debug` | Yes | Materialized bundle carries the narrowed refinement-detector removal window override |
| `clear_non_cond_mem_around_input=True` behavior | `video_correction_click_debug` | Yes: `reference_video_correction_click_debug` | Yes | Exporter wraps `_clear_non_cond_mem_around_input`; default correction bundle now materialized for this path |
| `clear_non_cond_mem_around_input=False` | `video_correction_click_no_clear_mem_debug` | Yes: `reference_video_correction_click_no_clear_mem_debug` | Yes | Materialized correction bundle with surrounding-memory clearing disabled |
| `add_all_frames_to_correct_as_cond=False` | `video_correction_click_not_all_frames_cond_debug` | Yes: `reference_video_correction_click_not_all_frames_cond_debug` | Yes | Materialized correction bundle; frame-8 correction stays non-conditioning for later propagation |
| `always_start_from_first_ann_frame=True` | `video_start_from_first_ann_debug` | Yes: `reference_video_start_from_first_ann_debug` | Yes | Materialized as a tracker-engine bundle, which is the upstream SAM3 path that actually exposes this branch; Candle also has a reference-backed test for the branch in addition to the unit-level processing-order coverage |
| multi-object tracking | `video_multi_object_debug` | Yes: `reference_video_multi_object_debug` | Yes | Materialized two-object bundle with observable outputs for both tracked objects |
| `clear_non_cond_mem_for_multi_obj=True` | `video_multi_object_clear_mem_debug` | Yes: `reference_video_multi_object_clear_mem_debug` | Yes | Materialized two-object correction bundle with multi-object surrounding-memory clearing enabled; Candle runtime branch is also covered by `clear_non_cond_mem_around_input_respects_multi_object_flag` |
| reverse propagation | `video_reverse_propagation_debug` | Yes: `reference_video_reverse_propagation_debug` | Yes | Materialized reverse-propagation bundle starting from frame 20 and tracking backward |

## Step 8: Final Output / Postprocess

| Requirement | Exact bundle(s) | Sufficient bundle with export data exists? | `export_reference.py` builds bundle? | Notes |
| --- | --- | --- | --- | --- |
| `_postprocess_output` default path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `postprocess_output` stage present after confirmation rerun |
| final resize-to-video path | `video_box_debug_default` | Yes: `reference_video_box_debug` | Yes | `postprocess_output.out_binary_masks` is video-resolution output |
| output non-overlap path | `video_output_non_overlap_debug` | Yes: `reference_video_output_non_overlap_debug` | Yes | Materialized and now consumed by a reference-backed Candle test |
| hole-filling disabled | `video_fill_hole_disabled_debug` | Yes: `reference_video_fill_hole_disabled_debug` | Yes | Materialized and now consumed by a reference-backed Candle test |
| hole-filling enabled | `video_fill_hole_enabled_debug` | Yes: `reference_video_fill_hole_enabled_debug` | Yes | Materialized and now consumed by a reference-backed Candle test |
| hidden / suppressed / unconfirmed object postprocess path | `video_postprocess_hidden_obj_debug`, `video_box_debug_temporal_disambiguation`, `video_postprocess_unconfirmed_box_debug`, `video_suppressed_obj_ids_text_bed_debug` | Yes: `reference_video_postprocess_hidden_obj_debug` certifies the final empty-output path, `reference_video_box_debug_temporal_disambiguation` certifies the delayed-yield / unmatched-hotstart visible-output path, `reference_video_postprocess_unconfirmed_box_debug` certifies non-empty `unconfirmed_obj_ids` followed by `removed_obj_ids` on the producer side, and `reference_video_suppressed_obj_ids_text_bed_debug` certifies non-empty `suppressed_obj_ids` from duplicate/occlusion suppression | Yes | Candle now has reference-backed coverage for the consumer path, delayed-yield hotstart, the confirmation/unmatched-removal producer slice in `propagate_one_direction`, and the duplicate/occlusion suppression producer path via the text-prompt replay parity test |
| empty / missing-object output path | `video_postprocess_hidden_obj_debug` | Yes: `reference_video_postprocess_hidden_obj_debug` | Yes | The official row now uses a negative-point prompt and produces zero visible objects on every frame, so Candle can certify the empty final-output branch against a real upstream bundle |
