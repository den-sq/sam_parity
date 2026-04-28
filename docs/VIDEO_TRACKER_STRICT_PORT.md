# SAM3 Video Tracker Strict Port Contract

This document defines the strict-port rewrite plan for the SAM3 video tracker core and
video propagation path.

The previous tracker core and the previous video propagation/orchestration path have been
removed. Until the strict port is complete, the remaining tracker/video entry points are
intentional scaffolds that stop with a "strict port in progress" error rather than
continuing with local behavior.

This plan covers SAM3 only. It does not cover SAM3.1 multiplex or the SAM3.1-specific
video tracking builders in `model_builder.py`.

## Current Step Status

- `1a. Exporter infrastructure`
  - implemented in `python/sam3_parity/export_reference.py`
- `1b. Required upstream bundle matrix definition and generation workflow`
  - implemented in:
    - `docs/video_tracker_strict_port_matrix.json`
    - `python/sam3_parity/generate_video_tracker_strict_port_matrix.py`
- `Stage 0 / export completeness`
  - all required matrix rows are now materialized on disk except the optional
    `video_forward_backbone_all_frames_debug` row
- `Step 2 / builder + config parity`
  - complete
- `Step 3 / prompt-frame seed + prompt modes`
  - complete under the current precision rule
  - remaining `all_points` residuals are tracked as the documented BF16 backend gap in
    the vision `patch_embed` path on this machine/runtime
- `Step 4 / SAM-head execution`
  - complete
- `Step 5 / memory conditioning`
  - complete
- `Step 6 / memory writing + state updates`
  - complete
- `Step 7 / predictor / orchestration`
  - complete, including CUDA certification of the remaining reference-backed rows
- `Step 8 / final output / postprocess`
  - materially complete for the currently-exported upstream surface
  - implemented and reference-backed for:
    - output non-overlap
    - fill-hole disabled
    - fill-hole enabled
    - default `_postprocess_output` / resize-to-video path
    - empty / missing-object output behavior via the negative-point hidden-output row
    - delayed-yield temporal-disambiguation visible-output behavior via the
      `reference_video_box_debug_temporal_disambiguation` row
    - confirmation / hotstart producer metadata via the
      `reference_video_postprocess_unconfirmed_box_debug` row
  - implemented locally for:
    - metadata-driven hide inputs (`removed_obj_ids`, `suppressed_obj_ids`,
      `unconfirmed_obj_ids`) on the postprocess consumer side
    - hotstart-delay buffering and unmatched-track removal in
      `video.rs::propagate_one_direction`
  - corrected temporal-disambiguation certification bundles now show that the
    visible missing behavior was the delayed-yield / unmatched-removal producer
    path, not exporter visibility alone
  - the remaining uncaptured Step 8 surface is narrower:
    - upstream rows with non-empty `suppressed_obj_ids`
    - duplicate/occlusion-driven association suppression rather than the
      confirmation/unmatched-removal row

This distinction is intentional: Step 1 is the repo-side implementation of the export
surface and required matrix. Stage 0 is the operational requirement that the bundles be
generated before runtime tracker work proceeds.

## Non-Negotiable Rules

1. Before coding a stage, produce and keep an explicit upstream-to-local function mapping.
2. Do not replace upstream behavior with heuristics or "equivalent" logic unless explicitly approved.
3. Preserve upstream state/tensor contracts exactly:
   - shapes
   - dtypes
   - thresholds
   - branch conditions
   - update order
   - tensor storage tier transitions when they affect semantics
   - execution precision when the underlying backend support is identical between
     Candle and upstream PyTorch
4. Add parity fixtures for intermediate boundaries, not just final outputs.
5. If a behavior cannot be matched exactly, stop and report the deviation before continuing.

### Precision rule

The strict port must use BF16 when it is available and semantically active in the upstream
runtime path being matched.

However, exact parity is only required for a precision mode when Candle and upstream PyTorch
have identical backend support for that mode on the current runtime target.

This means:

- if upstream and Candle both have working BF16 support for a given op/path on the current
  device, exact parity must be measured in that BF16 path
- if upstream uses BF16 for a path but Candle does not have a working BF16 backend for the
  same path on the current device/runtime, that gap must be recorded as a backend-support
  limitation rather than counted as a strict-port behavioral mismatch
- in that situation, strict parity falls back to:
  - exact parity for the highest precision mode that is supported identically by both sides
  - explicit documentation of the BF16-only residual difference

Examples of backend-support limitations include:

- BF16 CUDA kernels available in upstream PyTorch/cuDNN or cuBLAS but unavailable or failing
  in Candle on the same machine
- Flash-Attention-enabled upstream paths when Candle cannot run the same path on the same GPU

Such cases do not close the port automatically; they must still be documented precisely, with
the affected stage, operation, and residual diff.

## Source Scope

The strict port must be derived from these upstream SAM3 files:

- `sam3/model_builder.py`
- `sam3/model/sam3_tracker_base.py`
- `sam3/model/sam3_tracking_predictor.py`
- `sam3/model/sam3_video_base.py`
- `sam3/model/sam3_video_inference.py`

The strict port must not silently import behavior from:

- SAM3.1 multiplex builders and demos
- local Candle heuristics
- partially similar SAM2 codepaths

## Upstream-to-Local Function Map

The strict port will be implemented in the existing `tracker.rs` and `video.rs` files,
not in separate `strict_*` filenames.

### Tracker builder and constructor contract

- `model_builder.py:_create_tracker_maskmem_backbone`
  - local target: `tracker.rs::create_tracker_maskmem_backbone_config`
- `model_builder.py:_create_tracker_transformer`
  - local target: `tracker.rs::create_tracker_transformer_config`
- `model_builder.py:build_tracker`
  - local target: `tracker.rs::Sam3TrackerConfig::build_tracker`
- `sam3_tracker_base.py:Sam3TrackerBase.__init__`
  - local target: `tracker.rs::Sam3TrackerModel::new`
- `sam3_tracker_base.py:_build_sam_heads`
  - local target: `tracker.rs::Sam3TrackerModel::build_sam_heads`

### Tracker core

- `sam3_tracker_base.py:_get_tpos_enc`
  - local target: `tracker.rs::Sam3TrackerModel::get_tpos_enc`
- `sam3_tracker_base.py:_forward_sam_heads`
  - local target: `tracker.rs::Sam3TrackerModel::forward_sam_heads`
- `sam3_tracker_base.py:_use_mask_as_output`
  - local target: `tracker.rs::Sam3TrackerModel::use_mask_as_output`
- `sam3_tracker_base.py:_prepare_memory_conditioned_features`
  - local target: `tracker.rs::Sam3TrackerModel::prepare_memory_conditioned_features`
- `sam3_tracker_base.py:_encode_new_memory`
  - local target: `tracker.rs::Sam3TrackerModel::encode_new_memory`
- `sam3_tracker_base.py:_use_multimask`
  - local target: `tracker.rs::Sam3TrackerModel::use_multimask`
- `sam3_tracker_base.py:track_step`
  - local target: `tracker.rs::Sam3TrackerModel::track_frame`

### Predictor wrapper

- `sam3_tracking_predictor.py:init_state`
  - local target: `video.rs::Sam3VideoPredictor::start_session`
- `sam3_tracking_predictor.py:_obj_id_to_idx`
  - local target: `video.rs::Sam3VideoPredictor::alloc_or_lookup_object`
- `sam3_tracking_predictor.py:add_new_points_or_box`
  - local target: `video.rs::Sam3VideoPredictor::add_prompt`
- `sam3_tracking_predictor.py:add_new_mask`
  - local target: `video.rs::Sam3VideoPredictor::add_mask_prompt`
- `sam3_tracking_predictor.py:propagate_in_video_preflight`
  - local target: `video.rs::Sam3VideoPredictor::propagate_preflight`
- `sam3_tracking_predictor.py:propagate_in_video`
  - local target: `video.rs::Sam3VideoPredictor::propagate_in_video_stream`
- `sam3_tracking_predictor.py:_clear_non_cond_mem_around_input`
  - local target: `video.rs::Sam3VideoPredictor::clear_non_cond_mem_around_input`

### Video orchestration and postprocess

- `sam3_video_base.py:_det_track_one_frame`
  - local target: `video.rs::Sam3VideoTrackerCore::det_track_one_frame`
- `sam3_video_base.py:_tracker_add_new_objects`
  - local target: `video.rs::Sam3VideoTrackerCore::tracker_add_new_objects`
- `sam3_video_inference.py:_get_visual_prompt`
  - local target: `video.rs::Sam3VideoTrackerCore::get_visual_prompt`
- `sam3_video_inference.py:_run_single_frame_inference`
  - local target: `video.rs::Sam3VideoTrackerCore::run_single_frame_inference`
- `sam3_video_inference.py:_build_tracker_output`
  - local target: `video.rs::Sam3VideoTrackerCore::build_tracker_output`
- `sam3_video_inference.py:_postprocess_output`
  - local target: `video.rs::Sam3VideoTrackerCore::postprocess_output`

## Full Upstream Parity Surface

The strict port must cover all SAM3 branches that affect the tracker, predictor wrapper,
or final video outputs. "Covered" means either:

- exported and fixture-backed by upstream artifacts, or
- explicitly blocked pending a required upstream export bundle

No branch may remain silently source-derived once it can affect runtime semantics.

### Builder and constructor contract

The following must be captured and tested exactly:

- tracker builder defaults from `build_tracker(apply_temporal_disambiguation=False)`
- tracker builder defaults from `build_tracker(apply_temporal_disambiguation=True)`
- `maskmem_backbone` nested contract
- `transformer` nested contract
- prompt encoder contract
- SAM mask decoder contract
- `sam_mask_decoder_extra_args`
- parameterized tensor shapes constructed in `Sam3TrackerBase.__init__` and `_build_sam_heads`

Required constructor-side invariants:

- `image_size`
- `backbone_stride`
- `low_res_mask_size`
- `input_mask_size`
- `num_maskmem`
- `max_cond_frames_in_attn`
- `keep_first_cond_frame`
- `memory_temporal_stride_for_eval`
- `offload_output_to_cpu_for_eval`
- `trim_past_non_cond_mem_for_eval`
- `forward_backbone_per_frame_for_eval`
- `non_overlap_masks_for_mem_enc`
- `max_obj_ptrs_in_encoder`
- `use_memory_selection`
- `mf_threshold`
- `multimask_output_in_sam`
- `multimask_output_for_tracking`
- `multimask_min_pt_num`
- `multimask_max_pt_num`
- `sigmoid_scale_for_mem_enc`
- `sigmoid_bias_for_mem_enc`
- `sam_mask_decoder_extra_args.dynamic_multimask_via_stability`
- `sam_mask_decoder_extra_args.dynamic_multimask_stability_delta`
- `sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh`

Required constructed-parameter/tensor invariants:

- `mask_downsample`
- `maskmem_tpos_enc`
- `no_mem_embed`
- `no_mem_pos_enc`
- `no_obj_ptr`
- `no_obj_embed_spatial`
- prompt encoder `image_embedding_size`, `input_image_size`, `mask_input_size`
- mask decoder multimask count and object-score settings
- `obj_ptr_proj`
- `obj_ptr_tpos_proj`

### Tracker runtime branches

The strict port must cover all of these branches:

- `_use_mask_as_output` path
- standard `_forward_sam_heads` path
- `run_mem_encoder=False`
- `run_mem_encoder=True`
- `multimask_output=False`
- `multimask_output=True`
- multimask point-count gating via `_use_multimask`
- `use_memory_selection=False`
- `use_memory_selection=True`
- `keep_first_cond_frame=False`
- `keep_first_cond_frame=True`
- `memory_temporal_stride_for_eval=1`
- `memory_temporal_stride_for_eval>1`
- `non_overlap_masks_for_mem_enc=False`
- `non_overlap_masks_for_mem_enc=True`
- `offload_output_to_cpu_for_eval=False`
- `offload_output_to_cpu_for_eval=True`
- `trim_past_non_cond_mem_for_eval=False`
- `trim_past_non_cond_mem_for_eval=True`
- `forward_backbone_per_frame_for_eval=True`
- `forward_backbone_per_frame_for_eval=False` if reachable in the video codepath being ported

### Predictor-wrapper runtime branches

The strict port must cover all of these branches from `Sam3TrackerPredictor`:

- single-object tracking
- multi-object tracking
- box prompt
- point prompt
- mask prompt
- correction click on an existing object
- `always_start_from_first_ann_frame=False`
- `always_start_from_first_ann_frame=True`
- `clear_non_cond_mem_around_input=False`
- `clear_non_cond_mem_around_input=True`
- `clear_non_cond_mem_for_multi_obj=False`
- `clear_non_cond_mem_for_multi_obj=True`
- `max_point_num_in_prompt_enc<=0` or "use all points"
- `max_point_num_in_prompt_enc>0` truncation path
- `non_overlap_masks_for_output=False`
- `non_overlap_masks_for_output=True`
- `fill_hole_area=0`
- `fill_hole_area>0`
- `add_all_frames_to_correct_as_cond=True`
- `iter_use_prev_mask_pred=True`
- forward propagation
- reverse propagation

### Video-inference/output branches

The strict port must cover:

- initial visual box prompt path
- non-visual prompt path
- `_build_tracker_output`
- `_postprocess_output`
- video-resolution resize path
- object-score suppression path
- output non-overlap path
- fill-hole postprocess path
- empty/missing-object output path

## Required Upstream Export Matrix

The current `reference_video_box_debug` bundle is only one row in the required matrix.
It is useful, but it is not sufficient for strict parity.

The canonical source-controlled matrix is:

- `docs/video_tracker_strict_port_matrix.json`

The bundle names below must stay in sync with that manifest. The matrix generator is:

- `python/sam3_parity/generate_video_tracker_strict_port_matrix.py`

The following upstream bundles are required before the corresponding local stage can be
considered fully covered.

### Builder/config bundles

1. `video_box_debug_default`
   - `apply_temporal_disambiguation=False`
   - single object
   - box prompt
   - frames 0-3

2. `video_box_debug_temporal_disambiguation`
   - `apply_temporal_disambiguation=True`
   - same scenario as above

### Prompt-mode bundles

3. `video_point_debug_single_click`
   - positive single point
   - exercises multimask-eligible point count branch

4. `video_point_debug_multi_click`
   - multiple clicks
   - exercises non-multimask point-count branch and point truncation when configured

5. `video_point_debug_all_points`
   - multiple clicks
   - `max_point_num_in_prompt_enc<=0`

6. `video_mask_debug`
   - direct mask input
   - exercises `add_new_mask` / mask-input flow

### Correction and predictor-wrapper bundles

7. `video_correction_click_debug`
   - correction click on an already tracked object
   - exercises `iter_use_prev_mask_pred`
   - exercises `add_all_frames_to_correct_as_cond`

8. `video_correction_click_no_prev_mask_pred_debug`
   - correction click
   - `iter_use_prev_mask_pred=False`

9. `video_correction_click_prev_mem_debug`
   - correction click
   - `use_prev_mem_frame=True`

10. `video_correction_click_stateless_refinement_debug`
   - correction click
   - `use_stateless_refinement=True`
   - `refinement_detector_cond_frame_removal_window` override

11. `video_correction_click_no_clear_mem_debug`
   - correction click
   - `clear_non_cond_mem_around_input=False`

12. `video_correction_click_not_all_frames_cond_debug`
   - correction click
   - `add_all_frames_to_correct_as_cond=False`

13. `video_start_from_first_ann_debug`
   - prompt starts after frame 0
   - `always_start_from_first_ann_frame=True`
   - note: this branch is covered via the upstream tracker predictor path, not the SAM3 video-inference engine

### Multi-object and overlap bundles

14. `video_multi_object_debug`
   - at least two objects
   - exercises multi-object output layout

15. `video_multi_object_clear_mem_debug`
   - same as above, but with correction around an input frame
   - exercises `clear_non_cond_mem_for_multi_obj=True`

16. `video_output_non_overlap_debug`
   - `non_overlap_masks_for_output=True`

17. `video_mem_non_overlap_debug`
   - `non_overlap_masks_for_mem_enc=True`

### Memory-selection and long-history bundles

18. `video_long_history_stride1_debug`
   - enough frames and prompts to exceed:
     - `max_cond_frames_in_attn`
   - `memory_temporal_stride_for_eval=1`

19. `video_long_history_obj_ptr_overflow_debug`
   - enough frames and prompts to exceed:
     - `max_obj_ptrs_in_encoder`
   - proves object-pointer truncation at the encoder cap

20. `video_long_history_stride_gt1_debug`
   - same as above with `memory_temporal_stride_for_eval>1`

21. `video_long_history_keep_first_cond_debug`
   - enough conditioning frames to force selection/truncation
   - `keep_first_cond_frame=True`

22. `video_long_history_trim_mem_debug`
   - `trim_past_non_cond_mem_for_eval=True`

### Postprocess/output bundles

23. `video_fill_hole_disabled_debug`
   - `fill_hole_area=0`

24. `video_fill_hole_enabled_debug`
   - `fill_hole_area>0`

25. `video_reverse_propagation_debug`
   - reverse propagation

26. `video_postprocess_hidden_obj_debug`
   - forces the final `_postprocess_output` result to be empty via a negative-point prompt
   - covers object hiding and empty-output postprocess behavior

27. `video_postprocess_unconfirmed_box_debug`
   - drives non-empty `unconfirmed_obj_ids` followed by `removed_obj_ids`
   - covers the confirmation/unmatched-removal producer path

### Multimask bundles

28. `video_multimask_disabled_tracking_debug`
   - `multimask_output_for_tracking=False`

29. `video_multimask_disabled_sam_debug`
   - `multimask_output_in_sam=False`

### Storage/offload bundles

30. `video_offload_output_cpu_debug`
   - `offload_output_to_cpu_for_eval=True`

31. `video_forward_backbone_all_frames_debug`
   - only required if the non-per-frame backbone-forward path is reachable in the
     SAM3 video predictor flow being ported

## Information Still Needed

The plan is now structurally complete, but it is not fully executable yet. Additional
upstream information is still required before several implementation stages can be
described as exact, fixture-backed ports rather than source-derived intentions.

### Already available

The following information is available today:

- the repo-side Step 1 implementation:
  - exporter infrastructure in `export_reference.py`
  - source-controlled export matrix in `video_tracker_strict_port_matrix.json`
  - reproducible generator in `generate_video_tracker_strict_port_matrix.py`
- materialized internal upstream bundles for:
  - `reference_video_box_debug`
  - `reference_video_box_debug_temporal_disambiguation`
- source-level definitions for the full SAM3 tracker/predictor codepath from:
  - `model_builder.py`
  - `sam3_tracker_base.py`
  - `sam3_tracking_predictor.py`
  - `sam3_video_base.py`
  - `sam3_video_inference.py`

This is enough to define the shape of the plan and to complete fixture-backed validation
for the builder/config branches already exported. It is not enough to fully specify or
validate the entire strict port until the rest of the matrix has been materialized.

### Still required to fully describe and execute the plan

The following upstream artifacts are still required unless already generated from the
canonical matrix:

- all prompt-mode bundles other than the two existing box bundles
- all correction/predictor-wrapper bundles
- all multi-object/overlap bundles
- all long-history bundles
- all postprocess bundles
- all multimask bundles
- all storage/offload bundles

### Additional internal fields that may still need export support

If the existing internal fixture format is insufficient for any function-level parity
test, the upstream exporter must be extended before the corresponding Rust implementation
proceeds. Likely candidates include:

- raw SAM prompt-encoder sparse embeddings
- raw SAM prompt-encoder dense embeddings
- raw multimask candidates before best-mask selection
- raw IoU head outputs before dynamic multimask fallback logic
- full decoder token outputs used for object-pointer selection
- explicit postprocess decision metadata when non-overlap or hole-filling branches fire

### Planning rule for missing information

If a future implementation stage depends on an upstream branch or tensor contract that is
not yet covered by exported fixtures, the correct action is:

1. stop,
2. identify the missing upstream information,
3. extend the export plan or export script,
4. regenerate the necessary bundle,
5. only then continue implementation planning or coding for that stage.

No stage should be described as "fully planned" or "ready to implement" if it still
depends on source-derived-only behavior.

## Required Internal Fixture Contents

For every export bundle above, the upstream artifact set must include:

- observable bundle:
  - `reference.json`
  - `video_results.json`
  - `frames/`
  - `masks/`
  - `masked_frames/`
- debug bundle:
  - `debug/debug_manifest.json`
  - binary masks for the captured frames
- internal bundle:
  - `debug/internal_manifest.json`
  - `debug/internal_fixtures.safetensors`

The internal bundle must serialize, where relevant:

- exact config values active for that bundle
- exact shapes and dtypes
- exact selected frame indices
- exact prompt metadata
- exact object-score logits
- exact low-res masks
- exact high-res masks
- exact object pointers
- exact memory features
- exact positional encodings
- exact postprocess thresholds when they are applied

If a branch depends on additional internal tensors not currently exported, the export must
be extended before the Rust implementation proceeds.

## Required Function-Level Tests

Each local function must have upstream-backed tests. Source-derived tests may exist in
addition, but they are not sufficient on their own.

### Builder/config tests

- `Sam3TrackerConfig::build_tracker(false)` matches upstream export exactly
- `Sam3TrackerConfig::build_tracker(true)` matches upstream export exactly
- `create_tracker_maskmem_backbone_config(...)` matches upstream export exactly
- `create_tracker_transformer_config(...)` matches upstream export exactly
- `Sam3TrackerModel::new` exposes exact constructor-derived tensor shapes
- constructed parameter tensors match exported shapes and expected dtypes

### Tracker core tests

- `get_tpos_enc` matches exported tensor values
- `use_mask_as_output` matches exported intermediate tensors and outputs
- `forward_sam_heads` matches exported intermediate tensors and outputs
- `prepare_memory_conditioned_features` matches exported selected frames, token layouts,
  positional encodings, and fused feature tensors
- `encode_new_memory` matches exported mask transforms and memory outputs
- `use_multimask` matches upstream branch decisions for all prompt-count cases
- `track_frame` matches exported outputs for:
  - frame-0 prompt seed
  - frame-1 propagation
  - frame-2 propagation
  - frame-3 propagation

### Predictor-wrapper tests

- `start_session` matches upstream state initialization behavior
- `add_prompt` matches upstream box/point prompt normalization and ordering
- `add_mask_prompt` matches upstream mask-input behavior
- `propagate_preflight` matches upstream consolidation/update order
- `clear_non_cond_mem_around_input` matches upstream frame eviction behavior
- `propagate_in_video_stream` matches upstream frame order in forward and reverse modes

### Video-output tests

- `get_visual_prompt` matches upstream visual-prompt extraction exactly
- `det_track_one_frame` matches upstream frame-level orchestration exactly
- `tracker_add_new_objects` matches upstream add-object flow exactly
- `run_single_frame_inference` matches upstream orchestration exactly
- `build_tracker_output` matches upstream aggregation/output layout exactly
- `postprocess_output` matches upstream resize, hole-fill, non-overlap, and suppression exactly

## Stage Gates

The strict port may only proceed through these stages in order.

### Stage 0: Export completeness

Required before any runtime tracker work resumes:

- the full upstream export matrix above is either generated or explicitly marked as a blocker
- missing export fields are added before any dependent Rust implementation begins

### Stage 1: Builder/config parity

Required to pass before any tracker-runtime implementation:

- all builder/config fixture-backed tests green
- both `apply_temporal_disambiguation` branches fixture-backed

### Stage 2: Frame-0 prompt and tracker-seed parity

Required to pass before frame-1 propagation:

- box prompt
- point prompt
- mask prompt
- correction-click seed path if applicable

### Stage 3: Frame-1 propagation parity

Required to pass before frame-2 propagation:

- default bundle
- temporal-disambiguation bundle
- multimask-eligible point bundle

### Stage 4: Frame-2/frame-3 long-history parity

Required to pass before multi-object and postprocess branches:

- long-history bundle(s)
- object-pointer-limit bundle
- keep-first-conditioning bundle
- stride>1 bundle

### Stage 5: Predictor-wrapper and postprocess parity

Required to pass before replacing the strict-port scaffold with the real runtime path:

- multi-object bundle
- memory-clear bundle
- output non-overlap bundle
- fill-hole bundle
- reverse propagation bundle

### Stage 6: Default enablement

Required before the strict port becomes the default runtime path:

- all required bundles green
- no source-derived-only branch remains for reachable SAM3 runtime behavior

## Stop Conditions

Implementation must stop and report immediately when:

- a required upstream branch is not exported but affects runtime semantics
- an exported tensor is insufficient to reconstruct a local contract exactly
- a reachable SAM3 runtime branch has only source-derived coverage
- a local implementation decision would require a heuristic or "equivalent" substitute
