# SAM3 Parity Support Accessor Plan

Last updated: 2026-04-28

## Goal

Add a small, feature-gated accessor layer in Candle so the external
`sam_parity` repo can compile and run the extracted `tracker_parity.rs` and
`video_parity.rs` sources without:

- copying large chunks of Candle internals into the parity repo
- widening SAM3 internals in normal builds
- changing SAM3 runtime behavior

The intended feature gate already exists in Candle:

- `candle-transformers/Cargo.toml`: `sam3-parity-support = []`
- `candle-transformers/src/models/sam3/mod.rs`:
  `#[cfg(feature = "sam3-parity-support")] pub mod parity_support;`

## Why This Is Needed

The external parity crate currently preserves the extracted Rust test bodies:

- `rust/sam3-parity-cli/src/tracker_parity.rs`
- `rust/sam3-parity-cli/src/video_parity.rs`
- `rust/sam3-parity-cli/src/tracker_parity_support.rs`
- `rust/sam3-parity-cli/src/video_parity_support.rs`

Those files are intentionally parked today because they still reach into private
SAM3 tracker/video internals that were accessible when the tests lived
in-tree.

The real requirement is not "make internals public everywhere." It is "add a
thin parity-only shim so the external test harness can ask Candle for the same
intermediate results that the old in-tree tests already exercised."

## Design Constraints

- No model-logic changes.
- No behavior changes when `sam3-parity-support` is disabled.
- Keep the support surface narrow and test-oriented.
- Prefer wrappers and parity view types over exposing raw private structs.
- Keep the external parity repo responsible for fixture loading and assertions.
  Candle should only expose the minimum execution and inspection hooks.

## Current Compile Blockers

These are the private or `pub(super)` items the parked external harness still
needs.

### Tracker

Directly used by `tracker_parity.rs`:

- `Sam3TrackerModel::get_tpos_enc`
- `Sam3TrackerModel::use_multimask`
- `Sam3TrackerModel::use_mask_as_output`
- `Sam3TrackerModel::prepare_memory_conditioned_features`

Used by `tracker_parity_support.rs`:

- `Sam3TrackerModel::prepare_high_res_features`
- tracker compute dtype via `self.no_obj_ptr.dtype()`
- `Sam3TrackerModel::forward_sam_heads`
- `Sam3TrackerModel::build_memory_conditioning_prompt`
- `self.memory_transformer.forward(...)`

### Video

Directly used by `video_parity.rs`:

- `Sam3VideoPredictor.video_config`
- `Sam3VideoPredictor.sessions`
- `Sam3VideoTrackerCore::process_frame`
- `ObjectFrameOutput::score_value`
- `Sam3VideoSession.tracked_objects`
- `Sam3VideoSession.frame_outputs`
- `Sam3VideoSession.temporal_disambiguation_metadata`

## Recommended Shape

Keep `models::sam3::parity_support` as the public entry point, but do not make
it the place that reaches across private module boundaries.

Instead:

1. Add parity-only wrappers next to the private internals they delegate to.
2. Re-export those wrappers and parity view types from
   `src/models/sam3/parity_support.rs`.
3. Update the external `sam_parity` harness to use those accessors instead of
   field access or private method calls.

That avoids broadening core visibility just so a sibling module can call into
tracker/video internals.

## Suggested Candle Landing Spots

- `candle-transformers/src/models/sam3/parity_support.rs`
  - public re-exports only
- `candle-transformers/src/models/sam3/tracker/model.rs`
  - temporal-position and compute-dtype parity wrappers
- `candle-transformers/src/models/sam3/tracker/prompt_inputs.rs`
  - multimask and high-res feature parity wrappers
- `candle-transformers/src/models/sam3/tracker/sam_heads.rs`
  - SAM-head and mask-as-output parity wrappers
- `candle-transformers/src/models/sam3/tracker/memory_conditioning.rs`
  - memory-conditioning and prompt reconstruction parity wrappers
- `candle-transformers/src/models/sam3/video/propagation.rs`
  - `process_frame` and `score_value` parity wrappers
- `candle-transformers/src/models/sam3/video/session.rs`
  - session object/output accessors
- `candle-transformers/src/models/sam3/video/temporal_disambiguation.rs`
  - conversion from internal metadata to a parity-safe public view

## Phase 1: Minimum Support To Activate The Parked Harness

Phase 1 should be limited to the accessors needed by the currently preserved
external sources. This is the smallest useful patch.

### Tracker API

Recommended parity-only surface:

```rust
pub trait Sam3TrackerParityExt {
    fn parity_compute_dtype(&self) -> DType;

    fn parity_prepare_high_res_features(
        &self,
        high_res_features: &[Tensor],
    ) -> Result<Vec<Tensor>>;

    fn parity_use_multimask(
        &self,
        is_init_cond_frame: bool,
        point_count: usize,
    ) -> bool;

    fn parity_get_tpos_enc(
        &self,
        rel_pos_list: &[i64],
        device: &Device,
        max_abs_pos: Option<usize>,
        dummy: bool,
    ) -> Result<Tensor>;

    fn parity_forward_sam_heads(
        &self,
        backbone_features: &Tensor,
        point_prompt: Option<&(Tensor, Tensor)>,
        mask_inputs: Option<&Tensor>,
        high_res_features: Option<&[Tensor]>,
        multimask_output: bool,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState>;

    fn parity_use_mask_as_output(
        &self,
        backbone_features: &Tensor,
        high_res_features: Option<&[Tensor]>,
        mask_inputs: &Tensor,
        is_cond_frame: bool,
    ) -> Result<TrackerFrameState>;

    fn parity_prepare_memory_conditioned_features(
        &self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_vision_feats: &[Tensor],
        current_vision_pos_embeds: &[Tensor],
        feat_sizes: &[(usize, usize)],
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        use_prev_mem_frame: bool,
        packed_history: Option<&PackedPromptHistory>,
    ) -> Result<ParityPreparedMemoryConditioning>;

    fn parity_build_memory_conditioning_prompt(
        &self,
        frame_idx: usize,
        history: &BTreeMap<usize, TrackerFrameState>,
        num_frames: usize,
        track_in_reverse: bool,
        packed_history: Option<&PackedPromptHistory>,
    ) -> Result<ParityPreparedMemoryPrompt>;

    fn parity_memory_transformer_forward(
        &self,
        src: &Tensor,
        prompt: &Tensor,
        src_pos: Option<&Tensor>,
        prompt_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Result<Tensor>;
}
```

Recommended parity-only view types:

```rust
pub struct ParityPreparedMemoryConditioning {
    pub pix_feat_with_mem: Tensor,
    pub selected_conditioning_frame_indices: Vec<usize>,
    pub selected_memory_frame_indices: Vec<usize>,
    pub selected_object_pointer_frame_indices: Vec<usize>,
}

pub struct ParityPreparedMemoryPrompt {
    pub prompt: Option<Tensor>,
    pub prompt_pos: Option<Tensor>,
    pub num_obj_ptr_tokens: usize,
    pub selected_conditioning_frame_indices: Vec<usize>,
    pub selected_memory_frame_indices: Vec<usize>,
    pub selected_object_pointer_frame_indices: Vec<usize>,
}
```

Notes:

- Do not expose raw `PreparedMemoryConditioning` or `PreparedMemoryPrompt`.
  They are internal implementation structs and do not need to become part of
  Candle's normal SAM3 surface.
- `prepare_high_res_features_for_test` already exists under `#[cfg(test)]`.
  Phase 1 can either:
  - widen that helper to `#[cfg(feature = "sam3-parity-support")]`, or
  - add a new parity wrapper that delegates to `prepare_high_res_features(...)`.

### Video API

Recommended parity-only surface:

```rust
pub trait Sam3VideoPredictorParityExt {
    fn parity_video_config(&self) -> &VideoConfig;
    fn parity_video_config_mut(&mut self) -> &mut VideoConfig;
    fn parity_session(&self, session_id: &str) -> Option<&Sam3VideoSession>;
    fn parity_session_mut(&mut self, session_id: &str) -> Option<&mut Sam3VideoSession>;
}

pub trait Sam3VideoSessionParityExt {
    fn parity_tracked_objects(&self) -> &BTreeMap<u32, TrackedObject>;
    fn parity_tracked_objects_mut(&mut self) -> &mut BTreeMap<u32, TrackedObject>;
    fn parity_frame_outputs(&self) -> &BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>;
    fn parity_frame_outputs_mut(
        &mut self,
    ) -> &mut BTreeMap<usize, BTreeMap<u32, ObjectFrameOutput>>;
    fn parity_temporal_disambiguation_metadata(
        &self,
    ) -> BTreeMap<usize, ParityTemporalDisambiguationFrameMetadata>;
}

pub trait Sam3VideoTrackerCoreParityExt {
    fn parity_process_frame(
        &self,
        model: &Sam3ImageModel,
        compute_device: &Device,
        config: &VideoConfig,
        session: &mut Sam3VideoSession,
        frame_idx: usize,
        direction: PropagationDirection,
        output_threshold: f32,
    ) -> Result<VideoFrameOutput>;
}

pub trait ObjectFrameOutputParityExt {
    fn parity_score_value(&self) -> Result<f32>;
}
```

Recommended parity-only view type:

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParityTemporalDisambiguationFrameMetadata {
    pub removed_obj_ids: BTreeSet<u32>,
    pub suppressed_obj_ids: BTreeSet<u32>,
    pub unconfirmed_obj_ids: BTreeSet<u32>,
    pub matched_obj_ids: BTreeSet<u32>,
    pub unmatched_obj_ids: BTreeSet<u32>,
}
```

Notes:

- `TrackedObject`, `TrackerFrameState`, `ObjectFrameOutput`, and
  `VideoFrameOutput` are already public, so the parity layer only needs to
  expose session access to them.
- The temporal-disambiguation metadata should be returned as a parity-safe copy
  or mirror type. There is no need to make the internal
  `TemporalDisambiguationFrameMetadata` itself part of the normal public model
  API.

## Phase 1 Non-Goals

These accessors were previously mentioned as possible needs, but they do not
appear to be required by the currently parked external harness:

- `Sam3VideoTrackerCore::clear_non_cond_mem_around_input`
- `Sam3VideoTrackerCore::postprocess_output`

Those should stay out of the first patch unless a concrete external parity test
needs them. Keeping them out makes the initial Candle change smaller and easier
to review.

## Estimated Scope

This is still a narrow Candle patch.

Roughly:

- about 13-15 thin wrapper/accessor methods
- 3 parity-only view structs
- no checkpoint mapping changes
- no tensor math changes
- no runtime-path changes when the feature is off

The most invasive part is not the code volume. It is placing each wrapper in
the same module that owns the private implementation so the wrapper can safely
delegate without broad visibility changes.

## Implementation Strategy

1. Add parity-only wrapper methods or extension-trait impls in the owning
   tracker/video modules.
2. Add parity-only view structs and conversion helpers for memory-prompt and
   temporal-disambiguation outputs.
3. Re-export the traits and view types from `models::sam3::parity_support`.
4. Update `sam_parity` to replace:
   - direct field access such as `predictor.sessions`
   - direct calls to private methods such as `process_frame(...)`
   with the new parity-support entry points.
5. Re-enable the parked harness modules in `sam_parity` behind
   `--features full-parity`.

## Acceptance Criteria

The accessors are complete when all of the following are true:

1. `sam_parity` can compile the preserved `tracker_parity.rs` and
   `video_parity.rs` sources against Candle without editing away the original
   test intent.
2. The external parity repo no longer reaches into private Candle fields or
   private methods directly.
3. Candle exposes no new SAM3 public surface when `sam3-parity-support` is
   disabled.
4. There are no model-output or runtime-behavior changes in existing Candle
   tests caused by the accessor patch.
5. The external harness can run under:

```bash
cargo test -p sam3-parity-cli --features full-parity --no-run
```

and then proceed to fixture-backed execution as artifacts are available.
