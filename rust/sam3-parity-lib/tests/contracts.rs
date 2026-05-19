use sam3_parity_lib::contracts::{
    InteractiveReferenceMetadata, InteractiveReferenceStepMetadata, ParityBundleMetadata,
    VideoExportMetadata,
};

#[test]
fn parity_bundle_metadata_round_trips() {
    let metadata = ParityBundleMetadata {
        bundle_version: 1,
        image_path: Some("image.png".to_owned()),
        prompt: Some("shoe".to_owned()),
        effective_prompt: Some("shoe".to_owned()),
        boxes_cxcywh: vec![vec![0.5, 0.5, 0.25, 0.25]],
        box_labels: vec![true],
        image_size: Some(1008),
        preprocess_mode: Some("exact".to_owned()),
        stage_order: vec!["text.memory".to_owned(), "decoder.pred_logits".to_owned()],
    };
    let json = serde_json::to_string(&metadata).expect("serialize parity metadata");
    let decoded: ParityBundleMetadata =
        serde_json::from_str(&json).expect("deserialize parity metadata");
    assert_eq!(decoded, metadata);
}

#[test]
fn interactive_reference_metadata_round_trips() {
    let metadata = InteractiveReferenceMetadata {
        bundle_version: 1,
        image_path: "image.png".to_owned(),
        image_size: Some(1008),
        preprocess_mode: Some("exact".to_owned()),
        replay_script_path: Some("interactive.json".to_owned()),
        checkpoint_path: Some("sam3.pt".to_owned()),
        bpe_path: Some("tokenizer.json".to_owned()),
        steps: vec![InteractiveReferenceStepMetadata {
            name: Some("seed".to_owned()),
            step_points_xy_normalized: vec![vec![0.5, 0.5]],
            step_point_labels: vec![1],
            accumulated_points_xy_normalized: vec![vec![0.5, 0.5]],
            accumulated_point_labels: vec![1],
        }],
    };
    let json = serde_json::to_string(&metadata).expect("serialize interactive metadata");
    let decoded: InteractiveReferenceMetadata =
        serde_json::from_str(&json).expect("deserialize interactive metadata");
    assert_eq!(decoded, metadata);
}

#[test]
fn video_export_metadata_round_trips() {
    let metadata = VideoExportMetadata {
        bundle_version: 1,
        mode: "video_debug_bundle".to_owned(),
        source_path: "video.mp4".to_owned(),
        source_kind: "video".to_owned(),
        session_frame_count: 16,
        exported_frame_count: 8,
        frame_stride: 2,
        tokenizer_path: Some("tokenizer.json".to_owned()),
        prompt_text: Some("person".to_owned()),
        points_xy_normalized: vec![vec![0.25, 0.75]],
        point_labels: vec![1],
        boxes_cxcywh_normalized: vec![vec![0.5, 0.5, 0.2, 0.3]],
        box_labels: vec![1],
        frames_dir: "frames".to_owned(),
        masks_dir: "masks".to_owned(),
        masked_frames_dir: "masked_frames".to_owned(),
        results_path: "video_results.json".to_owned(),
        debug_dir: Some("debug".to_owned()),
    };
    let json = serde_json::to_string(&metadata).expect("serialize video metadata");
    let decoded: VideoExportMetadata =
        serde_json::from_str(&json).expect("deserialize video metadata");
    assert_eq!(decoded, metadata);
}
