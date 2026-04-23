#[derive(Debug, Clone, PartialEq)]
pub struct TensorComparisonReport {
    pub expected_shape: Vec<usize>,
    pub actual_shape: Vec<usize>,
    pub max_abs_diff: Option<f32>,
    pub max_abs_diff_flat_index: Option<usize>,
    pub expected_at_max_abs_diff: Option<f32>,
    pub actual_at_max_abs_diff: Option<f32>,
    pub mean_abs_diff: Option<f32>,
    pub rmse: Option<f32>,
    pub pass: bool,
    pub note: Option<String>,
}

