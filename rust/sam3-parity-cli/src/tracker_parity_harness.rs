#![allow(dead_code)]
#![allow(unused_imports)]

mod tracker_parity {
    use anyhow::{Context, Result};
    use candle::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::sam3;
    use candle_transformers::models::sam3::parity_support::*;

    use crate::full_parity_support::*;
    use crate::paths;

    include!("tracker_parity.rs");
}
