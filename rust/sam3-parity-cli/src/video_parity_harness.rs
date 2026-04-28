#![allow(dead_code)]
#![allow(unused_imports)]

mod video_parity {
    use candle::{DType, Device, IndexOp, Result, Tensor};
    use candle_transformers::models::sam3;
    use candle_transformers::models::sam3::parity_support::*;

    use crate::full_parity_support::*;
    use crate::paths;

    include!("video_parity_support.rs");
}
