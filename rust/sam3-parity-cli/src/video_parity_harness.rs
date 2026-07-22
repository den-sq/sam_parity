#![allow(dead_code)]
#![allow(unused_imports)]

mod video_parity {
    use candle::{DType, Device, IndexOp, Result, Tensor};
    use candle_examples::sam3_video::MediaFrameSource;
    use candle_transformers::models::sam3;
    use candle_transformers::models::sam3::parity_support::*;

    use crate::full_parity_support::*;
    use crate::paths;
    use crate::tokenization::Sam3Tokenizer;

    include!("video_parity_support.rs");
}
