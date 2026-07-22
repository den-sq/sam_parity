use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_transformers::models::sam3::TextPromptTokens;
use tokenizers::{PaddingDirection, PaddingParams, Tokenizer, TruncationParams};

const CLIP_EOT_TOKEN: &str = "<|endoftext|>";

/// Runtime/parity adapter for SAM3's CLIP tokenizer contract.
///
/// Tokenizer files and string processing intentionally live outside
/// candle-transformers; the model boundary receives IDs and masks only.
pub struct Sam3Tokenizer {
    tokenizer: Tokenizer,
}

impl Sam3Tokenizer {
    pub fn from_path(path: impl AsRef<Path>, context_length: usize) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer_path: PathBuf = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_owned()
        };
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!("failed to load tokenizer from {}", tokenizer_path.display())
            })?;
        let pad_id = *tokenizer
            .get_vocab(true)
            .get(CLIP_EOT_TOKEN)
            .with_context(|| format!("tokenizer is missing required token {CLIP_EOT_TOKEN}"))?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(context_length),
                direction: PaddingDirection::Right,
                pad_to_multiple_of: None,
                pad_id,
                pad_type_id: 0,
                pad_token: CLIP_EOT_TOKEN.to_owned(),
            }))
            .with_truncation(Some(TruncationParams {
                max_length: context_length,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<TextPromptTokens> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        Ok(TextPromptTokens::new(
            encoding.get_ids().to_vec(),
            encoding.get_attention_mask().to_vec(),
        )
        .with_display_text(text))
    }
}

#[cfg(test)]
mod tests {
    use super::Sam3Tokenizer;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Corpus {
        context_length: usize,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        text: String,
        input_ids: Vec<u32>,
        attention_mask: Vec<u32>,
    }

    #[test]
    fn facebook_sam3_tokenizer_matches_golden_corpus() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("examples/assets/text/tokenizer.json");
        let corpus: Corpus = serde_json::from_slice(
            &std::fs::read(root.join("tests/data/sam3_tokenizer_golden.json")).unwrap(),
        )
        .unwrap();
        let tokenizer = Sam3Tokenizer::from_path(path, corpus.context_length).unwrap();
        for case in corpus.cases {
            let tokens = tokenizer.encode(&case.text).unwrap();
            assert_eq!(
                tokens.input_ids, case.input_ids,
                "input IDs for {:?}",
                case.text
            );
            assert_eq!(
                tokens.attention_mask, case.attention_mask,
                "attention mask for {:?}",
                case.text
            );
        }
    }
}
