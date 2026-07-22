# CANDLE-2 tokenizer externalization

This change moves SAM3 string tokenization out of candle-transformers. The
model crate keeps its existing tensor boundary (Sam3ImageModel::encode_text_tokens)
and the video API now accepts TextPromptTokens, containing caller-supplied token
IDs and attention masks.

## Migration

Previously, video callers supplied a string in SessionPrompt.text and a
filesystem path in VideoSessionOptions.tokenizer_path. Callers must now
tokenize before entering the model crate:

    let tokenizer = Sam3Tokenizer::from_path(
        path,
        model.config().text.context_length,
    )?;
    let person = tokenizer.encode("person")?;
    let visual = tokenizer.encode("visual")?;

    let options = VideoSessionOptions {
        visual_prompt_tokens: Some(visual),
        ..Default::default()
    };
    let prompt = SessionPrompt {
        text: Some(person),
        /* geometry fields unchanged */
    };

sam3-parity-cli::tokenization::Sam3Tokenizer is the reference runtime adapter.
It preserves the Facebook SAM3 CLIP contract: special tokens are enabled,
sequences are truncated to the model context length, and right padding uses the
CLIP end-of-text token. Candle examples contain an equivalent adapter so string
convenience remains available without adding tokenizer or Oniguruma dependencies
to candle-transformers.

Box-only prompts implicitly used the string visual before this migration. That
sentinel is now explicit as VideoSessionOptions.visual_prompt_tokens. Point-only
consumers do not need a tokenizer or sentinel.

Text-encoding cache keys use only token IDs and attention masks. Optional
display_text is diagnostic metadata and cannot change cache identity.

## Certification

The checked-in corpus at tests/data/sam3_tokenizer_golden.json covers the visual
sentinel, a common text prompt, a medical phrase, empty input, and non-ASCII
input against the Facebook SAM3 tokenizer. Core tests cover token validation,
stable cache identity, cache cleanup, and CPU offload accounting. Existing
image, mixed text/geometry, and video parity tests continue to exercise the
unchanged tensor/model boundary.
