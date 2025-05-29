use candle_core::{Device, Result, Tensor};
use std::collections::HashSet;
use tiktoken_rs::CoreBPE;

/// listing 0501
pub fn text_token_ids(text: &str, tokenizer: &CoreBPE, dev: &Device) -> Result<Tensor> {
    let allowed_special = HashSet::from(["<|endoftext|>"]);
    let encoded = tokenizer.encode(text, allowed_special);
    let num_tokens = encoded.len();

    // encode tensor
    Tensor::from_vec(encoded, (1_usize, num_tokens), dev)
}

pub fn token_ids_to_text(token_ids: Tensor, tokenizer: &CoreBPE) -> anyhow::Result<String> {
    let flat = token_ids.squeeze(0)?;
    tokenizer.decode(flat.to_vec1::<u32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use rstest::*;
    use tiktoken_rs::get_bpe_from_model;

    #[fixture]
    pub fn txt_tokenizer() -> (String, CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
    }

    #[fixture]
    pub fn device() -> Device {
        Device::cuda_if_available(0).unwrap()
    }

    #[rstest]
    fn test_text_token_ids_and_back_to_text(
        #[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE),
        #[from(device)] device: Device,
    ) {
        let token_ids = text_token_ids(&txt[..], &tokenizer, &device);
        let decode_text = token_ids_to_text(token_ids.unwrap(), &tokenizer).unwrap();
        assert_eq!(txt, decode_text);
    }
}
