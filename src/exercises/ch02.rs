use crate::Exercise;

/// 2.1
pub struct X2P1;

impl Exercise for X2P1 {
    fn name(&self) -> String {
        String::from("2.1")
    }

    fn main(&self) {
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        let text =
            "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
        let token_ids = tokenizer.encode_with_special_tokens(text);
        println!("token ids: {:?}", token_ids);

        let decoded_text = tokenizer.decode(token_ids).unwrap();
        println!("decode text: {:?}", decoded_text);
    }
}

/// 2.2
pub struct X2P2;

impl Exercise for X2P2 {
    fn name(&self) -> String {
        String::from("2.2")
    }

    fn main(&self) {
        use crate::listings::ch02::{GPTDatasetIter, GPTDatasetV1};
        use candle_core::Device;
        use candle_datasets::Batcher;
        use std::fs;
        use tiktoken_rs::get_bpe_from_model;

        let raw_text = fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let tokenzier = get_bpe_from_model("gpt2").unwrap();
        let max_length = 4_usize;
        let stride = 2_usize;
        let dataset = GPTDatasetV1::new(&raw_text[..], tokenzier, max_length, stride);
        let device = Device::Cpu;
        let iter = GPTDatasetIter::new(&dataset, device, false);
        let batch_size = 2_usize;
        let mut batch_iter = Batcher::new_r2(iter).batch_size(batch_size);

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                println!(
                    "inputs: {:?}\n\ntargets: {:?}",
                    inputs.to_vec2::<u32>(),
                    targets.to_vec2::<u32>(),
                );
            }
            Some(Err(err)) => panic!("Error: {:?}", err),
            None => panic!("None"),
        }
    }
}
