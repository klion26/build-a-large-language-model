use crate::Exercise;

/// 2.1
pub struct X2P1 {}

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
pub struct X2P2 {}

impl Exercise for X2P2 {
    fn name(&self) -> String {
        String::from("2.2")
    }

    fn main(&self) {
        todo!()
    }
}
