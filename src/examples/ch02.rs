use crate::listings::ch02::create_dataloader_v1;
use crate::Example;

pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Use candel to generate an Embedding layer.")
    }

    fn page_source(&self) -> usize {
        42_usize
    }

    fn main(&self) {
        use candle_core::{DType, Device};
        use candle_nn::{embedding, VarBuilder, VarMap};

        let vocab_size = 6_usize;
        let output_dim = 3_usize;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let emb = embedding(vocab_size, output_dim, vs).unwrap();

        println!("{:?}", emb.embeddings().to_vec2::<f32>());
    }
}

pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Create absolute postitional embeddings.")
    }

    fn page_source(&self) -> usize {
        47_usize
    }

    fn main(&self) {
        use candle_core::{DType, Tensor};
        use candle_nn::{embedding, VarBuilder, VarMap};
        use std::fs;

        // create data batcher
        let raw_text = fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let max_length = 4_usize;
        let stride = max_length;
        let shuffle = false;
        let drop_last = false;
        let batch_size = 8_usize;
        let (_dataset, mut batch_iter) = create_dataloader_v1(
            &raw_text[..],
            batch_size,
            max_length,
            stride,
            shuffle,
            drop_last,
        );

        // get embeddings of first batch inputs
        match batch_iter.next() {
            Some(Ok((inputs, _target))) => {
                let varmap = VarMap::new();
                let dev = inputs.device();
                let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);

                let vocab_size = 50_257_usize;
                let output_dim = 256_usize;

                let mut final_dims = inputs.dims().to_vec();
                final_dims.push(output_dim);

                let token_embedding_layer =
                    embedding(vocab_size, output_dim, vs.pp("tok_emb")).unwrap();
                let token_embeddings = token_embedding_layer
                    .embeddings()
                    .index_select(&inputs.flatten_all().unwrap(), 0)
                    .unwrap();
                let token_embeddings = token_embeddings.reshape(final_dims).unwrap();
                println!(
                    "token embeddings dim: {:?}, {:?}",
                    token_embeddings.dims(),
                    token_embeddings
                );

                let context_length = max_length;
                let pos_embedding_layer =
                    embedding(context_length, output_dim, vs.pp("pos_emb")).unwrap();

                let pos_ids = Tensor::arange(0_u32, context_length as u32, dev).unwrap();

                let pos_embeddings = pos_embedding_layer
                    .embeddings()
                    .index_select(&pos_ids, 0)
                    .unwrap();

                println!("pos embeddings dims: {:?}", pos_embeddings.dims());

                let input_embeddings = token_embeddings.broadcast_add(&pos_embeddings).unwrap();
                println!("input embeddings dims: {:?}", input_embeddings.dims());
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }
}
