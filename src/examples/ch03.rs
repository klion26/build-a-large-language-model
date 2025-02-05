use crate::Example;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::init::DEFAULT_KAIMING_NORMAL;
use candle_nn::ops::softmax;

fn get_inputs() -> Tensor {
    Tensor::new(
        &[
            [0.43_f32, 0.15, 0.89], // Your
            [0.55, 0.87, 0.66],     // journey
            [0.57, 0.85, 0.64],     // starts
            [0.22, 0.58, 0.33],     // with
            [0.77, 0.25, 0.10],     // one
            [0.05, 0.80, 0.55],     // step
        ],
        &Device::Cpu,
    )
    .unwrap()
}
/// Example 03.01
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Computing attention scores as a dot product.")
    }

    fn page_source(&self) -> usize {
        57_usize
    }

    fn main(&self) {
        let inputs = get_inputs();
        let dev = inputs.device().to_owned();

        let query = inputs
            .index_select(&Tensor::new(&[1u32], &dev).unwrap(), 0)
            .unwrap();

        // compute attention scores
        let mut attention_scores: Option<Tensor> = None;
        for i in 0..inputs.dims()[0] {
            let x_i = inputs
                .index_select(&Tensor::new(&[i as u32], &dev).unwrap(), 0)
                .unwrap();
            let a_i = x_i
                .matmul(&query.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap();
            attention_scores = match attention_scores {
                None => Some(a_i),
                Some(prev_attention_scores) => {
                    Some(Tensor::cat(&[&prev_attention_scores, &a_i], 0).unwrap())
                }
            }
        }

        if let Some(attention_scores) = attention_scores {
            println!("Raw attention scores: {:?}", attention_scores);

            let sum = attention_scores.sum_all().unwrap();
            let normalized_attention_scores = attention_scores
                .broadcast_div(&sum)
                .unwrap()
                .to_vec1::<f32>();
            println!("{:?}", normalized_attention_scores);

            // softmax normalization
            let exponentiator = attention_scores.exp().unwrap();

            let exponentiator_sum = exponentiator.sum_all().unwrap();
            let softmax_attention_scores = exponentiator.broadcast_div(&exponentiator_sum).unwrap();
            println!(
                "Naive Softmax-normalized attention scores  {:?}",
                softmax_attention_scores
            );

            // candle softmax
            let softmax_attention_scores_candle = softmax(&attention_scores, 0).unwrap();
            println!(
                "Candle Softmax-normalized attention scores {:?}",
                softmax_attention_scores_candle
            );

            // compute second context vector
            let mut context_vec_2 = Tensor::zeros_like(&query).unwrap();
            for i in 0..inputs.dims()[0] {
                let x_i = inputs
                    .index_select(&Tensor::new(&[i as u32], &dev).unwrap(), 0)
                    .unwrap();
                context_vec_2 = context_vec_2
                    .add(
                        &x_i.broadcast_mul(&softmax_attention_scores.i(i).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
                println!(
                    "x_i {:?}, {:?}, {:?}",
                    x_i.to_vec2::<f32>(),
                    softmax_attention_scores_candle.i(i).unwrap(),
                    context_vec_2.to_vec2::<f32>()
                );
            }

            println!("Context vector 2: {:?}", context_vec_2.to_vec2::<f32>());
        }
    }
}

pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Manual computation of multiple context vectors simulataneously.")
    }

    fn page_source(&self) -> usize {
        62_usize
    }

    fn main(&self) {
        let inputs = get_inputs();

        // matmul to get attn scores
        let attn_scores = inputs.matmul(&inputs.t().unwrap()).unwrap();

        println!("attn_scores: {:?}", attn_scores);

        let attn_weights = softmax(&attn_scores, 1).unwrap();

        // check sum of each row
        let sum = attn_weights.sum(1).unwrap();

        let all_context_vectors = attn_weights.matmul(&inputs).unwrap();

        println!("Attention Weights: {:?}", attn_weights.to_vec2::<f32>());
        println!("All Rows Sum: {:?}\n\n", sum.flatten_all());
        println!(
            "Context vectors: {:?}",
            all_context_vectors.to_vec2::<f32>()
        );
    }
}

pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        let desc = "Implementing the self-attention mechanism with \
        trainable weights to compute single context vector.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        66_usize
    }

    fn main(&self) {
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let dev = inputs.device().to_owned();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, inputs.dtype(), &dev);

        let x_2 = inputs
            .index_select(&Tensor::new(&[1u32], &dev).unwrap(), 0)
            .unwrap();
        let d_in = x_2.dims()[1]; // input embedding dimension
        let d_out = 2_usize;

        // projections
        let init = DEFAULT_KAIMING_NORMAL;
        let w_query = vs.get_with_hints((d_in, d_out), "query", init).unwrap();
        let w_key = vs.get_with_hints((d_in, d_out), "key", init).unwrap();
        let w_value = vs.get_with_hints((d_in, d_out), "value", init).unwrap();

        // query , key, value vectors
        let query_2 = x_2.matmul(&w_query).unwrap();
        let key_2 = x_2.matmul(&w_key).unwrap();
        let value_2 = x_2.matmul(&w_value).unwrap();

        println!("Query 2 {:?}", query_2.to_vec2::<f32>());
        println!("Key  2 {:?}", key_2.to_vec2::<f32>());
        println!("Value 2 {:?}", value_2.to_vec2::<f32>());

        let keys = inputs.matmul(&w_key).unwrap();
        let values = inputs.matmul(&w_value).unwrap();

        println!("Keys shape:  {:?}", keys);
        println!("Values shae: {:?}", values);

        let attn_scores = query_2.matmul(&keys.t().unwrap()).unwrap();
        println!("Attn scores: {:?}", attn_scores.to_vec2::<f32>());

        let d_k = Tensor::new(&[f32::powf(keys.dims()[1] as f32, 0.5_f32)], &dev).unwrap();
        let attn_weights = softmax(&attn_scores.broadcast_div(&d_k).unwrap(), 1).unwrap();
        println!("Attn weights: {:?}", attn_weights.to_vec2::<f32>());

        let context_vec_2 = attn_weights.matmul(&values).unwrap();
        println!("Context vector 2: {:?}", context_vec_2.to_vec2::<f32>());
    }
}

/// Example 03.04
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from(
            "Implement self-attention mechanism to compute context vectors in the input sequence.",
        )
    }

    fn page_source(&self) -> usize {
        71_usize
    }

    fn main(&self) {
        use crate::listings::ch03::SelfAttentionV1;
        use candle_core::{DType, Module};
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // construct self attention layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        // run a random, embedded input sequence through self-attention
        let input_length = inputs.dims()[0];
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &Device::Cpu).unwrap();
        let context_vectors = attn_v1_layer.forward(&xs).unwrap();

        println!("context vectors: {:?}", context_vectors.to_vec2::<f32>());
    }
}
