use crate::Example;

/// Example 03.01
pub struct EG01 {}

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Computing attention scores as a dot product.")
    }

    fn page_source(&self) -> usize {
        57_usize
    }

    fn main(&self) {
        use candle_core::{Device, IndexOp, Tensor};
        use candle_nn::ops::softmax;

        let dev = Device::cuda_if_available(0).unwrap();
        let inputs = Tensor::new(
            &[
                [0.43_f32, 0.15, 0.89], // Your
                [0.55, 0.87, 0.66],     // journey
                [0.57, 0.85, 0.64],     // starts
                [0.22, 0.58, 0.33],     // with
                [0.77, 0.25, 0.10],     // one
                [0.05, 0.80, 0.55],     // step
            ],
            &dev,
        )
        .unwrap();

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
            }

            println!("Context vector 2: {:?}", context_vec_2.to_vec2::<f32>());
        }
    }
}
