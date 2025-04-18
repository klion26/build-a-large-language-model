use crate::listings::ch04::ExampleDeepNeuralNetwork;
use crate::Example;
use candle_core::{Error, Module, Tensor};

use anyhow::Result;

pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Getting logits with DummyGPTModel")
    }

    fn page_source(&self) -> usize {
        97_usize
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, DummyGPTModel};
        use candle_core::{DType, Device, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        let dev = Device::cuda_if_available(0).unwrap();

        // create batch
        let mut batch_tokens: Vec<u32> = Vec::new();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every effort moves you"));
        batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every day holds a"));

        let batch = Tensor::from_vec(batch_tokens, (2_usize, 4_usize), &dev).unwrap();

        println!("batch: {:?}", batch.to_vec2::<u32>());

        // create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = DummyGPTModel::new(Config::gpt2_124m(), vb).unwrap();

        // get logits
        let logits = model.forward(&batch).unwrap();
        println!("logits: {:?}", logits.to_vec3::<f32>());
        println!("output shape: {:?}", logits.shape());
    }
}

/// Example 04.02
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Manual computation of layern normalization")
    }

    fn page_source(&self) -> usize {
        100_usize
    }

    fn main(&self) {
        use candle_core::{DType, Device, Module, Tensor, D};
        use candle_nn::{linear_b, seq, Activation, VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // create batch
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 5_usize), vb.device()).unwrap();

        // create layer
        let layer = seq()
            .add(linear_b(5_usize, 6_usize, false, vb.pp("linear1")).unwrap())
            .add(Activation::Relu);

        // execute layer on batch
        let out = layer.forward(&batch_example).unwrap();
        println!("out: {:?}", out.to_vec2::<f32>());

        // calculate stats on outputs
        let mean = out.mean_keepdim(D::Minus1).unwrap();
        let var = out.var_keepdim(D::Minus1).unwrap();
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());

        // layer normalization
        let out_norm = (out
            .broadcast_sub(&mean)
            .unwrap()
            .broadcast_div(&var.sqrt().unwrap()))
        .unwrap();
        let mean = out_norm.mean_keepdim(D::Minus1).unwrap();
        let var = out_norm.var_keepdim(D::Minus1).unwrap();
        println!("normalizaed out: {:?}", out_norm.to_vec2::<f32>());
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());
    }
}

/// Example 04.03
pub struct EG03;
impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Example usage of `LayerNorm`.")
    }

    fn page_source(&self) -> usize {
        104_usize
    }

    fn main(&self) {
        use crate::listings::ch04::LayerNorm;
        use candle_core::{DType, Device, Module, Tensor, D};
        use candle_nn::{VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // create batch
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 5_usize), vb.device()).unwrap();

        // construct layer norm layer
        let emb_dim = 5_usize;
        let ln = LayerNorm::new(emb_dim, vb.pp("layer_norm")).unwrap();
        let out_ln = ln.forward(&batch_example).unwrap();

        // compute stats on out_ln
        let mean = out_ln.mean_keepdim(D::Minus1).unwrap();
        let var = out_ln.var_keepdim(D::Minus1).unwrap();
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());
    }
}

/// Example 04.05
pub struct EG05;

impl EG05 {
    fn print_gradients(model: ExampleDeepNeuralNetwork, x: &Tensor) -> Result<()> {
        use candle_nn::loss::mse;

        let output = model.forward(x)?;
        let target = Tensor::new(&[[0_f32]], x.device())?;

        let loss = mse(&output, &target)?;
        let grads = loss.backward()?;

        for (ix, tensor_id) in model.tensor_ids.iter().enumerate() {
            let grad_tensor = grads.get_id(tensor_id.to_owned()).ok_or_else(|| {
                Error::CannotFindTensor {
                    path: format!("{:?}", tensor_id),
                }
                .bt()
            })?;
            println!(
                "layer.{}.weight has gradient mean of {:?}",
                ix,
                grad_tensor.abs()?.mean_all()?.to_scalar::<f32>()?
            );
        }
        println!("\n");
        Ok(())
    }
}

impl Example for EG05 {
    fn description(&self) -> String {
        String::from("Comparison of gradients with and without shortcut connections.")
    }

    fn page_source(&self) -> usize {
        111_usize
    }

    fn main(&self) {
        use crate::listings::ch04::ExampleDeepNeuralNetwork;
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let layer_sizes = &[3_usize, 3, 3, 3, 3, 1];
        let sample_input = Tensor::new(&[[1_f32, 0., -1.]], vb.device()).unwrap();
        let model_without_shortcut =
            ExampleDeepNeuralNetwork::new(false, layer_sizes, vb.pp("model_wout_shortcut"))
                .unwrap();

        let model_with_shortcut =
            ExampleDeepNeuralNetwork::new(true, layer_sizes, vb.pp("model_with_shortcut")).unwrap();

        println!("model_without_shortcut gradients:");
        EG05::print_gradients(model_without_shortcut, &sample_input).unwrap();
        println!("model_with_shortcut gradients:");
        EG05::print_gradients(model_with_shortcut, &sample_input).unwrap();
    }
}
