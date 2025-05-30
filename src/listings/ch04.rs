use crate::candle_addons::{seqt, SequentialT};
use candle_core::{IndexOp, Module, Result, Tensor, TensorId, D};
use candle_nn::{
    embedding, linear_b, ops::softmax, seq, Dropout, Embedding, Linear, ModuleT, Sequential,
    VarBuilder,
};
use core::f64;
use std::cmp;
use std::ops::Mul;

use super::ch03::MultiHeadAttention;

const EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Config {
    pub fn gpt2_124m() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    pub fn gpt_sm_test() -> Self {
        Self {
            vocab_size: 500,
            context_length: 10,
            emb_dim: 12,
            n_heads: 3,
            n_layers: 2,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    #[allow(dead_code)]
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 1_024,
            n_heads: 16,
            n_layers: 24,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    #[allow(dead_code)]
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 1_280,
            n_heads: 20,
            n_layers: 36,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    #[allow(dead_code)]
    pub fn gpt2_xlarge() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 1_600,
            n_heads: 48,
            n_layers: 48,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

/// Listing 4.1
/// DummyGPTModel
pub struct DummyGPTModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    trf_blocks: Sequential,
    final_norm: DummyLayerNorm,
    out_head: Linear,
}

impl DummyGPTModel {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let token_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.context_length, cfg.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(cfg.drop_rate);
        let mut trf_blocks = seq();
        for _ in 0..cfg.n_layers {
            trf_blocks = trf_blocks.add(DummyTransformerBlock::new(cfg).unwrap());
        }
        let final_norm = DummyLayerNorm::new(cfg.emb_dim)?;
        let out_head = linear_b(cfg.emb_dim, cfg.vocab_size, false, vb.pp("out_head"))?;
        Ok(Self {
            token_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }
}

impl Module for DummyGPTModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.token_emb.forward(xs)?;

        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, true)?;
        x = self.trf_blocks.forward(&x)?;
        x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}
/// Listing 4.1
/// DummyLayerNorm
pub struct DummyLayerNorm {}

impl DummyLayerNorm {
    #[allow(unused_variables)]
    pub fn new(emb_dim: usize) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for DummyLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.to_owned())
    }
}

/// Listing 4.1
/// DummyTransformerBlock
pub struct DummyTransformerBlock {}

impl DummyTransformerBlock {
    #[allow(unused_variables)]
    pub fn new(cfg: Config) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for DummyTransformerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.to_owned())
    }
}

/// Listing 4.2
#[allow(dead_code)]
pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn new(emb_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let scale = vb.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0.))?;

        Ok(Self {
            eps: EPS,
            scale,
            shift,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let var = xs.var_keepdim(D::Minus1)?;
        let norm_xs = xs.broadcast_sub(&mean)?.broadcast_div(&var.sqrt()?)?;
        let out_norm = norm_xs
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;
        Ok(out_norm)
    }
}

/// Listing 4.3
pub struct GELU;
impl Module for GELU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (0.5_f64 * xs)?.mul(
            &((2_f64 / f64::consts::PI).sqrt() * (xs + (xs.mul(xs)?.mul(xs)? * 0.044715_f64))?)?
                // here can not use xs.broadcast_pow(), seems the broadcast_pow will do
                // the pow function did not return the result we think such as below
                // + (xs.broadcast_pow(&Tensor::full(3_f32, xs.dims(), xs.device())?)? * 0.044715_f64))?)?
                .tanh()?
                .broadcast_add(&Tensor::ones((1,), candle_core::DType::F32, xs.device())?)?,
        )
    }
}

/// Listing 4.4
pub struct FeedForward {
    pub layers: Sequential,
}

impl FeedForward {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let layers = seq()
            .add(linear_b(
                cfg.emb_dim,
                4_usize * cfg.emb_dim,
                true,
                vb.pp("first_layer"),
            )?)
            .add(GELU)
            .add(linear_b(
                4_usize * cfg.emb_dim,
                cfg.emb_dim,
                true,
                vb.pp("second_layer"),
            )?);
        Ok(Self { layers })
    }

    pub fn from_fields(layers: Sequential) -> Result<Self> {
        Ok(Self { layers })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.layers.forward(xs)
    }
}

/// Listing 4.5
/// ExampleDeepNeuralNetwork
pub struct ExampleDeepNeuralNetwork {
    use_shortcut: bool,
    pub layers: Vec<Sequential>,
    pub tensor_ids: Vec<TensorId>, // to be able to print gradients from GradStore
}

impl ExampleDeepNeuralNetwork {
    pub fn new(use_shortcut: bool, layer_sizes: &[usize], vb: VarBuilder<'_>) -> Result<Self> {
        let mut layers: Vec<Sequential> = Vec::new();
        let mut tensor_ids: Vec<TensorId> = Vec::new();
        for i in 0..layer_sizes.len() - 1_usize {
            let linear = linear_b(
                layer_sizes[i],
                layer_sizes[i + 1],
                true,
                vb.pp(format!("layer-{}", i)),
            )?;
            tensor_ids.push(linear.weight().id());
            layers.push(seq().add(linear).add(GELU))
        }

        Ok(Self {
            use_shortcut,
            layers,
            tensor_ids,
        })
    }
}

impl Module for ExampleDeepNeuralNetwork {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.to_owned();
        for layer in self.layers.iter() {
            let layer_output = layer.forward(&x)?;
            if (self.use_shortcut) && (xs.dims() == layer_output.dims()) {
                x = (xs + layer_output)?;
            } else {
                x = layer_output;
            }
        }
        println!("after {:?}", x.to_vec2::<f32>());
        Ok(x)
    }
}

/// Listing 4.6
pub struct TransformerBlock {
    att: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let att = MultiHeadAttention::new(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
            vb.pp("mha"),
        )?;

        let ff = FeedForward::new(cfg, vb.pp("ff"))?;
        let norm1 = LayerNorm::new(cfg.emb_dim, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(cfg.emb_dim, vb.pp("norm2"))?;
        let drop_shortcut = Dropout::new(cfg.drop_rate);
        Ok(Self {
            att,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }

    pub fn from_fields(
        att: MultiHeadAttention,
        ff: FeedForward,
        norm1: LayerNorm,
        norm2: LayerNorm,
        drop_shortcut: Dropout,
    ) -> Result<Self> {
        Ok(Self {
            att,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs.to_owned();
        let mut x = xs.to_owned();
        x = self.norm1.forward(&x)?;
        x = self.att.forward(&x)?;
        x = self.drop_shortcut.forward(&x, true)?;
        x = (x + shortcut)?;

        let shortcut = x.clone();
        x = self.norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.drop_shortcut.forward(&x, true)?;
        x = (x + shortcut)?;
        Ok(x)
    }
}

/// LIsting 4.7
pub struct GPTModel {
    toke_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    trf_blocks: SequentialT,
    final_norm: LayerNorm,
    out_head: Linear,
}

impl GPTModel {
    pub fn new(config: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let toke_emb = embedding(config.vocab_size, config.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(config.context_length, config.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(config.drop_rate);
        let mut trf_blocks = seqt();
        for ix in 0..config.n_layers {
            // the parameter `s` for `vb.pp` need be different
            trf_blocks =
                trf_blocks.add(TransformerBlock::new(config, vb.pp(format!("trf.{}", ix)))?);
        }
        let final_norm = LayerNorm::new(config.emb_dim, vb.pp("final_norm"))?;
        let out_head = linear_b(config.emb_dim, config.vocab_size, false, vb.pp("out_head"))?;
        Ok(Self {
            toke_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }

    pub fn from_fields(
        toke_emb: Embedding,
        pos_emb: Embedding,
        drop_emb: Dropout,
        trf_blocks: SequentialT,
        final_norm: LayerNorm,
        out_head: Linear,
    ) -> Result<Self> {
        Ok(Self {
            toke_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, true)
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.toke_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;
        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, train)?;
        x = self.trf_blocks.forward_t(&x, train)?;
        x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;

        Ok(logits)
    }
}

/// Listing 4.8
pub fn generate_text_simple(
    model: GPTModel,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let (_b, seq_len) = idx.dims2()?;
        let start_token_index = cmp::max(0isize, seq_len as isize - context_size as isize) as usize;
        let idx_cond = idx.i((.., start_token_index..seq_len))?;
        let logits = model.forward_t(&idx_cond, false)?;
        let (_b, c, _vocab_size) = logits.dims3()?;
        let logits = logits.i((.., c - 1, ..))?;
        let probas = softmax(&logits, 1)?;
        let idx_next = probas.argmax_keepdim(D::Minus1)?;
        idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
    }

    Ok(idx)
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{test_utils, DType, Device, IndexOp, Tensor};
    use candle_nn::{Activation, VarBuilder, VarMap};
    use rstest::*;

    #[fixture]
    pub fn vb() -> VarBuilder<'static> {
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, &dev)
    }

    #[fixture]
    pub fn batch_token_ids() -> Tensor {
        let dev = Device::cuda_if_available(0).unwrap();
        Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev).unwrap()
    }

    #[rstest]
    fn test_dummy_gpt_model_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, vb).unwrap();

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.token_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.n_layers);

        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
    }

    #[rstest]
    fn test_dummy_gpt_model_forward(vb: VarBuilder<'_>, batch_token_ids: Tensor) {
        let (batch_size, seq_len) = batch_token_ids.dims2().unwrap();

        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, vb).unwrap();

        let logits = model.forward(&batch_token_ids).unwrap();

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[rstest]
    fn test_layer_norm_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb).unwrap();
        assert_eq!(layer_norm.eps, EPS);
        assert_eq!(layer_norm.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.shift.dims(), &[cfg.emb_dim]);
        assert_eq!(
            layer_norm.scale.i(..=1).unwrap().to_vec1::<f32>().unwrap(),
            &[1., 1.]
        );

        assert_eq!(
            layer_norm.shift.i(..=1).unwrap().to_vec1::<f32>().unwrap(),
            &[0., 0.]
        );
    }

    #[rstest]
    fn test_layer_norm_forward(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let batch_size = 2_usize;
        let batch_example =
            Tensor::rand(0f32, 1f32, (batch_size, cfg.emb_dim), vb.device()).unwrap();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb).unwrap();

        let out_norm = layer_norm.forward(&batch_example).unwrap();
        let mean = out_norm.mean_keepdim(D::Minus1).unwrap();
        let var = out_norm.var_keepdim(D::Minus1).unwrap();

        let mean_minus_zero = mean
            .broadcast_sub(&mean.zeros_like().unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let var_minus_one = var
            .broadcast_sub(&var.ones_like().unwrap())
            .unwrap()
            .abs()
            .unwrap();

        assert_eq!(out_norm.dims(), &[batch_size, cfg.emb_dim]);
        assert_eq!(
            test_utils::to_vec2_round(&mean_minus_zero, 2_i32).unwrap(),
            [[0.0], [0.0]]
        );
        assert_eq!(
            test_utils::to_vec2_round(&var_minus_one, 2_i32).unwrap(),
            [[0.0], [0.0]]
        );
    }

    #[rstest]
    fn test_gelu_impl() {
        let dev = Device::cuda_if_available(0).unwrap();
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 3_usize), &dev).unwrap();

        // testing manual impl
        let gelu = GELU;
        let out = gelu.forward(&batch_example);

        // reference impl
        let candle_gelu = Activation::Gelu;
        let candle_out = candle_gelu.forward(&batch_example);

        // assert equality
        assert_eq!(
            test_utils::to_vec2_round(
                &(out.unwrap() - candle_out.unwrap()).unwrap().abs().unwrap(),
                2_i32
            )
            .unwrap(),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
    }

    #[rstest]
    fn test_feedforward_init(vb: VarBuilder<'_>) {
        let ff = FeedForward::new(Config::gpt_sm_test(), vb.pp("ff")).unwrap();

        assert_eq!(ff.layers.len(), 3_i64);
    }

    #[rstest]
    fn test_feedforward_forward(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let ff = FeedForward::new(cfg, vb.pp("ff")).unwrap();

        // create test batch
        let (batch_size, seq_len) = (2_usize, 3_usize);
        let batch_example =
            Tensor::rand(0f32, 1f32, (batch_size, seq_len, cfg.emb_dim), vb.device()).unwrap();
        let out = ff.forward(&batch_example).unwrap();

        assert_eq!(out.dims(), &[batch_size, seq_len, cfg.emb_dim]);
    }

    #[rstest]
    fn test_example_deep_neural_network_init(vb: VarBuilder<'_>) {
        let layer_sizes = &[3_usize, 2, 2, 1];
        let model = ExampleDeepNeuralNetwork::new(true, layer_sizes, vb).unwrap();

        assert_eq!(model.layers.len(), layer_sizes.len() - 1usize);
        assert_eq!(model.use_shortcut, true);
    }

    #[rstest]
    fn test_example_deep_neural_network_forward(vb: VarBuilder<'_>) {
        let layer_sizes = &[3_usize, 2, 2, 1];
        let model = ExampleDeepNeuralNetwork::new(true, layer_sizes, vb.pp("model")).unwrap();
        let sample_input = Tensor::new(&[[1f32, 0., 1.], [0., 1., 0.]], vb.device()).unwrap();

        let output = model.forward(&sample_input).unwrap();

        assert_eq!(output.dims(), &[2_usize, 1_usize]);
    }

    #[rstest]
    fn test_transformer_block_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer")).unwrap();

        assert_eq!(transformer_block.att.num_heads(), cfg.n_heads);
        assert_eq!(transformer_block.att.drop_p(), cfg.drop_rate);
        assert_eq!(
            transformer_block.att.w_key().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );

        assert_eq!(
            transformer_block.att.w_query().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.att.w_value().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );

        assert_eq!(transformer_block.att.head_dim(), cfg.emb_dim / cfg.n_heads);
        assert_eq!(transformer_block.ff.layers.len(), 3_i64);
        assert_eq!(transformer_block.norm1.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(transformer_block.norm2.scale.dims(), &[cfg.emb_dim]);
    }

    #[rstest]
    fn test_transformer_block(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer")).unwrap();

        let batch_size = 2_usize;
        let num_tokens = 4_usize;
        let batch_example = Tensor::rand(
            0f32,
            1f32,
            (batch_size, num_tokens, cfg.emb_dim),
            vb.device(),
        )
        .unwrap();

        let out = transformer_block.forward(&batch_example).unwrap();
        assert_eq!(out.dims(), batch_example.dims());

        println!("{:?}", out.to_vec3::<f32>().unwrap());
    }

    #[rstest]
    fn test_gpt_model_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb).unwrap();

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.toke_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.n_layers);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
    }

    #[rstest]
    fn test_gpt_model_forward(vb: VarBuilder<'_>, batch_token_ids: Tensor) {
        let (batch_size, seq_len) = batch_token_ids.dims2().unwrap();

        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb).unwrap();

        let logits = model.forward(&batch_token_ids).unwrap();
        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[rstest]
    fn test_generate_text_simple(vb: VarBuilder<'_>, batch_token_ids: Tensor) {
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb).unwrap();

        // create sample idx
        let (batch_size, seq_len) = batch_token_ids.dims2().unwrap();
        let (context_size, max_new_tokens) = (2_usize, 3_usize);
        let idx =
            generate_text_simple(model, batch_token_ids, max_new_tokens, context_size).unwrap();

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
    }
}
