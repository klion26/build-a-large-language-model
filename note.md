# Ch02
# Ch03
1. first the input will be transformed to token(binary token, ref to ch02 for more),
  this will be a tensor(eg two-dim matrix, the first dim will be each input, each element in the two dim
  will be the token of each input.
2. calculate every probability of every token to each other
  this step will be matrix(use two-dim tensor as example) multiply
3. normalize the result for each input(the first dim in the two-dim matrix)
   reason from the book:
  `The main goal behind the normalization is to obtain attention weights that sum up to 1. This normalization is a convention that is useful for interpretation and maintaining training stability in an LLM.`
  `The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding samll gradients`
4. set three attention weight matrix(`query` - the current input, `key` - the index for the `value` will return for the `query`, and `value` - the return value for the `query`)
5. update attention weight matrix with input(train)
6. mask out the future tokens that did not be needed(eg, the input come before the `query`)
7. drop out some tokens to prevent overfitting


adjust the `d_out` for each `CausalAttention` instance (`MultiHeadAttentionWrapper` also) will have
an impact to the dimension of the final context_vector. (see `d_out` in `X3P2`).
The `MultiHeadAttentionWrapper` will concat each underlying instance.

`MultiHeadAttention` will initialize one large weight matrix, only perform **one matrix multiplication** with the inputs
to obtain a query matrix Q, and then split the query matrix into Q1 and Q2.
`MultiHeadAttentionWrapper` will split the weight matrix into two sub matrix, and do two matrix multiplication.

`MultiHeadAttention` is much more efficient than `MultiHeadAttentionWrapper` because there are less matrix multiplication operators.

In `MultiHeadAttention` we need to perform a transpose operation, the reason is that: 
This transposition is crucial for correctly aligning the queries, keys, and values across the different heads and 
performing batched matrix multiplications efficiently.
> From the book

Why multi-head 
> The multi-head component involves splitting the attention mechanism into multiple "heads".
> Each head learns different aspects of the data, allowing the model to simultaneously attend to information
> from different aspects of the data, allowing the model to simultaneously attend to information from
> different representation subspaces at different positions. This improves the model's performance in complex tasks.


# Ch04
 A GPT model
 
[input from user] -> [tokenize] -> [embedding] -> [transformer block] -> [output layers]

From [embedding layers] to [output layers] is GPT model

the [transformer block] containing the masked multi-head attention module describled in ch03

the [transformer block] contains 4 steps
- Layer nomalization   -- nomalization the input with mean 0 and variance 1
- GELU activation      -- use a smooth activation
- Feed forward network  -- add a new layer, to increase the representation of the model
- Shortcut connections  - preserve the flow of gradients during the backward pass in training.

The main idea behind layer normalization is to adjust the activations(outputs) of a neural network layer
to have a mean of 0 and a variance of 1, also known as unit variance.
> but why do we need to do this?
> Maybe this wants to normalize the input. -- get some input with small mean and variance

The `scale` and `shift` in Layer normalization are two trainable parameters that the LLM automatically adjusts during training
if it is determined that doing so would improve the model's performance on its training task.
> How to detect whether need to adjust the `scale` and `shift`?

> Historically, the ReLU activation function has been commonly used in deep learning due to its simplicity and effectiveness
> across various neural network architectures. However, in LLMs, several other activation functions are employed beyond
> the traditional ReLU. Two notable examples are GELU(Gaussian error linear unit) and SwiGLU(Swish-gated linear unit).

The smoothness of GELU can lead to better optimization properties during training, as it allows for more nuanced 
adjustments to the model's parameters.(See Figure 4.18 for more detail)


1. Inputs embedded into tokens
2. Transformer block
  2.1 First layer norm
    2.1.1 LayerNorm1 
    2.1.2 Maked multi-head attention
    2.1.3 Dropout
    2.1.4 Add short cut connector before 2.1(LayerNorm1)
  2.2 Second layer norm
    2.2.1 LayerNorm2
    2.2.2 Feed forward
      2.2.2.1 Linear layer
      2.2.2.2 GELU activation
      2.2.2.3 Linear layer
    2.2.3 Dropout
    2.2.4 Add shortcut connector before 2.2


![](https://raw.githubusercontent.com/klion26/ImageRepo/master/20250424190512.png)
> The pic above is an overview of the GPT model architecture showing the flow of data through the GPT model.
Starting from the bottom, tokenized text is first converted into token embeddings, which are then augmented with
> positional embeddings. This combined information forms a tensor that is passed through a series of
> transformer blocks shown in the center(each containing multi-head attention and feed forward neural network
> layers with dropout and layer normalization), which are stacked on top of each other and repeated 12 times.
> The output from the final transformer block then goes through a final layer normalization step before
> reaching the linear output layer. This layer maps the transformer's output to a high-dimensional space
> (in this case, 50257 dimensions, corresponding to the model's vocabulary size) to predict the next token in the sequence.