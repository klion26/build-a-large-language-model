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


Model size if the memory used for the total parameters(please see X4P2 for more detail).
```
total_params = sum(varmap.all_vars().iter().dims) // sum all of the dimensions of the parameters
total_size_bytes = total_params * 4     /// 4 is for f32
```


The step by step process by which an LLM generates text, one token at a time.
Starting with an initial input context("Hello, I am"), the model predicts a subsequent token
during each iteration, appending it to the input context round of prediction. As shown, the first iteration adds "a,"
the second "model," and the third "ready," progressively building the sentence.
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202505271012793.png)

One iteration in GPT Model for text generation
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202505271022502.png)

In `listing 4.8`
- the parameter `idx` is a (batch, n_tokens) array of indices in the current context.
- `idx.i((.., cmp::max(0usize, seq_len - context_size)..seq_len))?;` will crop current context if it exceeds the supported context size,
e.g., if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context
- `logits.i((.., c - 1, ..))?` focuse only on the last time step, so that (batch, n_token, vocab_size) becomes(batch, vocab_size)
- `probas = softmax(&logits, 1)?` probas has shape (batch, vocab_size)
- `idx_next = probas.argmax_keepdim(D::Minus1)?` idx_next has shape (batch, 1).
- `idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?` Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)

The six iterations of a token prediction cycle
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202505281002495.png))

In example 0408, it will print `Hello, I amintent hasht deepen Diffrou*/`, The reason the model is unable to produce coherent text is that we haven't
trained it yet.

Summary for this chapter
- Layer normalization stabilizes training by ensuring that each layer's outputs have a consistent mean and variance.
- Shortcut connections are connections that skip one or more layers by feeding the output of one layer directly to a deeper layer,
  which helps mitigate the **vanishing gradient problem** when training deep neural networks, such LLMs.
- Transformer blocks are a core structural component of GPT models, combining masked multi-head attention modules with
fully connected feed forward networks that use the GELU activation function.
- GPT models are LLMs with many repeated transformer blocks that have millions to billions of parameters.
- GPT models come in various sizes, for example, 124,345,762, and 1,542 million parameters, which we can implement with the same
`GPTModel` Python class.
- The text-generation capability of a GPT-like LLM involves decoding output tensors into human-readable text by
sequentially predicting one token at a time based on a given input context.
- Without training, a GPT model generates incoherent text, which underscores the importance of model training for coherent text generation.

# Ch05

The three main stages of coding an LLM: 1) Building an LLM; 2) Foundation model; 3) fine-tuning a foundation model.
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202505290848322.png)

In the context of LLMs and other deep learning models, *weights* refer to the trainable parameters that the learning process adjusts.
These weights are also known as *weight parameters* or simple *parameters*.

The topics covered in chapter 5
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202505290857287.png)

- Text evaluation: Implement the loss computation to evaluate how well the model performs
- Training&validation losses: Apply the loss to the entire dataset, which we split into a training and validation portion
- LLM training function: Train the model to generate human-like text
- Text generation strategies: Implement additional LLM text generation strategies to reduce training data memorization
- Weight saving&loading: Implement function to save and load the LLM weights to use or continue training the LLM later
- Pretrained weights from OpenAI: Load pretrained weights from OpenAI into out LLM model.

The goal of model training is to ensure that the probability values corresponding to the highlighted target token IDs are maximized.

Calculating the loss involves several steps such as below
![](https://raw.githubusercontent.com/klion26/ImageRepo/master/202506032027020.png)


At its core, the cross entropy loss is a popular measure in machine learning and deep learning that measures the difference
between two probability distributions.

*perplexity* is a measure often used alongside cross entropy loss to evaluate the performance of models in tasks like language modeling.
It can provide a more interpretable way to understand the uncertainty of a model in predicting the next token in a sequence.
# Q
- what is the different effect of different `batch size`? 