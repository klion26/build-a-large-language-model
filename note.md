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
4. set three attention weight matrix(`query` - the current input, `key` - the index for the `value` will return for the `query`, and `value` - the return value for the `query`)
5. update attention weight matrix with input(train)
6. mask out the future tokens that did not be needed(eg, the input come before the `query`)
7. drop out some tokens to prevent overfitting
   