# What is a gpt?
Basically its a function that takes a sequence of tokens and outputs, for each position a probability distribution over what the next token should be

# Training loop
Give it a real text, ask it to predict the next token at every position, measure how wrong it is (loss function) then nudge the parameters to be less wrong

# Flow
text  →  tokens  →  embeddings  →  [Transformer block] × N  →  logits  →  probabilities
                       +                (attention + MLP)
                  position info

# First step (Tokenization)
It can't understand text, since math can't read, that's why we tokenize text, mapping characters to integers

# Second step (Embedding)
The integer 42 means nothing numerically, 42 isn't more meaningful than 7, so the first thing the model does is look up a learned vector for each token ID

It's like a table:
embedding_table[42]  →  [ 0.2, -0.7, 1.1, 0.03, ... ]   (dim = d_model, e.g. 64 or 768)

The table is a parameter, the model learns it. Tokens that should behave similarly (e.g., vowels) end up with similar vectors, because the training signal pushes them there

Shape transformation: (batch, block_size) of ints → (batch, block_size, d_model) of floats

So xb of shape (4, 8) becomes (4, 8, d_model).

# Third Step (Positional Encoding, letting the model know where things are)
Attention (fourth step) is order-blind. It sees a bag of token vectors and has no idea which came first. But in reality "dog bites man" ≠ "man bites dog", order matters