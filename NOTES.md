# What is a gpt?
Basically a fancy autocomplete, it predicts the next work in a sentence:
The cat is _ -> GPT -> The cat is fat

# Input
## Tokenization
Computers can't understand text, since it's just math inside, that's why we convert characters into tokens (mapping characters into integers)

## Embedding Layer
Embedding = Sequence of numbers in a certain dimension
Basically it shows relation between words, `man - king + woman = queen`

# Output
## Softmax
Convert raw final outputs (logits) of a neural network into probabiliites

$$
\sigma (\mathbf{z})_{i}=\frac{e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}
$$

Breakdown:
- $\sigma(\mathbf{z})_i$ : The output probability for the i-th class. All outputs range between 0 and 1, and they sum up to exactly 1
- $z_{i}$ : The input score (logit) for the i-th class
- $e^{z_{i}}$ : The exponential function applied to the input score. This turns all negative numbers into positive values and magnifies the differences between scores
- $\sum_{j=1}^{K} e^{z_j}$ : The normalization term. It sums up the exponential values of all K classes, ensuring that the final output values behave as a valid probability distribution

Problem: Using integers that are way too big for the data type, i.e $e^{275}$, ect..

Solution: Stable Softmax, for every xi, we subtract with the maximum xi value in the vector
Formula:

$$
\sigma (\mathbf{z})_{i}=\frac{e^{z_{i}-\max (\mathbf{z})}}{\sum _{j=1}^{K}e^{z_{j}-\max (\mathbf{z})}}
$$

## Cross-Entropy Loss
Measure how wrong our GPT is

$$
L=-\sum _{c=1}^{K}y_{c}\log (p_{c})
$$

- $K$ : The total number of classes (or vocabulary size in a language model)
- $y_{c}$ : A binary indicator (0 or 1) indicating if class $c$ is the correct classification. In practice, this represents a one-hot encoded vector
- $p_{c}$: The model's predicted probability for class $c$ (the output from the Softmax function).

Problem: Model predicts [0, 0, 0, 1], but in reality, it's supposed to be [0, 0, 1, 0], model does log(0), which gives $-\infty$

Solution:
$$
L=-z_{c}+\max (\mathbf{z})+\log \sum _{j=1}^{K}e^{z_{j}-\max (\mathbf{z})}
$$

By cancelling out the log and exponential: $\log \left(e^{z_{c}-\max (\mathbf{z})}\right)\implies z_{c}-\max (\mathbf{z})$  it never encounters a 0 or a $\log(0)$ error anywhere, no matter how wrong the model is, because now it's just a simple subtraction

# Transformer
## Attention
Currently, tokens have no corelation with each other, so we use attention, one way to look at it is: 
> "For each token, which other tokens should I look at to figure out what comes next?"

Each token builds three vectors from its current embedding (trainable weights):
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain / advertise?"
- Value (V): "What do I actually offer if you attend to me?"

Flow: Embeddings -> Dot product of WQ & WK & WV to get Q, K -> then $Q \cdot K^T$ -> Softmax (divide by $\sqrt{d_k}$ ) to get Weights ($W$) -> $W \cdot V$ to finally get context

But a problem arises, models can just cheat by looking at the next token, thats why we do Causal Attention, by masking the next tokens for every token before doing $W \cdot V$

## Multi-Head Attention
One attention head can only learn one "type of relationship." So we run several in parallel (e.g., 4 heads), each with its own Q/K/V weights, then concatenate their outputs. Different heads specialize, one might track syntax, another coreference, etc..

## Layer Normalization
To stabilize and accelerate the training of deep neural networks, we use Layer Normalization (LayerNorm)

Without normalization, layers later in the network constantly have to adapt to shifting distributions of inputs from earlier layers. A problem known as internal covariate shift. LayerNorm fixes this by enforcing a consistent distribution of activations

$$
\text{LayerNorm}(x)_{k}=\gamma _{k}\cdot \left(\frac{x_{k}-\mu }{\sqrt{\sigma ^{2}+\epsilon }}\right)+\beta _{k}
$$

Breakdown:
- $\gamma_{k}$ : gain (trainable)
- $\mu$ : mean
- $\sigma^{2}$ : variance
- $\epsilon$ : a very small value e.g 1e-5, so we never divide by 0
- $\beta_{k}$ : bias (trainable)

## Feed Forward Network (MLP)
After attention mixes information between tokens, each token gets processed individually by a small 2-layer MLP:

`x → Linear(d_model → 4*d_model) → ReLU → Linear(4*d_model → d_model)`

(Original GPT paper uses GELU, but I will use ReLU since it's simpler)

This is where most of the model's parameters live, and where a lot of "reasoning" capacity comes from. Attention gathers context; the MLP thinks about it

## Residual Connection
`Residual: x = x + sublayer(x)`

instead of replacing x, we add the sublayer's output to it. This gives gradients a direct highway back to the input, so a 12-layer network trains as easily as a 2-layer one