# What is a GPT?
Basically a fancy autocomplete, it predicts the next work in a sentence:
The cat is _ -> GPT -> The cat is fat

# Input
## Tokenization
Computers can't understand text, since it's just math inside, that's why we convert characters into tokens (mapping characters into integers)

## Embedding Layer
Embedding = Sequence of numbers in a certain dimension
Basically it shows relation between words, `man - king + woman = queen`

So in here, we turn our tokens into a vector, usually just a random small vector, since it will get it's meaning during training

Two embeddings are added together before anything else happens:
- Token embedding: a (vocab_size, d_model) table. token_embedding[id] gives us that token's starting vector. Values are random at init; training is what makes them meaningful (the man - king + woman = queen example, it emerges from the loss signal, from training)
- Position embedding: a (block_size, d_model) table. Attention is order-blind, so without this the model couldn't distinguish "dog bites man" from "man bites dog". Position t always uses row t of this table

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

## The actual math behind Residual Connection
Here's what it means for gradients:

Forward:  `y = x + f(x)`\
Backward: `dx = dy + f'(dy)`

The gradient at `x` is the sum of two paths: one through the direct connection (dy passes through unchanged), one through the sublayer (f'(dy)).

# Backpropping
The GPT has alot of parameters, right now they're random. To train them is to nudge them so the loss goes down

To nudge a parameter correctly we have to know
> "If I increase this specific parameter by a tiny amount, does the loss go up or down, and by how much?"

That's the gradient. If dL/dW is positive, increasing W increases the loss, so we want to decrease W. If it's negative, increase W. The update is always: `W = W - learning_rate * dL/dW`

Do this to every parameter a few thousand times, and the model learns. It's called gradient descent

So basically: compute dL/dW for every parameter W in the network

## Why the naive idea won't work
One way to compute dL/dW for a parameter: nudge it by eps, re-run the forward pass, see how the loss changes. Do that for every parameter. I.e numerical gradient

But: for P parameters, that's 2P forward passes. With P = 100,000, that's 200,000 forward passes per training step. Not effective

Backprop is a way to get all P gradients in a single backward pass, about the same cost as one forward pass

## CALCULUS!!! (Chain Rule)
Consider a simple chain of operations:
> x  →  a  →  b  →  L\
where a = f(x), b = g(a), L = h(b).

If I want dL/dx (how the loss changes when I wiggle x), we can use the echain rule:
> dL/dx  =  dL/db  *  db/da  *  da/dx

In other words: the effect of x on L is the product of the effects along the path. If x moves slightly, a moves by da/dx, which makes b move by db/da, which makes L move by dL/db. All multiplied together

Each factor here is a "local gradient", the derivative of one operation with respect to its own input. Each is a simple, known formula. The chain rule stitches them together

Notice:
- `dL/db = dL/db`, we need this to compute `dL/da = dL/db * db/da`
- `dL/da` is then reused to compute `dL/dx = dL/da * da/dx`

So we compute gradients from the loss backward:
- Step 1:  `dL/db` (comes from the loss function)
- Step 2:  `dL/da = dL/db * db/da`
- Step 3:  `dL/dx = dL/da * da/dx`

Each step takes one "upstream" gradient (dL/db), multiplies it by a local gradient, and produces the next "downstream" gradient (dL/da). One pass, one multiplication per operation

This is backpropagation: propagate the gradient backward through the graph, one step at a time

## The rule
Every operation in the network, no matter how complex, obeys this rule:

> Given: `dL/dout` (the gradient of the loss with respect to this operation's output)\
Produce: `dL/din = dL/dout * dout/din` (times the local gradient), and if the operation has parameters, also `dL/dW = dL/dout * dout/dW`

The dout/din factor is the local gradient, it depends only on the operation itself, not on anything downstream. For every operation type, this local gradient is a known, closed-form formula. We look it up (or derive it once) and plug it in

This gives us the interface every layer's backward must obey:
```
def backward(self, dout):
    # dout: gradient of loss w.r.t. this layer's output
    # returns: gradient of loss w.r.t. this layer's input
    # side effect: store gradients of any parameters in self.dW, self.db, etc.
    return din
```

Once every layer has this method, backprop is just calling them in reverse order

## Local gradient formulas
Every backward pass is built from a handful of these:

### Addition: `y = a + b`
```
dL/da = dL/dy      dL/db = dL/dy
```
Gradient passes through unchanged, **duplicated**. Why residuals are easy: `x = x + f(x)` splits the gradient into two paths

### Scalar multiply: `y = c·x`
```
dL/dx = dL/dy · c
```
Just scale

### Elementwise multiply: `y = a * b`
```
dL/da = dL/dy · b
dL/db = dL/dy · a
```
Gradients **swap**, each input gets the other's value.

### Matmul: `Y = X @ W`
```
dL/dX = dL/dY @ W.T
dL/dW = X.T @ dL/dY
```
**The most important pair in the whole network.** If a matmul backward has a shape error, it's almost always a missing transpose

### ReLU: `y = max(x, 0)`
```
dL/dx = dL/dy · (x > 0)
```
Mask. Positive values pass through; non-positive block

### LayerNorm
The closed-form for `dx`:

$$
dx = \text{inv\_std} \cdot \bigl(d\hat{x} - \text{mean}(d\hat{x}) - \hat{x} \cdot \text{mean}(d\hat{x} \cdot \hat{x})\bigr)
$$

where `d x̂ = dout · γ`, and all means are over the feature axis. The `mean(dx̂)` term accounts for the shift, the `x̂ · mean(dx̂ · x̂)` term for the scale

### Reshape / transpose / concat
- reshape backward: reshape back
- transpose backward: transpose back (its own inverse)
- concatenate backward: split into the original pieces

These three appear in attention backward

## Training Loop
|Phase|What it does|Cost|
|--------|--------|--------|
|Forward|input → predictions → loss|1 pass|
|Backward|loss → input, computing every dL/dW along the way|~1 pass|
|Update|W -= lr * dL/dW for every parameter|trivial|

## TL;DR
Backprop is the chain rule applied backward through the network, reusing each step's gradient so the whole thing costs one pass instead of one pass per parameter

Each layer's backward is: "multiply the incoming gradient by my local derivative and pass it along." Write it once per layer, check it against finite differences, and we're done

## Gradient checking, how to verify we're right
When we implement a layer's backward, **verify it against a numerical estimate** before trusting it.

### The idea
For a scalar loss L:
- Nudge a parameter W by `+ε` → loss `L⁺`
- Nudge it by `−ε` → loss `L⁻`
- Numerical gradient ≈ `(L⁺ − L⁻) / (2ε)`
- Compare to what our backward produced. Should match to ~6 decimals

### The scalar-loss trick for testing a single layer
Test each layer in isolation. Given `out = layer.forward(x)`, pick a random `dout` (upstream gradient), and define:
```
L = sum(out * dout)
```
Then `dL/dout = dout` **by construction**, the gradient is exactly the arbitrary vector we chose. This lets us test the *layer's* backward in isolation, with a controllable upstream gradient.

### What to expect
| Relative error | Verdict |
|----------------|---------|
| < 1e-5 | **Correct** (numerical error from finite differences) |
| 1e-5 to 1e-3 | Borderline — accumulated over many layers, or a small bug |
| > 1e-3 | **Bug** in the formula |

### What the failure pattern tells us
- **Only `dx` fails, params pass** → bug is in the `dx` path (last step of the backward)
- **Only `dW` fails, `dx` passes** → bug is in the parameter gradient (the `.T @` pattern or the sum axis)
- **Everything fails** → bug is in a shared intermediate, or the loss function itself

### Do this per layer
Implement MLP backward → check. LayerNorm → check. Attention → check. Embedding → check. **Then** chain them. Never trust an unchecked backward. Without this, a subtle LayerNorm bug buried in 2 stacked blocks is essentially invisible from the loss curve alone

## Softmax + Cross-Entropy Backward
The forward derivation is in the Cross-Entropy section above. The backward is astonishingly clean once combined:

$$
\frac{\partial L}{\partial \text{logits}} = \frac{\text{softmax}(\text{logits}) - \text{one\_hot}(y)}{B \cdot T}
$$

No logs, no max-subtraction. The whole `log(0)` problem vanishes because we never take a log of a probability in the backward. The "cancel log and exp" trick in the forward derivation is exactly what makes this possible.

# Adam (Adaptive Moment Estimation)
It's a refinement of gradient descent that changes how each parameter is nudged, not which gradients are computed. Backprop gives us dL/dW; Adam turns that into the actual step size

## The two problems with vanilla SGD
`W = W - lr * dL/dW` has two weaknesses:

1. Same learning rate for every parameter: Some parameters see huge gradients (recently initialized, noisy), others tiny (deep, converged). A single lr is either too big for the first group or too small for the second
2. One gradient step is noisy. Each step is computed on one minibatch, so the direction is approximate. Following it blindly means zigzagging. Averaging over recent steps would give a truer direction

Adam fixes both, per-parameter, with two running averages

## The two moments
For each parameter W, maintain two running averages:

First moment m (momentum), exponential moving average of the gradient:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Smooths the direction. Consistent direction → m grows. Flipping → m damps. It's a low-pass filter on the noisy per-step signal

Second moment v (variance-like), exponential moving average of the squared gradient:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Estimates this parameter's typical gradient magnitude. Because it's per-parameter, it says how "trustworthy" or "large" this gradient usually is

## Bias correction
Both m and v start at 0. So at t=1: `m_1 = 0.1·g_1` (with β₁=0.9), 10× too small. Adam corrects:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

At t=1: `m̂₁ = 0.1·g_1 / 0.1 = g_1` (exact). As t grows, correction → 1, so it stops mattering

## The update

$$
\theta_t = \theta_{t-1} - \text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Look at the ratio `m̂ / √v̂`:
- Consistently one way, stable magnitude → ratio ≈ ±1 → step ≈ lr
- Noisy or small gradient → ratio small → small step
- Rare huge spike → m̂ grows slowly (smoothed), √v̂ grows fast → small effective step. Spikes suppressed

The ratio is bounded roughly by ±1 (Cauchy–Schwarz), so the effective step size per parameter is at most **lr**, regardless of gradient magnitude. That's why Adam doesn't blow up on steep loss surfaces

## Hyperparameters
| Name | Typical | Meaning |
|------|---------|---------|
| lr | 3e-4 (transformers) | base step size |
| β₁ | 0.9 | first moment decay (~10-step memory) |
| β₂ | 0.999 | second moment decay (~1000-step memory) |
| ε | 1e-8 | numerical floor, avoids div-by-zero |

`β₂ > β₁` because variance estimates are noisier than mean estimates — squared gradient jitters more, so it needs a longer average to get a stable scale estimate

# AdamW (Adam with decoupled weight decay)

## The problem with L2 in Adam
We might add L2 regularization: `λ·W` folded into the gradient. In vanilla Adam with L2:

$$
g_{adam} = g + \lambda \cdot W
$$

The problem: `λ·W` gets folded into m and v, which are then **rescaled by the adaptive ratio m̂/√v̂**...

The problem: `λ·W` gets folded into m and v, which are then **rescaled by the adaptive ratio m̂/√v̂**. For a parameter with small gradients, √v̂ is small, so the effective decay gets *amplified*. For big gradients, it gets *suppressed*. The actual decay per parameter is **not uniform**, it depends on gradient history. Not what "regularization" should mean

## The fix: decouple
Apply decay directly to the parameter, separate from the moments:
- moments: `m = β₁·m + (1-β₁)·g` — decay does NOT touch the gradient
- adaptive step: `θ -= lr · m̂ / (√v̂ + ε)`
- decay: `θ -= lr · λ · θ` — uniform, every step, every weight matrix

Now every weight matrix shrinks by `lr·λ` per step, regardless of gradient magnitude. Uniform, predictable. This is why it's called "decoupled"

## The code change from OG Adam
One line added to the update loop:
```
if p.ndim >= 2:
    p -= self.lr * self.wd * p
```

## What `ndim >= 2` does

Skip decay on:
- **Biases** — shift, don't scale. Decaying toward 0 is wrong
- **LayerNorm γ, β** — per-feature scale/shift. Decaying γ toward 0 would kill LayerNorm's ability to rescale

Decay only weight matrices: WQ, WK, WV, WO, W1, W2, W_out. The `ndim >= 2` check separates them cleanly.

## When Adam vs AdamW actually matters

At small scale (100k–500k params, few thousand steps): Adam vs AdamW nearly identical. The difference appears when:
- **Training longer** — the train/val gap grows, AdamW's uniform decay slows overfitting more cleanly
- **Scaling up** — AdamW is what makes large models train stably
- **Tuning weight_decay** — with AdamW, λ is a real interpretable knob. With Adam+L2, it's tangled up with the adaptive ratio

## Companion: cosine LR decay

AdamW is usually paired with cosine decay:
```python
def get_lr(step, max_steps, base_lr, warmup=200):
    if step < warmup:
        return base_lr * step / warmup   # linear warmup
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```
- **Warmup**: ramp lr from 0 to base in first 200 steps. Prevents early divergence from noisy initial gradients
- **Cosine**: lr smoothly decays from base to 0. Late in training, small steps let the model settle into a narrow minimum

Typically gives 5–10% better final val loss for the same steps.

# Reading a training run

## The loss curve

Watch two things: **the level** and **the gap**.

| Symptom | Meaning |
|---------|---------|
| Loss stays at `log(vocab_size)` | Gradients are zero or broken — check `dW_out`, `Transformer.backward`'s `+`s |
| Loss drops smoothly | Correct — gradient descent is working |
| Loss → NaN early | lr too high, or a bug producing inf/nan gradients |
| Loss drops then explodes | Gradient explosion — lower lr, or grad clipping |
| Train drops, val flat/rising | Overfitting — smaller model, more data, or stop earlier |
| Train ~ val, both dropping | Underfitting — train longer, or bigger model |
| Train ≪ val, big gap | Overfitting — regularize or stop |

At start with fresh random weights, loss should be **exactly `log(vocab_size)`** — the random-guessing baseline. If it's lower, you have a bug leaking information. If it's higher, weights are mis-scaled.

## Train vs val noise

Train loss is computed on **one batch** (e.g. `4 × 32 = 128` tokens), so it's very noisy — you'll see zigzag. Val loss is averaged over 20 batches (2560 tokens), so it's smooth. **This is normal, not a problem.** Don't worry about train loss jumping around.

## Reading the sample text

At character level (vocab ~95), what you see depends on how much the model has learned:

| Learned | Not learned (yet) |
|---------|-------------------|
| Real words ("the", "is", "for") | Semantics (fake words like "rigents") |
| Word boundaries (spaces in right places) | Long-range coherence (sentences don't follow) |
| Capitalization (sentence starts, "I") | Consistent grammar (tense mixing) |
| Punctuation (.,!?"()...) | Real vocabulary (needs subword tokenization) |
| English-shaped pseudo-words ("shought", "tr ingers") | |

Character-level models can produce English-shaped **forms** but not real **meaning**. That's why real GPTs use subword tokenization (BPE, ~50k vocab) — they can produce real words.

## Progress expectation for character-level GPT

| Params | Steps | Expected val | Sample quality |
|--------|-------|--------------|----------------|
| 200k | 5k | ~1.85 | Punctuation ✓, spaces ✓, some real words, gibberish elsewhere |
| 1.2M | 20k | ~1.4–1.5 | Real words more often, longer phrases, more grammar |
| 10M | 100k+ | ~1.2 | Coherent sentences, mostly grammatical |
| 124M (GPT-2) | 300k | ~0.9 | Coherent paragraphs, some reasoning |

The "coherent prose" ceiling for a given size is reached way before you think.

# Practical details that aren't obvious

## Why init std = 0.02

Embeddings and weight matrices use `randn * 0.02`. GPT-2 convention. Too large (std=1.0) → activations blow up through the stack, training diverges. Too small → gradients vanish, nothing learns. 0.02 is the empirical sweet spot for the depth of modern GPTs.

LayerNorm γ is init at 1 (not 0.02), β at 0, because γ scales the normalized output — starting at 1 means LayerNorm is "identity-ish" at first.

## Pre-norm vs post-norm

```
Pre-norm:  x = x + sublayer(LayerNorm(x))    ← GPT-2/3/LLaMA
Post-norm: x = LayerNorm(x + sublayer(x))    ← 2017 original Transformer
```

Pre-norm trains more stably at depth — LayerNorm runs on the *input* to each sublayer, so every sublayer sees a well-scaled input. Post-norm requires learning rate warmup and is more brittle. **Use pre-norm.**

## The output head `W_out`

The final projection:

$$
\text{logits} = x_{\text{final}} @ W_{\text{out}}
$$

Shape: `(B, T, d_model) @ (d_model, vocab_size) = (B, T, vocab_size)`.

Not shared with the token embedding. Some models "tie" them (use `token_emb.W.T` as `W_out`) — GPT-2 ties them. For a learning model, keep separate — clearer, one less thing to get wrong.

## Temperature and sampling

In `generate`, `logits_last = logits_last / temperature` before softmax. Effect:

- `temperature = 1.0`: sample from the model's actual distribution
- `temperature < 1.0` (e.g. 0.7): sharper, more confident/repetitive
- `temperature > 1.0` (e.g. 1.3): flatter, more random/creative
- `temperature → 0`: approaches argmax (greedy, deterministic)

**Always sample, don't just take argmax** — argmax gives repetitive degenerate text ("the the the..."). Sampling gives variety at the cost of occasional nonsense.

## Block vs GPT, what's what

| Name | What |
|------|------|
| Block (Transformer) | One layer: attn + MLP + residuals + norms. ~10 lines |
| GPT | `embedding → Block × N → final LayerNorm → output head` |

The GPT is the Block **stacked N times**. Early blocks learn local patterns, deeper blocks learn abstract structure. Depth is what turns "nice autocomplete" into something that can hold a sentence in its head. GPT-2 small has 12 blocks, GPT-3 has 96.