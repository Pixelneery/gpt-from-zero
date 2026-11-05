# How GPT Works
> Revised using Claude
## 0. The Big Picture

A GPT is a fancy autocomplete. It reads some text and guesses the next piece:

```
"The cat is ___"  ->  GPT  ->  "fat"
```

Everything in these notes serves one loop:

```
1. Turn text into numbers            (tokens -> embeddings)
2. Run the numbers through the model (attention + MLP, stacked N times)
3. Get a probability for every possible next token
4. Measure how wrong we were         (loss)
5. Work out which weights caused the error   (backprop)
6. Nudge every weight a little       (optimizer)
7. Repeat a few thousand times
```

Steps 1-3 are the **forward pass**. Step 4 is the **loss**. Step 5 is the **backward pass**. Step 6 is the **optimizer**. If you get lost later, come back to this list.

**How to read these notes:** each idea is *intuition -> formula -> gotcha*. Sections marked **Details** are optional. Skip them on the first read.

---

## 1. Forward Pass: Text In, Probabilities Out

### 1.1 Tokenization

Computers only understand numbers, so we map each character (or word piece) to an integer called a **token**.

```
"cat" -> [c, a, t] -> [3, 1, 20]
```

### 1.2 Embeddings

An **embedding** is a list of numbers (a vector) that represents a token. Words with related meanings end up with similar vectors. The famous example: `king - man + woman ≈ queen`.

We don't hand-design these. We start with small random numbers, and **training gives them meaning**.

Two tables are looked up and **added together** before anything else happens:

| Table | Shape | What it does |
|---|---|---|
| Token embedding | `(vocab_size, d_model)` | Row `id` is the starting vector for that token |
| Position embedding | `(block_size, d_model)` | Row `t` is the vector for "I am at position t" |

**Why positions?** Attention doesn't care about order by itself. Without position info, "dog bites man" and "man bites dog" would look identical.

### 1.3 Attention

**Intuition:** Right now every token is on its own. Attention lets each token ask: *"Which other tokens should I look at to figure out what comes next?"*

Each token makes three vectors from its embedding (using trainable weights):

- **Query (Q):** what I'm looking for (like a search query)
- **Key (K):** what I contain (like a label)
- **Value (V):** what I give you if you pick me (the actual content)

**Steps:**

1. `Q = X @ WQ`, `K = X @ WK`, `V = X @ WV`
2. `scores = Q @ K.T / sqrt(d_k)` (how well each query matches each key)
3. Apply the **causal mask**, then **softmax** to get weights
4. `context = weights @ V` (a weighted mix of values)

**Causal mask:** the model must not peek at future tokens (that would be cheating, since the future is the answer). So each token can only look at itself and earlier tokens. We block the rest by masking them out before the softmax.

**Why divide by `sqrt(d_k)`?** It keeps the scores from getting huge, which would make the softmax too extreme.

### 1.4 Multi-Head Attention

One attention head learns one kind of relationship. So we run several in parallel (e.g. 4), each with its own Q/K/V weights, and **concatenate** the results. One head might track grammar, another might track which noun a pronoun refers to, and so on.

### 1.5 Feed-Forward Network (MLP)

After attention mixes information **between** tokens, the MLP processes each token **on its own**:

```
x -> Linear(d_model -> 4*d_model) -> ReLU -> Linear(4*d_model -> d_model)
```

Think of it as: **attention gathers context, the MLP thinks about it.** Most of the model's parameters live here.

(The original GPT uses GELU instead of ReLU. We use ReLU because it's simpler.)

### 1.6 Residual Connections

```
x = x + sublayer(x)
```

**Intuition:** keep the original, add a correction. Don't replace it.

**Why it matters:** during training, gradients can flow straight back through the `+`, so a 12-layer network trains about as easily as a 2-layer one.

### 1.7 Layer Normalization (LayerNorm)

**Intuition:** keep the numbers at a steady scale so each layer gets well-behaved input.

For each token's vector: subtract the mean, divide by the standard deviation, then apply a learnable scale (`γ`) and shift (`β`).

$$
\text{LayerNorm}(x)_{k}=\gamma _{k}\cdot \left(\frac{x_{k}-\mu }{\sqrt{\sigma ^{2}+\epsilon }}\right)+\beta _{k}
$$

- `μ`: mean, `σ²`: variance
- `ε`: tiny number (like 1e-5) so we never divide by zero
- `γ` (gain) and `β` (bias): trainable

### 1.8 Putting It Together: Block vs GPT

| Name | What it is |
|---|---|
| **Block** | One layer: LayerNorm -> attention -> residual, then LayerNorm -> MLP -> residual |
| **GPT** | `embeddings -> Block × N -> final LayerNorm -> output head` |

Early blocks tend to learn local patterns. Deeper blocks learn more abstract structure. GPT-2 small has 12 blocks, GPT-3 has 96.

**Pre-norm vs post-norm:**

```
Pre-norm:  x = x + sublayer(LayerNorm(x))    <- GPT-2/3, LLaMA (use this)
Post-norm: x = LayerNorm(x + sublayer(x))    <- original 2017 Transformer
```

Pre-norm trains more stably when the network is deep.

### 1.9 The Output Head

A final linear layer turns each token's vector into one score per vocabulary token:

```
logits = x_final @ W_out      shape: (B, T, d_model) @ (d_model, vocab_size) -> (B, T, vocab_size)
```

These raw scores are called **logits**. GPT-2 reuses the token embedding table as `W_out` ("weight tying"). For learning, keep them separate. It's clearer.

---

## 2. Loss: How Wrong Are We?

### 2.1 Softmax: scores -> percentages

**Intuition:** turn logits into probabilities that are between 0 and 1 and add up to 100%.

$$
\sigma (\mathbf{z})_{i}=\frac{e^{z_{i}}}{\sum _{j=1}^{K}e^{z_{j}}}
$$

- `e^z` makes everything positive and widens the gaps between scores
- Dividing by the sum makes the results add up to 1

**Gotcha: overflow.** `e^275` is too big for the number type. **Fix: stable softmax.** Subtract the biggest logit first. The answer is mathematically the same, but the numbers stay small.

$$
\sigma (\mathbf{z})_{i}=\frac{e^{z_{i}-\max (\mathbf{z})}}{\sum _{j=1}^{K}e^{z_{j}-\max (\mathbf{z})}}
$$

### 2.2 Cross-Entropy: how surprised is the model?

**Intuition:** if the model gave the right answer a high probability, the loss is small. If it gave the right answer a tiny probability, the loss is huge.

$$
L=-\sum _{c=1}^{K}y_{c}\log (p_{c})
$$

- `y_c` is 1 for the correct token and 0 for all others (one-hot)
- `p_c` is the model's probability for token `c`

Since only the correct token has `y = 1`, this simplifies to `L = -log(p_correct)`.

**Gotcha: log(0).** If the model gives the right answer probability 0, we get `-∞`. **Fix:** combine softmax and the log into one formula, so we never take the log of a raw probability:

$$
L=-z_{c}+\max (\mathbf{z})+\log \sum _{j=1}^{K}e^{z_{j}-\max (\mathbf{z})}
$$

This works because `log(e^(z_c - max))` just equals `z_c - max`. It's a plain subtraction, so no `log(0)` can happen, however wrong the model is.

### 2.3 Tiny Worked Example

Vocabulary of 3 tokens. The model outputs logits `[2.0, 1.0, 0.1]`, and the correct token is **index 0**.

| Step | Result |
|---|---|
| Subtract max (2.0) | `[0, -1, -1.9]` |
| Exponentiate | `[1.000, 0.368, 0.150]` (sum = 1.518) |
| Softmax probabilities | `[0.659, 0.242, 0.099]` |
| Loss = `-log(0.659)` | **0.417** |
| Gradient = `softmax - one_hot` | `[-0.341, 0.242, 0.099]` |

Read the gradient like this: the correct token (index 0) gets a **negative** gradient, so its logit should go **up**. The wrong tokens get **positive** gradients, so their logits should go **down**. That's the whole learning signal.

---

## 3. Backward Pass: Who Is to Blame?

### 3.1 The goal

Weights start random. Training means nudging them so the loss goes down. For each weight `W` we need to know:

> "If I increase this weight a tiny bit, does the loss go up or down, and by how much?"

That's the **gradient** `dL/dW`.

- Positive -> increasing `W` raises the loss -> **decrease** `W`
- Negative -> **increase** `W`

```
W = W - learning_rate * dL/dW
```

Do this for every weight, a few thousand times. That's **gradient descent**.

### 3.2 Why not just try it?

The naive way: nudge one weight, re-run the forward pass, see how the loss changed. That's the **numerical gradient**. With 100,000 weights it needs about 200,000 forward passes **per step**. Way too slow.

**Backpropagation** gets *all* the gradients in **one backward pass**, costing about the same as one forward pass.

### 3.3 The chain rule

Say `x -> a -> b -> L`. How does wiggling `x` change `L`? Multiply the effects along the path:

```
dL/dx = dL/db * db/da * da/dx
```

If `x` moves a bit, `a` moves, which moves `b`, which moves `L`. Multiply the effects.

Each factor is a **local gradient**: the derivative of one simple operation. We work **backward from the loss**, reusing each result:

1. `dL/db` (comes from the loss function)
2. `dL/da = dL/db * db/da`
3. `dL/dx = dL/da * da/dx`

### 3.4 The one rule every layer follows

> Given the gradient arriving from the layer after me (`dout`), multiply by my local derivative and pass the result backward.

Every layer implements the same interface:

```python
def backward(self, dout):
    # dout: gradient of loss w.r.t. this layer's output
    # returns: gradient of loss w.r.t. this layer's input
    # side effect: store gradients of any parameters (self.dW, self.db, ...)
    return din
```

Once every layer has this, backprop is just **calling `backward` in reverse order**.

### 3.5 Local gradient cheat sheet

| Operation | Forward | Backward |
|---|---|---|
| Add | `y = a + b` | `dL/da = dL/dy`, `dL/db = dL/dy` (gradient is **copied** to both) |
| Scalar multiply | `y = c·x` | `dL/dx = dL/dy · c` |
| Elementwise multiply | `y = a * b` | `dL/da = dL/dy · b`, `dL/db = dL/dy · a` (they **swap**) |
| Matmul | `Y = X @ W` | `dL/dX = dL/dY @ W.T`, `dL/dW = X.T @ dL/dY` |
| ReLU | `y = max(x, 0)` | `dL/dx = dL/dy · (x > 0)` (a mask: positives pass, the rest are blocked) |
| Reshape | | reshape back |
| Transpose | | transpose back |
| Concatenate | | split into the original pieces |

**Matmul is the most important one.** If you get a shape error, you almost always forgot a transpose.

**Residuals are easy:** `y = x + f(x)` splits the gradient into two paths:

```
dx = dy + f'(x) · dy
      ^        ^
  direct     through the sublayer
```

The direct path passes `dy` through unchanged. That's the "highway" that keeps deep networks trainable.

<details>
<summary><b>Details: LayerNorm backward (optional)</b></summary>

$$
dx = \text{inv\_std} \cdot \bigl(d\hat{x} - \text{mean}(d\hat{x}) - \hat{x} \cdot \text{mean}(d\hat{x} \cdot \hat{x})\bigr)
$$

where `dx̂ = dout · γ`, `x̂` is the normalized input, and all means are over the feature axis. The `mean(dx̂)` term accounts for the shift (mean subtraction), and the `x̂ · mean(dx̂ · x̂)` term accounts for the scale (dividing by the std).

</details>

### 3.6 Softmax + Cross-Entropy backward (the nice one)

Combined, the backward is surprisingly clean:

$$
\frac{\partial L}{\partial \text{logits}} = \frac{\text{softmax}(\text{logits}) - \text{one\_hot}(y)}{B \cdot T}
$$

No logs, no max-subtraction. (You saw this in the worked example above.) The `B · T` is the number of tokens in the batch, so the loss is an average.

### 3.7 Gradient checking: how to know your backward is right

**Never trust an unchecked backward.** A small bug buried inside stacked blocks is almost invisible in the loss curve.

**The idea:** compare your backward's answer to a numerical estimate.

- Nudge a weight by `+ε` -> loss `L⁺`
- Nudge it by `-ε` -> loss `L⁻`
- Numerical gradient ≈ `(L⁺ - L⁻) / (2ε)`

**The scalar-loss trick** (to test one layer on its own): run `out = layer.forward(x)`, pick a random `dout`, and define:

```
L = sum(out * dout)
```

Now `dL/dout = dout` exactly, so you control the upstream gradient.

**How to judge the result:**

| Relative error | Verdict |
|---|---|
| < 1e-5 | Correct |
| 1e-5 to 1e-3 | Borderline (small bug, or error piling up) |
| > 1e-3 | Bug in the formula |

**What the failure pattern tells you:**

- Only `dx` fails, parameters pass -> bug in the `dx` path (the last step of backward)
- Only `dW` fails, `dx` passes -> bug in the parameter gradient (check the `.T @` pattern or the sum axis)
- Everything fails -> bug in a shared intermediate, or in the loss itself

**Workflow:** check MLP -> LayerNorm -> attention -> embedding, **one at a time**, and only *then* chain them together.

---

## 4. The Optimizer: Turning Gradients into Steps

Backprop tells us the gradients. The **optimizer** decides how big a step to take.

### 4.1 Plain SGD and its two weaknesses

```
W = W - lr * dL/dW
```

1. **One learning rate for everything.** Some weights see huge gradients, others tiny ones. One `lr` is too big for some and too small for others.
2. **Noisy steps.** Each gradient comes from one small batch, so it's only approximately right. Following it blindly makes you zigzag.

### 4.2 Adam: momentum + per-weight step size

**Intuition:** SGD with memory. It averages recent gradients to smooth out the noise, and it scales each weight's step by how large that weight's gradients usually are.

Adam keeps two running averages for each weight:

**First moment `m` (smoothed direction):**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Gradients keep pointing the same way -> `m` grows. They keep flipping -> `m` shrinks.

**Second moment `v` (typical size of the gradient):**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**Bias correction.** Both start at 0, so early values are too small (at step 1, `m` is only 10% of the real gradient). We fix it:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The correction fades away as `t` grows.

**The update:**

$$
\theta_t = \theta_{t-1} - \text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

How to read the ratio `m̂ / √v̂`:

- Steady direction, stable size -> ratio ≈ ±1 -> step ≈ `lr`
- Noisy or tiny gradient -> small ratio -> small step
- One-off huge spike -> `m̂` rises slowly, `√v̂` rises fast -> the spike is damped

So no weight ever moves much more than `lr` per step, which is why Adam is stable.

**Typical settings:**

| Name | Typical value | Meaning |
|---|---|---|
| lr | 3e-4 (for transformers) | base step size |
| β₁ | 0.9 | `m` remembers about the last 10 steps |
| β₂ | 0.999 | `v` remembers about the last 1000 steps |
| ε | 1e-8 | avoids dividing by zero |

(`β₂` is bigger because squared gradients jitter more, so they need a longer average to settle.)

### 4.3 AdamW: Adam with proper weight decay

**What is weight decay?** A gentle pull that shrinks weights toward zero each step. It fights overfitting.

**The problem with L2 inside Adam:** if you add `λ·W` into the gradient, it gets mixed into `m` and `v` and then rescaled by Adam's ratio. So weights with small gradients get **more** decay and weights with big gradients get **less**. That's uneven and unpredictable.

**The fix: decouple it.** Leave the gradient alone and shrink the weight directly:

```
m, v   -> updated with the plain gradient (no decay inside)
theta -= lr * m_hat / (sqrt(v_hat) + eps)      # the Adam step
theta -= lr * wd * theta                       # decay, uniform for all weights
```

**The code change** (one extra line in the update loop):

```python
if p.ndim >= 2:
    p -= self.lr * self.wd * p
```

**Why `ndim >= 2`?** It applies decay only to weight matrices (WQ, WK, WV, WO, W1, W2, W_out) and **skips**:

- **Biases:** they shift, not scale, so pulling them toward 0 is wrong
- **LayerNorm `γ` and `β`:** decaying `γ` to 0 would kill LayerNorm's ability to rescale

**Does it matter?** At small scale (100k-500k params, a few thousand steps), Adam and AdamW are almost identical. It starts to matter when you train longer, scale up, or want to tune `weight_decay` as a clean, meaningful knob.

### 4.4 Learning-rate schedule: warmup + cosine decay

```python
def get_lr(step, max_steps, base_lr, warmup=200):
    if step < warmup:
        return base_lr * step / warmup   # linear warmup
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

- **Warmup:** start at 0 and ramp up over the first ~200 steps. Early gradients are noisy, and big steps then can derail training.
- **Cosine decay:** slowly shrink `lr` to 0. Small steps late in training let the model settle in.

Usually worth 5-10% better final validation loss.

---

## 5. Training Loop Summary

| Phase | What happens | Cost |
|---|---|---|
| Forward | input -> predictions -> loss | 1 pass |
| Backward | loss -> input, computing every `dL/dW` along the way | ~1 pass |
| Update | optimizer changes every weight | trivial |

**TL;DR:** Backprop is the chain rule run backward through the network, reusing each step's gradient, so one pass gives every weight's gradient. Each layer's backward is just: *multiply the incoming gradient by my local derivative and pass it on.*

---

## 6. Reading a Training Run

### 6.1 The loss curve

Watch two things: **the level** and **the gap** between train and val.

| What you see | What it means |
|---|---|
| Loss stuck at `log(vocab_size)` | Gradients are zero or broken. Check `dW_out` and the `+`s in `Transformer.backward` |
| Loss drops smoothly | Working |
| Loss goes NaN early | `lr` too high, or a bug creating inf/nan |
| Loss drops, then explodes | Exploding gradients. Lower `lr` or clip gradients |
| Train drops, val flat or rising | Overfitting. Smaller model, more data, or stop earlier |
| Train ≈ val, both dropping | Underfitting. Train longer or use a bigger model |
| Train ≪ val (big gap) | Overfitting. Regularize or stop |

**Sanity check at step 0:** with fresh random weights, loss should be **exactly `log(vocab_size)`** (random guessing). Lower than that means information is leaking through a bug. Higher means the weights are badly scaled.

### 6.2 Train vs val noise

Train loss is measured on **one small batch** (e.g. 4 × 32 = 128 tokens), so it zigzags. Val loss is averaged over many batches (e.g. 20 × 128 = 2560 tokens), so it's smooth. **That's normal.** Don't panic about a jumpy train loss.

### 6.3 Reading the generated text (character-level model)

| Already learned | Not learned yet |
|---|---|
| Real short words ("the", "is", "for") | Meaning (fake words like "rigents") |
| Spaces in the right places | Long-range coherence (sentences don't connect) |
| Capital letters at sentence starts | Consistent grammar (tenses get mixed) |
| Punctuation | A real vocabulary (needs subword tokens) |
| English-shaped pseudo-words ("shought") | |

A character-level model can copy the **shape** of English but not its **meaning**. Real GPTs use subword tokenization (BPE, ~50k vocab) to produce real words.

### 6.4 What to expect

| Params | Steps | Expected val loss | Sample quality |
|---|---|---|---|
| 200k | 5k | ~1.85 | Punctuation and spaces OK, some real words, mostly gibberish |
| 1.2M | 20k | ~1.4-1.5 | More real words, longer phrases, some grammar |
| 10M | 100k+ | ~1.2 | Coherent sentences, mostly grammatical |
| 124M (GPT-2) | 300k | ~0.9 | Coherent paragraphs, some reasoning |

---

## 7. Practical Details That Aren't Obvious

### 7.1 Why init with std = 0.02?

Embeddings and weight matrices start as `randn * 0.02` (the GPT-2 convention).

- Too big (e.g. 1.0) -> activations blow up through the layers, training diverges
- Too small -> gradients vanish, nothing learns

**Exception:** LayerNorm `γ` starts at **1** and `β` at **0**, so LayerNorm begins as roughly "do nothing."

### 7.2 Temperature and sampling

When generating text, divide the logits by a temperature before softmax:

```
logits_last = logits_last / temperature
```

| Temperature | Effect |
|---|---|
| `1.0` | Sample from the model's true distribution |
| `< 1` (e.g. 0.7) | Sharper: more confident, more repetitive |
| `> 1` (e.g. 1.3) | Flatter: more random, more creative |
| `-> 0` | Approaches always picking the top token (greedy) |

**Always sample instead of always taking the top token.** Always taking the top one gives dull loops ("the the the..."). Sampling gives variety, at the cost of the occasional nonsense.

---

## 8. Explanations
| Component | What it does (plain words) | Shapes / key formula | Backward rule |
|---|---|---|---|
| Token embedding | Look up a vector per token | `(vocab_size, d_model)` | Add the gradient into the rows that were used |
| Position embedding | Look up a vector per position | `(block_size, d_model)` | Same as above |
| LayerNorm | Keep numbers at a steady scale | `γ·(x-μ)/√(σ²+ε)+β` | Closed-form `dx` (see Details) |
| Q/K/V projections | Make query, key, value | `Q = X@WQ` etc. | Matmul rule |
| Attention | Mix in info from earlier tokens | `softmax(QKᵀ/√d_k + mask) @ V` | Matmul + softmax + mask |
| Multi-head | Several attentions in parallel | Concatenate heads | Split the gradient back |
| MLP | Think about each token | `Linear -> ReLU -> Linear` | Matmul + ReLU mask |
| Residual | Keep the input, add a correction | `x + f(x)` | Gradient copied to both paths |
| Output head | Vectors -> logits | `(d_model, vocab_size)` | Matmul rule |
| Softmax + CE | Logits -> loss | `-z_c + max + log Σ e^(z-max)` | `(softmax - one_hot) / (B·T)` |
| AdamW | Turn gradients into steps | `lr · m̂/(√v̂+ε)` plus decay on `ndim ≥ 2` | n/a (optimizer) |

**Matmul backward, once more:** `dL/dX = dL/dY @ W.T` and `dL/dW = X.T @ dL/dY`.