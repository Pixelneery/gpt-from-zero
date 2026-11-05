import numpy as np
from tokenizer import get_batch, chars, encode, decode
import time

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def numerical_grad(f, arr, eps=1e-6):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps; fp = f()
        arr[i] = old - eps; fm = f()
        grad[i] = (fp - fm) / (2 * eps)
        arr[i] = old
    return grad

def cross_entropy(logits, targets):
    B, T, V = logits.shape
    logits2 = logits.reshape(B*T, V)
    targets2 = targets.reshape(B*T)

    probs = softmax(logits2)

    # loss = -mean(log(prob of correct token))
    loss = -np.log(probs[np.arange(B*T), targets2] + 1e-12).mean()

    # gradient
    dlogits2 = probs.copy()
    dlogits2[np.arange(B*T), targets2] -= 1.0 # softmax - one_hot
    dlogits2 /= (B*T)

    return loss, dlogits2.reshape(B, T, V)

def get_lr(step, max_steps, base_lr, warmup=200):
    if step < warmup:
        return base_lr * step / warmup # linear warmup
    progress = (step - warmup) / (max_steps - warmup)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress)) # cosine decay

class Embedding:
    def __init__(self, num_embeddings, embedding_dim, std=0.02):
        # Turn token into random vector
        self.W = np.random.randn(num_embeddings, embedding_dim) * std

    def forward(self, idx):
        # Pass itself
        self.idx = idx
        return self.W[idx]

    def backward(self, dout):
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.idx, dout)

class TokenAndPositionalEmbedding:
    def __init__(self, vocab_size, block_size, d_model):
        self.token_emb = Embedding(vocab_size, d_model) # the vector for token ID i
        self.pos_emb = Embedding(block_size, d_model) # the vector for position i

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_emb.forward(idx)
        
        pos_ids = np.arange(T)[None, :]
        pos = self.pos_emb.forward(pos_ids)

        return tok + pos

    def backward(self, dout):
        self.token_emb.backward(dout)
        self.pos_emb.backward(dout.sum(axis=0, keepdims=True))

'''
class CausalAttention:
    def __init__(self, d_model, head_dim=None):
        head_dim = head_dim or d_model
        self.d_model = d_model
        self.head_dim = head_dim

        self.WQ = np.random.randn(d_model, head_dim) * 0.02
        self.WK = np.random.randn(d_model, head_dim) * 0.02
        self.WV = np.random.randn(d_model, head_dim) * 0.02

    def forward(self, x):
        B, T, _ = x.shape

        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        # transpose: (B, T, head_dim) -> (B, head_dim, T)
        scores = Q @ K.transpose(0, 2, 1)
        scores = scores / np.sqrt(self.head_dim)

        # causal mask
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

        weights = softmax(scores)

        out = weights @ V

        return out

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.heads = [CausalAttention(d_model, self.head_dim) for _ in range(n_heads)]
        self.WO = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x):
        outs = [head.forward(x) for head in self.heads]
        concat = np.concatenate(outs, axis=-1)
        return concat @ self.WO
'''

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0 # Ensure we can divide the to the heads evenly
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.WQ = np.random.randn(d_model, d_model) * 0.02
        self.WK = np.random.randn(d_model, d_model) * 0.02
        self.WV = np.random.randn(d_model, d_model) * 0.02
        self.WO = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x):
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        self.x = x # Caching

        # get training values
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        # split into heads
        # (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        Q = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        self.Q, self.K, self.V = Q, K, V # Caching

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(Dh)

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        self.mask = mask # Caching
        scores = np.where(mask, -np.inf, scores)

        weights = softmax(scores)
        self.weights = weights # Caching

        out = weights @ V
        # merge heads back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        self.out_merged = out # Caching

        return out @ self.WO

    def backward(self, dout):
        B, T, D = dout.shape
        H, Dh = self.n_heads, self.head_dim

        # Through WO:  y = out_merged @ WO
        dout2 = dout.reshape(-1, D)
        out_merged2 = self.out_merged.reshape(-1, D)
        self.dWO = out_merged2.T @ dout2
        dpre = dout @ self.WO.T

        # un-transpose back to heads
        dpre = dpre.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # Through out = weights @ V
        dW_attn = dpre @ self.V.transpose(0, 1, 3, 2)
        dV = self.weights.transpose(0, 1, 3, 2) @ dpre

        # Through softmax
        # dL/dscores_i = w_i * (dW_attn_i - sum_j w_j * dW_attn_j)
        dot = (dW_attn * self.weights).sum(axis=-1, keepdims=True)
        dScores = self.weights * (dW_attn - dot)

        # Through scaling + mask
        dScores = dScores / np.sqrt(Dh)
        dScores = np.where(self.mask, 0.0, dScores)

        # Through scores Q @ K^T
        dQ = dScores @ self.K
        dK = dScores.transpose(0, 1, 3, 2) @ self.Q

        # un-transpose Q, K, V back to (B,T,D)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Through Q=x@WQ, K=x@WK, V=x@WV
        x2 = self.x.reshape(-1, D)
        dQ2 = dQ.reshape(-1, D)
        dK2 = dK.reshape(-1, D)
        dV2 = dV.reshape(-1, D)

        self.dWQ = x2.T @ dQ2
        self.dWK = x2.T @ dK2
        self.dWV = x2.T @ dV2

        dx = ((dQ2 @ self.WQ.T).reshape(B, T, D)
            + (dK2 @ self.WK.T).reshape(B, T, D)
            + (dV2 @ self.WV.T).reshape(B, T, D))
        return dx

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model) # gain, init 1
        self.beta = np.zeros(d_model) # bias, init 0
        self.eps = eps

    def forward(self, x):
        self.x = x
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mu) * self.inv_std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        # dγ, dβ, sum over everything except the feature axis
        sum_axes = tuple(range(dout.ndim - 1))
        self.dgamma = (dout * self.x_hat).sum(axis=sum_axes)
        self.dbeta = dout.sum(axis=sum_axes)

        # dx̂ = dout · γ
        dx_hat = dout * self.gamma

        # dx through x̂ → x (the closed form)
        mean_dx_hat = dx_hat.mean(axis=-1, keepdims=True)
        mean_dx_hat_x_hat = (dx_hat * self.x_hat).mean(axis=-1, keepdims=True)

        dx = self.inv_std * (dx_hat - mean_dx_hat - self.x_hat * mean_dx_hat_x_hat)
        return dx

class MLP:
    def __init__(self, d_model, hidden_mult=4):
        hidden = d_model * hidden_mult
        self.W1 = np.random.randn(d_model, hidden) * 0.02
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        self.x = x # caching 
        h = x @ self.W1 + self.b1
        self.a = np.maximum(h, 0) # cache post-ReLU
        out = self.a @ self.W2 + self.b2
        return out

    def backward(self, dout):
        # dout: (B, T, d_model), gradient of loss w.r.t. thiS layer's output
        B, T, D = dout.shape
        dout2 = dout.reshape(-1, D) # (B*T, d_model)
        a2 = self.a.reshape(-1, self.a.shape[-1]) # (B*T, hidden)

        # y = a @ w2 + b2
        self.dW2 = a2.T @ dout2
        self.db2 = dout2.sum(axis=0)
        da = dout @ self.W2.T

        # a = ReLU(h)  →  gradient flows where h > 0
        dh = da * (self.a > 0)

        # h = x @ w1 + b1
        dh2 = dh.reshape(-1, dh.shape[-1])
        x2 = self.x.reshape(-1, self.x.shape[-1])
        self.dW1 = x2.T @ dh2
        self.db1 = dh2.sum(axis=0)
        dx = dh @ self.W1.T

        return dx

class Transformer:
    def __init__(self, d_model, n_heads):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x)) # pre-norm attention + residual
        x = x + self.mlp.forward(self.ln2.forward(x)) # pre-norm MLP + residual
        return x

    def backward(self, dx_out):
        d_mlp_in = self.mlp.backward(dx_out)
        d_ln2_in = self.ln2.backward(d_mlp_in)
        dx_after_attn = dx_out + d_ln2_in

        d_attn_in = self.attn.backward(dx_after_attn)
        d_ln1_in = self.ln1.backward(d_attn_in)
        return dx_after_attn + d_ln1_in

class GPT:
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = TokenAndPositionalEmbedding(vocab_size, block_size, d_model)
        self.blocks = [Transformer(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, idx, targets=None):
        x = self.emb.forward(idx)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        self.x_final = x # caching
        logits = x @ self.W_out

        if targets is not None:
            loss, _ = cross_entropy(logits, targets)
            return logits, loss
        return logits, None

    def backward(self, dlogits):
        B, T, V = dlogits.shape
        D = self.W_out.shape[0]

        dlogits2 = dlogits.reshape(B*T, V)
        x_final2 = self.x_final.reshape(B*T, D)
        self.dW_out = x_final2.T @ dlogits2
        dx = dlogits @ self.W_out.T

        dx = self.ln_f.backward(dx)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        self.emb.backward(dx)

def generate(gpt, idx, max_new_tokens=100, temperature=1.0, vocab_size=None, block_size=None):
    for _ in range(max_new_tokens):
        # crop context to block_size, the model can't handle more
        idx_cond = idx[:, -gpt.block_size:]

        # forward, take logits at the last position
        logits, _ = gpt.forward(idx_cond)
        logits_last = logits[:, -1, :]

        # temperature-scaled softmax
        logits_last = logits_last / temperature
        probs = softmax(logits_last)

        # sample 1 token per batch element
        next_ids = np.array([
            np.random.choice(gpt.vocab_size, p=probs[b])
            for b in range(probs.shape[0])
        ])
        next_ids = next_ids[:, None]

        # append to sequence
        idx = np.concatenate([idx, next_ids], axis=1)

    return idx

class AdamW:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_grads):
        self.t += 1
        for name, p, g in params_grads:
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)

            # moments (identical to Adam, gradient is NOT touched by decay)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g * g)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # decoupled weight decay, only on 2D weight matrices
            if p.ndim >= 2:
                p -= self.lr * self.wd * p

            # adaptive update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
 
def get_params_grads(gpt):
    """Returns a list of (name, param, grad) with live references."""
    pg = [
        ("emb.tok", gpt.emb.token_emb.W, gpt.emb.token_emb.dW),
        ("emb.pos", gpt.emb.pos_emb.W,   gpt.emb.pos_emb.dW),
    ]
    for i, b in enumerate(gpt.blocks):
        pg += [
            (f"b{i}.ln1.g",  b.ln1.gamma, b.ln1.dgamma),
            (f"b{i}.ln1.b",  b.ln1.beta,  b.ln1.dbeta),
            (f"b{i}.WQ",     b.attn.WQ,   b.attn.dWQ),
            (f"b{i}.WK",     b.attn.WK,   b.attn.dWK),
            (f"b{i}.WV",     b.attn.WV,   b.attn.dWV),
            (f"b{i}.WO",     b.attn.WO,   b.attn.dWO),
            (f"b{i}.ln2.g",  b.ln2.gamma, b.ln2.dgamma),
            (f"b{i}.ln2.b",  b.ln2.beta,  b.ln2.dbeta),
            (f"b{i}.mlp.W1", b.mlp.W1,    b.mlp.dW1),
            (f"b{i}.mlp.b1", b.mlp.b1,    b.mlp.db1),
            (f"b{i}.mlp.W2", b.mlp.W2,    b.mlp.dW2),
            (f"b{i}.mlp.b2", b.mlp.b2,    b.mlp.db2),
        ]
    pg += [
        ("ln_f.g", gpt.ln_f.gamma, gpt.ln_f.dgamma),
        ("ln_f.b", gpt.ln_f.beta,  gpt.ln_f.dbeta),
        ("W_out",  gpt.W_out,      gpt.dW_out),
    ]
    return pg

np.random.seed(0)

vocab_size = len(chars)
block_size = 128
d_model    = 128
n_heads    = 4
n_layers   = 6

gpt = GPT(vocab_size, block_size, d_model, n_heads, n_layers)
opt = AdamW(lr=3e-4, weight_decay=0.01)

def evaluate(split, n_batches=20):
    losses = []
    for _ in range(n_batches):
        xb, yb = get_batch(split)
        _, loss = gpt.forward(xb, yb)
        losses.append(loss)
    return float(np.mean(losses))

import time
t0 = time.time()

def save_checkpoint(gpt, path):
    state = {name: p for name, p, _ in get_params_grads(gpt)}
    np.savez(path, **state)

MAX_STEPS = 20000
for step in range(MAX_STEPS):
    opt.lr = get_lr(step, MAX_STEPS, base_lr=3e-4, warmup=200)
    xb, yb = get_batch("train")
    logits, loss = gpt.forward(xb, yb)
    _, dlogits = cross_entropy(logits, yb)
    gpt.backward(dlogits)
    opt.step(get_params_grads(gpt))

    if step % 500 == 0 or step == MAX_STEPS - 1:
        print(f"step {step:5d} | lr {opt.lr:.2e} | train {loss:.4f} | "
              f"val {evaluate('val'):.4f} | {time.time()-t0:.0f}s")

    if step % 2000 == 0 and step > 0:
        save_checkpoint(gpt, f"ckpt_{step}.npz")

print("\n" + "=" * 40)
print("SAMPLE:")
prompt = np.array([[encode("T")[0]]])
out = generate(gpt, prompt, max_new_tokens=300, temperature=0.8)
print(decode(out[0].tolist()))