import numpy as np
from tokenizer import get_batch, chars, encode, decode

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class Embedding:
    def __init__(self, num_embeddings, embedding_dim, std=0.02):
        # Turn token into random vector
        self.W = np.random.randn(num_embeddings, embedding_dim) * std

    def forward(self, idx):
        # Pass itself
        return self.W[idx]

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

        # get training values
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        # split into heads
        # (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2)
        scores = scores / np.sqrt(self.head_dim)

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

        weights = softmax(scores)
        out = weights @ V

        # merge heads back
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

        return out @ self.WO

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model) # gain, init 1
        self.beta = np.zeros(d_model) # bias, init 0
        self.eps = eps

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        x_hat = (x-mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class MLP:
    def __init__(self, d_model, hidden_mult=4):
        hidden = d_model * hidden_mult
        self.W1 = np.random.randn(d_model, hidden) * 0.02
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0) # ReLU
        out = h @ self.W2 + self.b2
        return out

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

class GPT:
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = TokenAndPositionalEmbedding(vocab_size, block_size, d_model)
        self.blocks = [Transformer(d_model, n_heads) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, idx):
        x = self.emb.forward(idx)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        logits = x @ self.W_out
        return logits

def generate(gpt, idx, max_new_tokens=100, temperature=1.0, vocab_size=None, block_size=None):
    for _ in range(max_new_tokens):
        # crop context to block_size, the model can't handle more
        idx_cond = idx[:, -gpt.block_size:]

        # forward, take logits at the last position
        logits = gpt.forward(idx_cond)
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