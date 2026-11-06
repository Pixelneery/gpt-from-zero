import os
import numpy as np
from tokenizer import chars, encode, decode, get_batch
from gpt import GPT, get_params, generate

# --- config (must match training!!) ---
vocab_size = len(chars)
block_size = 128
d_model    = 128
n_heads    = 4
n_layers   = 6

# --- verify checkpoint ---
ckpt_path = "ckpt_18000.npz"
if not os.path.exists(ckpt_path):
    print(f"Model ({ckpt_path}) not found")
    print("Available models:", [f for f in os.listdir(".") if f.endswith(".npz")])
    raise SystemExit(1)

# --- build architecture (weights overwritten) ---
np.random.seed(0)
model = GPT(vocab_size, block_size, d_model, n_heads, n_layers)

# --- load weights in place ---
ckpt = np.load(ckpt_path)
for name, p in get_params(model):
    p[...] = ckpt[name]
ckpt.close()
print(f"loaded {len(ckpt.files)} arrays")

xb, yb = get_batch("val")
_, loss = model.forward(xb, yb)
print(f"val loss: {loss:.4f}")

# --- generate ---
prompt = "You are guilty! Please"
out = generate(model, np.array([encode(prompt)]),
               max_new_tokens=1000, temperature=0.8)
print(f"\n{prompt}{decode(out[0].tolist())[len(prompt):]}")