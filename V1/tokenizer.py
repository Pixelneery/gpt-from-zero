import numpy as np

# Get file
with open("script.txt", "r", encoding="UTF-8") as f:
    text = f.read()

# Extract vocabs
chars = sorted(list(set(text)))

# Mapping
stoi = {
    ch:i for i, ch in enumerate(chars)
}
itos = {
    i:ch for i, ch in enumerate(chars)
}
# Gives each character a index

def encode(s):
    # Converts string to list of integers
    return [stoi[c] for c in s]

def decode(l):
    # Converts list of integers to string
    return ''.join([itos[i] for i in l])

# Tokenize everything and convert to numpy array
data = np.array(encode(text), dtype=np.int16)

# Split 90% for training, 10% for validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128 # The context length of the model
batch_size = 4 # How many sequence to process in parallel

def get_batch(split):
    # Train or validate?
    data_split = train_data if split == 'train' else val_data
    
    # Generate random starting indices for each sequence in the batch
    ix = np.random.randint(0, len(data_split) - block_size, size=(batch_size,))

    # Stack sequences into a batch tensor (batch_size, block_size)
    x = np.stack([data_split[i:i+block_size] for i in ix])
    y = np.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x,y