import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.2
temperature = 0.8
top_k = 20

# A seed for reproducibility
torch.manual_seed(1337)

# Open the Victor Hugo dataset file
with open('data/victor_hugo-texts.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract unique characters from the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers based on alphabetical order
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers-> string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # First 90% for train, rest val
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # (B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (B,T)
    x, y = x.to(device), y.to(device)
    return x, y

# Disable gradients for evaluation
@torch.no_grad()
def estimate_loss():
    out = {}
    # Evaluation mode (turns off dropout, set LayerNorm to eval mode)
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Average loss over all tokens in the batch
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # Average loss over eval_iters batches to avoid fluctuation
        out[split] = losses.mean()
    # Back to training mode
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # Each token embedding is projected into Key, Query and Value spaces
        # Using three learned projection matrices Wk, Wq, Wv
        # So that each head only works in a head_size-dimentional subspace
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Define a lower triangular matrix (register_buffer = non trainable param)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Drop attention connections to prevent the model from relying too heavily on one specific token
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention scores (affinities), scaling to keep variance stable
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        # Attention distribution over only previous tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create separate head objects
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs of each head along C dimension
        # out dim = head_size * num_heads = n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Projection does cross-head feature mixing to combine features from diff heads
        # Because after concatenation heads outputs are independent
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    # FeedForward is done independently per token (computation within each token)
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (FF) """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # Normalizes across the embedding dimension to stabilize activations
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Use pre-layer normalisation for a more stable optimization
        # Use residual connections to learn how to improve x instead of learning the new x
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token & position embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stacked transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        # For each token we get logits over all possible next characters
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        # For each token we predict the logits over next character
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        # Compute loss in training only
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Autoregressive generation
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Use temperature < 1 for more determinism
            logits = logits / temperature
            # Apply Top-K
            values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold,
                                torch.full_like(logits, -float("Inf")),
                                logits)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# For plotting loss curves
train_losses = []
val_losses = []
steps = []

for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        # Append the losses to the lists for plotting
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        steps.append(iter)
        # Print the current losses
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Plot the training and validation loss curves
plt.figure()
plt.plot(steps, train_losses, label="Train Loss")
plt.plot(steps, val_losses, label="Validation Loss")

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

# Save the plot as an image 
plt.grid(True)
plt.savefig("results/loss_plot.png")
plt.show()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))