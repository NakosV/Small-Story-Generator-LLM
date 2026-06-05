import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# There used to be more datasets here that's why I have this approach. Even though right now there only one, this is highly scalable
datasets = {'Dataset 1': 'where_your_dataset_is/dataset.txt'}

model_dir = 'your/own/path'
os.makedirs(model_dir, exist_ok=True)

block_size = 256
batch_size = 16
learning_rate = 0.0001
max_iterations = 10000
evaluation_interval = 500
evaluation_iterations = 200
checkpoint_interval = 1000
n_emb = 512
dropout = 0.2
n_layer = 8
n_head = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Useful to speed up the training process
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

enc = tiktoken.get_encoding("gpt2")
vocabulary_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special=set())
decode = lambda l: enc.decode(l)

# The Self-Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * C ** -0.5 
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)
        return weight @ v

# It just connects many Self-Attention Heads together
class MultiHead_Attention(nn.Module):
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_of_heads)])
        self.projection = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

# A simple Neural Network that makes connections
class Feed_Forward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),)

    def forward(self, x):
        return self.net(x)

# Connects the Attention and the Feed_Forward 
class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.self_attention_heads = MultiHead_Attention(n_head, head_size)
        self.feed_forward = Feed_Forward(n_emb)
        self.layer_normalization1 = nn.LayerNorm(n_emb)
        self.layer_normalization2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.self_attention_heads(self.layer_normalization1(x))
        x = x + self.feed_forward(self.layer_normalization2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(* [Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.layer_normalization_final = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_normalization_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# The actual training of the model
def train(name, data_path):
    print(f"Training: {name.upper()}\n")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Vocabulary size: {vocabulary_size}")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data  = data[n:]

    print(f"Training tokens: {len(train_data):,}")
    print(f"Test tokens: {len(test_data):,}")

    def get_batch(split):
        d  = train_data if split == 'train' else test_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x  = torch.stack([d[i: i + block_size] for i in ix])
        y  = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def get_loss():
        out = {}
        model.eval()
        for split in ['train', 'test']:
            losses = torch.zeros(evaluation_iterations)
            for i in range(evaluation_iterations):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = LanguageModel(vocabulary_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iterations, eta_min=1e-5
    )
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M\n")
    torch.cuda.empty_cache()

    checkpoint_dir = os.path.join(model_dir, name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_times = []

    for iteration in range(max_iterations):
        t_start = time.time()

        if iteration % evaluation_interval == 0:
            losses = get_loss()
            elapsed = sum(epoch_times[-evaluation_interval:]) if epoch_times else 0
            remaining_epochs = max_iterations - iteration
            avg_epoch_time = (elapsed / min(len(epoch_times), evaluation_interval)) if epoch_times else 0
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds)) if iteration > 0 else '--:--:--'
            print(f"Step {iteration:>5} | Training Loss: {losses['train']:.4f} | "
                  f"Test Loss: {losses['test']:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | ETA: {eta_str}")

        if iteration > 0 and iteration % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocabulary_size': vocabulary_size,
            }, ckpt_path)
            print(f"checkpoint_epoch_{iteration}.pt")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_times.append(time.time() - t_start)

    final_path = os.path.join(model_dir, f'{name}_model_BPE.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocabulary_size':  vocabulary_size,
        'hyperparameters': {
            'block_size': block_size, 'n_emb': n_emb,
            'n_layer': n_layer, 'n_head': n_head, 'dropout': dropout,
        }
    }, final_path)
    print(f"\nFinal model saved: {name}_model_BPE.pt")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample  = decode(model.generate(context, max_new_tokens=600)[0].tolist())
    print(f"\nSample:\n{sample}\n")

for name, path in datasets.items():
    train(name, path)

print("\nTraining Completed")