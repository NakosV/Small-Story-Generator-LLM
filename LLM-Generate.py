import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

model_path = '/where/your/model/is/saved'
writingprompts_path = '/where/your/second/dataset/is'
finetuned_path = '/where/you/want/your/final/model/to/be/saved'

# With the way that I have done it, you need to have the same variables and structure from the model in both of the codes otherwise it will not work

# Same variables as the other code
block_size = 256
n_emb = 512
dropout = 0.2
n_layer = 8
n_head = 8
finetune_batch = 8

# Way lower learning rate in order for the model NOT to forget what it has already learned. We just teach it another writing style on top of the one that it already knows from the training
writingprompts_steps = 10000
writingprompts_lr = 0.00008
max_tokens = 300

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

enc = tiktoken.get_encoding("gpt2")
vocabulary_size = enc.n_vocab
encode = lambda s: enc.encode(s, allowed_special=set())
decode = lambda l: enc.decode(l)

# Same set up of the model as the other code
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_emb, head_size, bias=False)
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

class MultiHead_Attention(nn.Module):
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_of_heads)])
        self.projection = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class Feed_Forward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

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
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.layer_normalization_final = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocabulary_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.layer_normalization_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            top_k_val = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k_val)
            logits[logits < values[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def load_model(path):
    print(f"Loading model from {path}...")
    model = LanguageModel(vocabulary_size).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# A prompt to get the model started
def get_response(model, user_input, max_new_tokens=max_tokens):
    prompt = (
        "Below is a writing prompt. Write a creative story based on it.\n\n"
        f"### Prompt:\n{user_input}\n\n"
        "### Story:\n"
    )
    encoded = encode(prompt)
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    full_text = decode(output)
    response_marker = "### Story:\n"
    if response_marker in full_text:
        response = full_text.split(response_marker)[-1].strip()
        for stop in ["Below is a writing", "###", "\n\n\n"]:
            if stop in response:
                response = response.split(stop)[0].strip()
    else:
        response = full_text[len(decode(encoded)):].strip()
    return response if response else "..."

def run_finetuning(model):
    print(f"Dataset: {writingprompts_path}")
    print(f"Steps: {writingprompts_steps} | LR: {writingprompts_lr}")

    with open(writingprompts_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    max_chars = 20_000_000
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"  Dataset trimmed to {max_chars/1e6:.0f}MB")

    data      = torch.tensor(encode(text), dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=writingprompts_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=writingprompts_steps, eta_min=writingprompts_lr * 0.1
    )
    model.train()

    for step in range(writingprompts_steps):
        ix = torch.randint(len(data) - block_size, (finetune_batch,))
        x  = torch.stack([data[i: i + block_size] for i in ix]).to(device)
        y  = torch.stack([data[i + 1: i + block_size + 1] for i in ix]).to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            print(f"  Step {step:>4}/{writingprompts_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    return model

def chat_loop(model, label=""):
    while True:
        user_input = input("Prompt: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        response = get_response(model, user_input)
        print(f"\nStory:\n{response}\n")

if __name__ == "__main__":
    if os.path.exists(finetuned_path):
        print("\nFound a fine-tuned story model.")
        choice = input("Load it? (yes/no): ").strip().lower()
        if choice == 'yes':
            model = load_model(finetuned_path)
            chat_loop(model, label="[Fine-tuned]")
            exit()

    model = load_model(model_path)
    chat_loop(model, label="[Base Gutenberg]")

    print("\nWould you like to apply fine-tuning (WritingPrompts)?")
    choice = input("(yes/no): ").strip().lower()
    if choice != 'yes':
        print("Fine-tuning skipped")
        exit()

    model = run_finetuning(model)

    torch.save({'model_state_dict': model.state_dict(),
                'vocabulary_size': vocabulary_size}, finetuned_path)
    print(f"Final model saved: {finetuned_path}\n")

    print("Fine-tuning complete! Starting story session...\n")
    chat_loop(model, label="[Fine-tuned]")
