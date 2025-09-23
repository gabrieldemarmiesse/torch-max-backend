import torch
import torch.nn as nn
from torch_max_backend import max_backend


class SimpleAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        mask = mask[None, None, :, :]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )
        return self.out_proj(context)


class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.att = SimpleAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            dtype=cfg["dtype"],
        )
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

    def forward(self, in_idx):
        x = self.tok_emb(in_idx)
        x = self.att(x)
        return self.out_head(x)


QWEN3_CONFIG = {
    "vocab_size": 100,
    "emb_dim": 64,
    "n_heads": 4,
    "head_dim": 16,
    "dtype": torch.float32,
}

torch.manual_seed(123)
model = SimpleModel(QWEN3_CONFIG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


input_token_ids_tensor = torch.tensor([[1, 2, 3]], device=device)

model.eval()
model = torch.compile(model, backend=max_backend, fullgraph=True, disable=False)

# Just a single forward pass
with torch.no_grad():
    logits = model(input_token_ids_tensor)
    print("Forward pass completed")
