import torch
import torch.nn as nn
from torch_max_backend import max_backend


class SimpleAttention(nn.Module):
    def __init__(self, d_in, num_heads, head_dim=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else d_in // num_heads
        self.d_out = num_heads * self.head_dim

    def forward(self, x):
        b, num_tokens, d = x.shape

        # Simple attention without projections
        x_reshaped = x.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Self-attention scores
        attn_scores = x_reshaped @ x_reshaped.transpose(2, 3)

        # Add mask back
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        mask = mask[None, None, :, :]
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # Apply attention and reshape back
        context = (
            (attn_weights @ x_reshaped)
            .transpose(1, 2)
            .reshape(b, num_tokens, self.d_out)
        )
        return context


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

model.to("cuda")


input_token_ids_tensor = torch.tensor([[1, 2, 3]], device="cuda")

model.eval()
model = torch.compile(model, backend=max_backend, fullgraph=True, disable=False)

# Just a single forward pass
with torch.no_grad():
    logits = model(input_token_ids_tensor)
    print("Forward pass completed")
