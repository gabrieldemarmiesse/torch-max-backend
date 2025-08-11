import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from max_torch_backend import MaxCompiler, get_accelerators
from torch._dynamo import mark_dynamic


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    # Return cos and sin instead of complex exponentials to avoid complex64
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cos, freqs_sin, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cos.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cos.view(*shape), freqs_sin.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # Reshape to separate real and imaginary parts
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # Reshape frequencies for broadcasting
    freqs_cos, freqs_sin = reshape_for_broadcast(freqs_cos, freqs_sin, xq_r)

    # Apply rotation using real arithmetic: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine back to original shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(
            config.n_embd, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.n_embd, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)

        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_head // self.n_kv_heads
        self.head_dim = config.head_dim

        # Precompute frequencies for RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Grouped multi-query attention
        keys = xk.repeat_interleave(self.n_rep, dim=2)
        values = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.full(
            (seqlen, seqlen), float("-inf"), device=scores.device, dtype=scores.dtype
        )
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * config.n_embd / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama2Config:
    def __init__(self, vocab_size, max_seq_len, **kwargs):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Default Llama2-7B parameters
        self.n_embd = 4096
        self.n_layer = 32
        self.n_head = 32
        self.n_kv_heads = 32
        self.head_dim = self.n_embd // self.n_head
        self.multiple_of = 256
        self.norm_eps = 1e-5
        self.attn_pdrop = 0.0

        # Override with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class Llama2(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.max_seq_len is not None
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_name="llama2-7b", **kwargs):
        """
        Load pretrained Llama2 model. For now, creates a model with random weights.
        In a real implementation, this would load actual pretrained weights.
        """
        print(f"Creating Llama2 model: {model_name} (random weights for demo)")

        config_args = {
            "llama2-7b": dict(
                vocab_size=32000, max_seq_len=2048, n_embd=4096, n_layer=32, n_head=32
            ),
            "llama2-13b": dict(
                vocab_size=32000, max_seq_len=2048, n_embd=5120, n_layer=40, n_head=40
            ),
        }

        if model_name not in config_args:
            print(f"Unknown model {model_name}, using llama2-7b config")
            model_name = "llama2-7b"

        config_args[model_name].update(kwargs)
        config = Llama2Config(**config_args[model_name])
        model = cls(config)

        return model

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference mode - only compute logits for the last token
            logits = self.output(h[:, -1:, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop idx to last max_seq_len tokens
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len :]
            )

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def load_tokenizer():
    """
    Simple tokenizer for demo purposes.
    In practice, you would use the actual Llama2 tokenizer.
    """

    class SimpleTokenizer:
        def __init__(self):
            # Create a simple vocab with basic ASCII chars + special tokens
            self.vocab = {chr(i): i for i in range(256)}
            # Add some common tokens
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
            for i, token in enumerate(special_tokens):
                self.vocab[token] = 256 + i

            # Extend to 32000 tokens (Llama2 vocab size) with dummy tokens
            for i in range(260, 32000):
                self.vocab[f"<token_{i}>"] = i

            self.inv_vocab = {v: k for k, v in self.vocab.items()}

        def encode(self, text):
            # Simple character-level encoding
            return [self.vocab.get(c, self.vocab["<unk>"]) for c in text]

        def decode(self, tokens):
            return "".join([self.inv_vocab.get(t, "<unk>") for t in tokens])

    return SimpleTokenizer()


def main():
    device = "cuda" if len(list(get_accelerators())) >= 2 else "cpu"
    print(f"Using device: {device}")

    # Create Llama2 model
    model = Llama2.from_pretrained("llama2-7b")
    model.eval()
    model.to(device)

    # Load tokenizer
    enc = load_tokenizer()

    print("\n" + "=" * 50)
    print("Testing Llama2 generation without torch.compile")
    print("=" * 50)

    prompts = ["The future of artificial intelligence", "Once upon a time"]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            generated = model.generate(
                tokens, max_new_tokens=30, temperature=0.8, top_k=50
            )
            generated_text = enc.decode(generated[0].tolist())
            print(f"Generated: {generated_text}")

    print("\n" + "=" * 50)
    print("Testing Llama2 generation with compiled single forward step")
    print("=" * 50)

    # Compile just the forward pass with fullgraph=True and MaxCompiler
    compiled_forward = torch.compile(model.forward, fullgraph=True, backend=MaxCompiler)

    @torch.no_grad()
    def generate_with_compiled_step(idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop idx to last max_seq_len tokens
            idx_cond = (
                idx
                if idx.size(1) <= model.config.max_seq_len
                else idx[:, -model.config.max_seq_len :]
            )

            # Mark dynamic dimensions for compilation
            mark_dynamic(idx_cond, 0)  # Batch dimension
            mark_dynamic(idx_cond, 1)  # Sequence length dimension

            # Use compiled forward pass
            logits, _ = compiled_forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        generated = generate_with_compiled_step(
            tokens, max_new_tokens=30, temperature=0.8, top_k=50
        )
        generated_text = enc.decode(generated[0].tolist())
        print(f"Generated (compiled step): {generated_text}")

    print("\n" + "=" * 50)
    print("Testing completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
