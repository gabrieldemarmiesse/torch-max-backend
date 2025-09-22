import os
from pathlib import Path
from huggingface_hub import hf_hub_download

import torch
import torch.nn as nn
from torch_max_backend import max_backend
from torch._dynamo import mark_dynamic

os.environ["TORCH_MAX_BACKEND_PROFILE"] = "1"
os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"
os.environ["TORCH_MAX_BACKEND_DEBUG_GRAPH"] = "1"


def _create_masks(seq_len, device):
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

    # mask_global (future is masked: j > i)
    #     j:  0 1 2 3 4 5 6 7
    #  i
    #     0:  0 1 1 1 1 1 1 1
    #     1:  0 0 1 1 1 1 1 1
    #     2:  0 0 0 1 1 1 1 1
    #     3:  0 0 0 0 1 1 1 1
    #     4:  0 0 0 0 0 1 1 1
    #     5:  0 0 0 0 0 0 1 1
    #     6:  0 0 0 0 0 0 0 1
    #     7:  0 0 0 0 0 0 0 0
    mask_global = torch.triu(ones, diagonal=1)

    # far_past (too far back is masked: i - j >= sliding_window)
    # where sliding_window = 4
    #     j:  0 1 2 3 4 5 6 7
    #  i
    #     0:  0 0 0 0 0 0 0 0
    #     1:  0 0 0 0 0 0 0 0
    #     2:  0 0 0 0 0 0 0 0
    #     3:  0 0 0 0 0 0 0 0
    #     4:  1 0 0 0 0 0 0 0
    #     5:  1 1 0 0 0 0 0 0
    #     6:  1 1 1 0 0 0 0 0
    #     7:  1 1 1 1 0 0 0 0
    far_past = torch.triu(ones, diagonal=512).T

    # Local (sliding_window) = future OR far-past
    # mask_local
    #     j:  0 1 2 3 4 5 6 7
    # i
    # 0:      0 1 1 1 1 1 1 1
    # 1:      0 0 1 1 1 1 1 1
    # 2:      0 0 0 1 1 1 1 1
    # 3:      0 0 0 0 1 1 1 1
    # 4:      1 0 0 0 0 1 1 1
    # 5:      1 1 0 0 0 0 1 1
    # 6:      1 1 1 0 0 0 0 1
    # 7:      1 1 1 1 0 0 0 0
    mask_local = mask_global | far_past
    return mask_global, mask_local


class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, input_ids):
        # Forward pass
        _, seq_len = input_ids.shape
        _ = input_ids + 1
        _, mask_local = _create_masks(seq_len, torch.device("cuda"))
        return mask_local


GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 1,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": ["sliding_attention"],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


torch.manual_seed(123)
model = Gemma3Model(GEMMA3_CONFIG_270M)


model(torch.tensor([1, 2, 3]).unsqueeze(0))


total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")


device = torch.device("cuda")

model.to(device)


CHOOSE_MODEL = "270m"

repo_id = f"google/gemma-3-{CHOOSE_MODEL}-it"


local_dir = Path(repo_id).parts[-1]


from tokenizers import Tokenizer


class GemmaTokenizer:
    def __init__(self, tokenizer_file_path: str):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        # Attempt to identify EOS and padding tokens
        eos_token = "<end_of_turn>"
        self.pad_token_id = eos_token
        self.eos_token_id = eos_token

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
if not os.path.exists(tokenizer_file_path):
    try:
        tokenizer_file_path = hf_hub_download(
            repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir
        )
    except Exception as e:
        print(f"Warning: failed to download tokenizer.json: {e}")
        tokenizer_file_path = "tokenizer.json"

tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)


prompt = "Give me a short introduction to large language models."
prompt = apply_chat_template("Give me a short introduction to large language models.")


input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)


model = torch.compile(model, backend=max_backend)


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            mark_dynamic(token_ids, 1)
            out = model(token_ids)[:, -1]
            raise RuntimeError("Debugging")
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)


input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)


for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=500,
    eos_token_id=tokenizer.encode("<end_of_turn>")[-1],
):
    token_id = token.squeeze(0).tolist()
    print(tokenizer.decode(token_id), end="", flush=True)
