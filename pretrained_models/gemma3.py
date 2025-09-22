import os
from pathlib import Path
from huggingface_hub import hf_hub_download

import torch
from torch_max_backend import max_backend
from torch._dynamo import mark_dynamic

os.environ["TORCH_MAX_BACKEND_PROFILE"] = "1"
os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"
os.environ["TORCH_MAX_BACKEND_DEBUG_GRAPH"] = "1"


def _create_masks(seq_len, device):
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    mask_global = torch.triu(ones, diagonal=1)

    far_past = torch.triu(ones, diagonal=512).T
    mask_local = mask_global | far_past
    return mask_global, mask_local


def forward(input_ids):
    # Forward pass
    _, seq_len = input_ids.shape
    _ = input_ids + 1
    _, mask_local = _create_masks(seq_len, torch.device("cuda"))
    return mask_local


torch.manual_seed(123)
model = forward


model(torch.tensor([1, 2, 3]).unsqueeze(0))


device = torch.device("cuda")


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


def generate_text_basic_stream(model, token_ids, max_new_tokens):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            mark_dynamic(token_ids, 1)
            _ = model(token_ids)[:, -1]
            raise RuntimeError("Debugging")


input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)


for token in generate_text_basic_stream(
    model=model, token_ids=input_token_ids_tensor, max_new_tokens=500
):
    token_id = token.squeeze(0).tolist()
    print(tokenizer.decode(token_id), end="", flush=True)
