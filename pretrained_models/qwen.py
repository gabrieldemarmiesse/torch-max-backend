import os
from torch_max_backend import max_backend, get_accelerators
from torch._dynamo import mark_dynamic

# TODO: cleanup

os.environ["TORCH_MAX_BACKEND_PROFILE"] = "1"
os.environ["TORCH_MAX_BACKEND_VERBOSE"] = "1"

model_name = "Qwen/Qwen3-0.6B"
USE_REASONING_MODEL = False  # The base model
USE_REASONING_MODEL = True  # The "thinking" model

device = "cuda" if len(list(get_accelerators())) >= 2 else "cpu"
# Use
# USE_REASONING_MODEL = True
# For Qwen3 Coder Flash model as well
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.0
TOP_K = 1
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"
else:
    filename = "qwen3-0.6B-base.pth"
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(repo_id=repo_id, filename=filename, local_dir=local_dir)

from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

print(f"Using device: {device}")
model.to(device)
model = torch.compile(model, backend=max_backend, fullgraph=True)
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"
else:
    tok_filename = "tokenizer-base.json"

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tok_filename,
    repo_id=repo_id,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL,
)
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)

# from llms_from_scratch.ch05 import generate
import time


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        mark_dynamic(idx_cond, 1)

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=80,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.0,
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0]) / total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024**3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
