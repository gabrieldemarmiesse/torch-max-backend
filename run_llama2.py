from fms.models import get_model
from fms.models.hf import to_hf_api
import torch
from transformers import pipeline, AutoTokenizer
from max_torch_backend import MaxCompiler

# fms model
llama = get_model("llama", "7b")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# huggingface model backed by fms internals
llama_hf = to_hf_api(llama)

print("compiling...")
# compile the model -- in HF, the decoder only
llama_hf.decoder = torch.compile(llama_hf.decoder, backend=MaxCompiler)
print("model compiled")
# generate some text -- the first time will be slow since the model needs to be compiled, but subsequent generations should be faster.
llama_generator = pipeline(task="text-generation", model=llama_hf, tokenizer=tokenizer)
print(
    llama_generator(
        """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""
    )
)
