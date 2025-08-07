from transformers import CompileConfig, LlamaForCausalLM, LlamaTokenizer
from max_torch_backend import MaxCompiler

model = LlamaForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-step-50K-105b").cuda()
tokenizer = LlamaTokenizer.from_pretrained("PY007/TinyLlama-1.1B-step-50K-105b")

# Automatic compile configuration, used with static cache
compile_config = CompileConfig(backend=MaxCompiler)

# Generation with static cache and compile config
input = tokenizer.encode("Hello there, how", return_tensors="pt").cuda()
output = model.generate(
    input,
    do_sample=False,
    max_new_tokens=300,
    cache_implementation="static",
    compile_config=compile_config,
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
