import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset


data = load_dataset("HuggingFaceFW/fineweb-edu" ,streaming=True)

data = data.take(100_000) #~~100-200 mil token

device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    dtype=torch.float16,
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2")


class StopForwad(Exception): pass

acts = {}
def hook(module , inp , out):
    acts["vector"] = out[0]
    raise StopForwad


h = model.model.layers[13].register_forward_hook(hook)




