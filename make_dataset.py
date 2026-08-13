import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset
from model import JumpRELUSAE
import time
import tqdm

data = load_dataset("HuggingFaceFW/fineweb-edu" ,streaming=True , split="train")

data = data.take(10)

device = "cuda"
Scope_model = JumpRELUSAE()

gemma = AutoModelForCausalLM.from_pretrained(
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


h = gemma.model.layers[13].register_forward_hook(hook)
n_tok = 0

def train(model , x):
    pass

for text in data:

    t_0 = time.time()

    token = tokenizer(text["text"] , return_tensors = "pt" )
    x = token.inputs_id.to(device)

    with torch.inference_mode():
        try:
            gemma(x)
        except StopForwad:
            pass

    v = acts["vector"]

    #write train loop for Scope

    t_1 = time.time()

    

    
    
    



