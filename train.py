import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset
from model import JumpRELUSAE , mse_loss 
import time
import tqdm

data = load_dataset("HuggingFaceFW/fineweb-edu" ,streaming=True , split="train")

data = data.take(1)

device = "cuda"
SAE_model = JumpRELUSAE().to(device)


gemma = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    dtype=torch.float16,
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")


class StopForwad(Exception): pass

acts = {}
def hook(module , inp , out):
    acts["vector"] = out[0]
    raise StopForwad


h = gemma.model.layers[13].register_forward_hook(hook)

lr = 1e-3
optimizer = torch.optim.Adam(SAE_model.parameters(), lr=lr )

lam = 0.01


for text in data:

    t_0 = time.time()

    token = tokenizer(text["text"] , return_tensors = "pt" , truncation=True, max_length=100 )
    x = token.input_ids.to(device)
    seq_len = x.shape[-1]

    with torch.no_grad():
        try:
            gemma(x)
        except StopForwad:
            pass

    v = acts["vector"].float()
    assert torch.isfinite(v).all()
    
    for i in range(500):

        f ,x_hat , active =  SAE_model(v)
        l0 = active.sum(-1).mean()
        loss = mse_loss(v , x_hat) + lam * l0

        print(f"loss: {loss}")
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()       

        

    torch.cuda.synchronize()
    t_1 = time.time()

    tok_per_sec = seq_len/(t_1-t_0)


    print(f"time:{t_1-t_0:.4f}| sequence len:{seq_len}| tok_per_sec:{tok_per_sec:.4f}| max_memory:{torch.cuda.max_memory_allocated()/1e9:.4f}")
    print(f"loss:{loss.item():.4f}| threshold mean:{SAE_model.threshold.mean().item():.6f}| l0:{l0.item():.4f}")

    

    
    
    



