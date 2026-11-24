import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset
from model import JumpRELUSAE , mse_loss 
import time
import os
import math

#TODO zapisywanie lossu w pętli 
os.makedirs("loss" , exist_ok=True)
device = "cuda"

data = load_dataset("HuggingFaceFW/fineweb-edu" ,streaming=True , split="train")


gemma = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    dtype=torch.float16,
    device_map=device
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")


SAE_model = JumpRELUSAE().to(device)

class StopForwad(Exception): pass

acts = {}
def hook(module , inp , out):
    acts["vector"] = out[0]
    raise StopForwad



lr = 7e-5
start_lr = 0.1 * lr
WARM_UP = 1000

def get_lr(step):
    if step >= WARM_UP:
        return lr
    progres = step/WARM_UP
    factor = 0.5 * (1-math.cos(math.pi * progres))
    
    return start_lr + (lr-start_lr) * factor
    
lam = 0.01
LAM_WARM_UP = 10_000

def get_lam(step):
    return lam * min(1.0 ,step/LAM_WARM_UP)

def shuffle(t):
    return t[torch.randperm(t.size(0) , device = t.device)]


h = gemma.model.layers[13].register_forward_hook(hook)


optimizer = torch.optim.Adam(SAE_model.parameters(), betas= (0,0.999))


token_count = 0
total_tokens = 400_000_000
step = 0

gemma_emb = 2304
BATCH_SIZE = 4096


token_pool = torch.empty(0,gemma_emb , device=device)
TOKEN_POOL_SIZE = 100_000

for text in data:
    t_0 = time.time()

     
    token = tokenizer(text["text"] , return_tensors = "pt")
    token_ids = token.input_ids.to(device)
    token_count += token_ids.shape[-1]
    

    with torch.no_grad():
        try:
            gemma(token_ids)
        except StopForwad:
            pass

    v = acts["vector"].float()
    assert torch.isfinite(v).all()
     
    v = v[1:]
    scale_v = math.sqrt(gemma_emb)
    v = v/scale_v

    token_pool = torch.cat([token_pool,v])



    if token_pool.size(0) >= TOKEN_POOL_SIZE:
        n_tokens = token_pool.size(0)
        print(f"tokens in pool size: {n_tokens}")
        token_pool = shuffle(token_pool)

        while n_tokens >= BATCH_SIZE:
            step +=1
            
            
            x = token_pool[:BATCH_SIZE]
            token_pool = token_pool[BATCH_SIZE:]

            f ,x_hat , active =  SAE_model(x)
            l0 = active.sum(-1).mean()

            
            loss = mse_loss(x , x_hat) + get_lam(step) * l0


            for p in optimizer.param_groups:
                p["lr"] = get_lr(step)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()       

                

            
            
            # TODO dodaj wyświetlanie parametrów co batch 
    t_1 = time.time()

    tok_per_sec = n_tokens/(t_1-t_0)


    print(f"time:{t_1-t_0:.4f}| tok_per_sec:{tok_per_sec:.4f}| max_memory:{torch.cuda.max_memory_allocated()/1e9:.4f}")
    print(f"loss:{loss.item():.4f}| threshold mean:{SAE_model.threshold.mean().item():.6f}| l0:{l0.item():.4f}")

    if token_count >= total_tokens:
                # TODO dodaj zakończenie pętli
                break
            

        
    
    



