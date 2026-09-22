import torch
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset
from model import JumpRELUSAE , mse_loss 
import time
import os

#dodaj zapisywanie lossu w pętli 
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

# TODO dodaj funkcje get
def get_lr(step):
    # zrób cosine warmup 
    pass

def get_lam(step):
    # zrób warmup do lamdba 
    pass

class MakeBatch:
    def __init__(self,x) -> None:
        pass

h = gemma.model.layers[13].register_forward_hook(hook)

lr = 7e-5
optimizer = torch.optim.Adam(SAE_model.parameters(), lr=lr , betas= (0,0.999))

lam = 0.01

token_count = 0
total_tokens = 400_000_000
step = 0

gemma_emb = 2304
BATCH_SIZE = 4096


token_pool = torch.empty(0,gemma_emb , device=device)
TOKEN_POOL_SIZE = 100_000

for text in data:
    

    # dodaj akulumacje do batcha 4096 
    # Najpierw gemma tokenizuje
    # dodaje tokeny do puli która jest bardzo duża 
    # mieszam losowo  
    # daje modelowi do uczenia 
    # W praktyce przepisać pętle 
    token = tokenizer(text["text"] , return_tensors = "pt")
    x = token.input_ids.to(device)
    token_count += x.shape[-1]
    

    with torch.no_grad():
        try:
            gemma(x)
        except StopForwad:
            pass

    v = acts["vector"].float()
    assert torch.isfinite(v).all()

    token_pool = torch.cat([token_pool,v])

    if token_pool.size(0) >= TOKEN_POOL_SIZE:
        #shuffle 
        pass

        while token_pool.size(0) >= BATCH_SIZE:
            step +=1
            t_0 = time.time()
            
            

            f ,x_hat , active =  SAE_model(v)
            l0 = active.sum(-1).mean()

            # TODO dodaj funkcje get 
            loss = mse_loss(v , x_hat) + lam * l0


            loss.backward()
            optimizer.step()

            optimizer.zero_grad()       

                

            torch.cuda.synchronize()
            t_1 = time.time()

            tok_per_sec = BATCH_SIZE/(t_1-t_0)


            print(f"time:{t_1-t_0:.4f}| tok_per_sec:{tok_per_sec:.4f}| max_memory:{torch.cuda.max_memory_allocated()/1e9:.4f}")
            print(f"loss:{loss.item():.4f}| threshold mean:{SAE_model.threshold.mean().item():.6f}| l0:{l0.item():.4f}")

            if token_count >= total_tokens:
                    # TODO dodaj zakończenie pętli
                    break
            

        
    
    



