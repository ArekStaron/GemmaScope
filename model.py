import torch
from torch import nn 

class JumpRELUSAE(nn.Module):

    def __init__(self , d_model = 2304 , d_sea = 16384):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model , d_sea))
        self.B_enc = nn.Parameter(torch.zeros(d_sea))
        self.W_dec = nn.Parameter(torch.zeros(d_sea , d_model))
        self.B_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.full((d_sea,), 0.001))


        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=-1 , keepdim = True)
            self.W_enc.copy_(self.W_dec.T)


    def encode(self , x):
        pre = x  @ self.W_enc + self.B_enc
        return pre * ( pre > self.threshold)


    def decode(self , f):
        return f @ self.W_dec + self.B_dec

    def forward(self, x ):
        f = self.encode(x)
        x_hat = self.decode(f)
        return f , x_hat




    