import torch
from torch import nn 

class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre , threshold , eps):
        ctx.save_for_backward(pre , threshold)
        ctx.eps = eps
        return pre * (pre> threshold)


    @staticmethod
    def backward(ctx, grad_out):
        pre , threshold = ctx.saved_tensors
        eps = ctx.eps

        mask = (pre> threshold)
        grad_pre = mask * grad_out

        near = ((pre -threshold).abs() < eps).float() 
        near = near/ (2 * eps)
        grad_thr = -threshold * near * grad_out

        grad_thr = grad_thr.reshape(-1 , grad_thr.shape[-1]).sum(0)

        return grad_pre , grad_thr , None
    
class JumpRELUSAE(nn.Module):

    def __init__(self , d_model = 2304 , d_sea = 16384):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model , d_sea))
        self.B_enc = nn.Parameter(torch.zeros(d_sea))
        self.W_dec = nn.Parameter(torch.zeros(d_sea , d_model))
        self.B_dec = nn.Parameter(torch.zeros(d_model))
        self.threshold = nn.Parameter(torch.full((d_sea,), 0.001))
        self.eps = 0.001

        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=-1 , keepdim = True)
            self.W_enc.copy_(self.W_dec.T)

    def encode_pre(self,x):
        return  x  @ self.W_enc + self.B_enc

    def encode(self , pre):
        return JumpReLU.apply(pre, self.threshold,self.eps)


    def decode(self , f):
        return f @ self.W_dec + self.B_dec

    def forward(self, x ):
        pre = self.encode_pre(x)
        active =  self.active_feature(pre)
        f = self.encode(pre)
        x_hat = self.decode(f)
        return f , x_hat , active

    def active_feature(self , pre):
        return ActiveFeatures.apply(pre , self.threshold, self.eps)



def mse_loss(x , x_hat):
    return ((x - x_hat)**2).sum(-1).mean()


class ActiveFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx , pre , threshold, eps):
        ctx.save_for_backward(pre , threshold)
        ctx.eps = eps
        return (pre > threshold).float()

    @staticmethod
    def backward(ctx , grad_out):
        pre ,threshold = ctx.saved_tensors
        eps = ctx.eps

        mask = (pre> threshold)
        grad_pre = mask * grad_out

        near = ((pre -threshold).abs() < eps).float()
        near = near/ (2 * eps)
        grad_thr = -near * grad_out
        grad_thr = grad_thr.reshape(-1 , grad_thr.shape[-1]).sum(0)

        return grad_pre , grad_thr , None