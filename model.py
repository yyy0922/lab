"""Mamba-1 explicit selective recurrence; portable research implementation.

Not the official fused CUDA implementation. Discretization follows Mamba-1's
selective scan: exp(delta*A) state + delta*B*u, then C readout and D skip.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def mask_for(lengths, t):
    return torch.arange(t,device=lengths.device)[None,:] < lengths[:,None]

def reverse_valid(x,lengths):
    idx=(lengths[:,None]-1-torch.arange(x.shape[1],device=x.device)[None,:]).clamp_min(0)
    if x.ndim==3: idx=idx[:,:,None].expand_as(x)
    return x.gather(1,idx)

def affine_scan(a,b,initial):
    """Inclusive affine prefix scan; (a2,b2)o(a1,b1)=(a2*a1,b2+a2*b1)."""
    step=1
    while step<a.shape[1]:
        b=torch.cat([b[:,:step],b[:,step:]+a[:,step:]*b[:,:-step]],1)
        a=torch.cat([a[:,:step],a[:,step:]*a[:,:-step]],1)
        step*=2
    return b+a*initial[:,None]

class SelectiveBlock(nn.Module):
    def __init__(self,d,state,mode='none'):
        super().__init__()
        self.inner=2*d; self.state=state; self.rank=max(1,math.ceil(d/16)); self.mode=mode
        self.norm=nn.LayerNorm(d)
        self.inp=nn.Linear(d,4*d,bias=False)
        self.conv=nn.Conv1d(2*d,2*d,4,groups=2*d,padding=3)
        self.xproj=nn.Linear(2*d,self.rank+2*state,bias=False)
        self.dt=nn.Linear(self.rank,2*d)
        with torch.no_grad():
            delta=torch.exp(torch.empty(2*d).uniform_(math.log(.001),math.log(.1)))
            self.dt.bias.copy_(delta+torch.log(-torch.expm1(-delta)))
        self.Alog=nn.Parameter(torch.arange(1,state+1).float().log().repeat(2*d,1))
        self.skip=nn.Parameter(torch.ones(2*d))
        self.out=nn.Linear(2*d,d,bias=False)
        if mode!='none':
            self.group=nn.Embedding(5,8,padding_idx=0) if mode!='generic' else nn.Linear(2*d,8)
            self.offset=nn.Linear(8,2*d+2*state,bias=False)
            nn.init.zeros_(self.offset.weight)

    def forward(self,x,g,lengths):
        valid=mask_for(lengths,x.shape[1]); m=valid[:,:,None]
        u,z=self.inp(self.norm(x)).chunk(2,-1)
        u=F.silu(self.conv((u*m).transpose(1,2))[:,:,:x.shape[1]].transpose(1,2))*m
        dt,b,c=torch.split(self.xproj(u),[self.rank,self.state,self.state],-1)
        dt=self.dt(dt)
        if self.mode!='none':
            emb=self.group(u) if self.mode=='generic' else self.group(g)
            dd,db,dc=torch.split(.1*torch.tanh(self.offset(emb)),[self.inner,self.state,self.state],-1)
            if self.mode!='bc': dt=dt+dd
            if self.mode!='delta': b=b+db; c=c+dc
        dt=F.softplus(dt)
        A=-self.Alog.exp()
        h=x.new_zeros(x.shape[0],self.inner,self.state)
        outputs=[]
        for begin in range(0,x.shape[1],128):
            end=min(begin+128,x.shape[1]); step=dt[:,begin:end,:,None]
            keep=valid[:,begin:end,None,None]
            decay=torch.where(keep,torch.exp(step*A),1.)
            drive=torch.where(keep,step*b[:,begin:end,None,:]*u[:,begin:end,:,None],0.)
            states=checkpoint(affine_scan,decay,drive,h,use_reentrant=False) if self.training and torch.is_grad_enabled() else affine_scan(decay,drive,h)
            h=states[:,-1]
            outputs.append(((states*c[:,begin:end,None,:]).sum(-1)+self.skip*u[:,begin:end])*F.silu(z[:,begin:end]))
        return (x+self.out(torch.cat(outputs,1)))*m

class BiMamba(nn.Module):
    def __init__(self,d,state,mode):
        super().__init__(); self.block=SelectiveBlock(d,state,mode); self.merge=nn.Linear(2*d,d)
    def forward(self,x,g,lengths):
        a=self.block(x,g,lengths)
        b=reverse_valid(self.block(reverse_valid(x,lengths),reverse_valid(g,lengths),lengths),lengths)
        return self.merge(torch.cat([a,b],-1))*mask_for(lengths,x.shape[1])[:,:,None]

class TCN(nn.Module):
    def __init__(self,d,dilations=(1,2,4)):
        super().__init__()
        self.blocks=nn.ModuleList([nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Conv1d(d,d,3,dilation=k)),nn.utils.parametrizations.weight_norm(nn.Conv1d(d,d,3,dilation=k))]) for k in dilations])
        self.dilations=dilations
    def forward(self,x,mask):
        x=x.transpose(1,2); m=mask[:,None,:]
        for layers,k in zip(self.blocks,self.dilations):
            residual=x
            for conv in layers:
                x=F.dropout(F.relu(conv(F.pad(x,(2*k,0)))),.1,self.training)*m
            x=(x+residual)*m
        return x.transpose(1,2)

class Detector(nn.Module):
    def __init__(self,vocab_size,variant='C6',width=64,state=16,tau=1,rho=.2):
        super().__init__(); self.variant=variant; self.width=width; self.tau=max(1,tau); self.rho=rho
        self.embedding=nn.Embedding(vocab_size+1,width,padding_idx=0)
        self.input_bias=nn.Parameter(torch.zeros(width))
        self.concat=variant=='B'
        if self.concat: self.group_embed=nn.Embedding(5,8,padding_idx=0); self.group_project=nn.Linear(width+8,width)
        mode='all' if variant in ['C3','C4','C5','C6','D','F0','F1','F2'] else 'none'
        if variant=='C': mode='generic'
        if variant=='D_delta': mode='delta'
        if variant=='D_bc': mode='bc'
        if variant=='C1': self.lstm=nn.LSTM(width,width,batch_first=True)
        else: self.long=BiMamba(width,state,mode)
        self.fusion=variant in ['C4','C6','F0','F1','F2']
        if self.fusion: self.short=TCN(width,(1,)); self.short_project=nn.Linear(width,width)
        if variant=='F1': self.weight=nn.Parameter(torch.tensor(0.))
        self.local=TCN(width)
        self.attn_in=nn.Linear(2*width,width)
        self.qkv=nn.Linear(width,3*width); self.attn_out=nn.Linear(width,width)
        self.head=nn.Sequential(nn.Linear(3*width,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,2))

    def methods(self,ids,groups,lengths):
        mask=mask_for(lengths,ids.shape[1]); m=mask[:,:,None]
        x=(self.embedding(ids)+self.input_bias)*m
        if self.concat: x=self.group_project(torch.cat([x,self.group_embed(groups)],-1))*m
        if self.variant=='C1': route=self.lstm(x)[0]*m
        else: route=self.long(x,groups,lengths)
        if self.fusion:
            alpha=torch.sigmoid((self.tau-lengths.float())/max(1,self.tau/4))[:,None,None]
            if self.variant=='F0': alpha=.5
            if self.variant=='F1': alpha=self.weight.sigmoid()
            route=alpha*self.short_project(self.short(x,mask))+(1-alpha)*route
            route=route*m
        local=self.local(x,mask)
        att=self.attn_in(torch.cat([route,local],-1))*m
        n,t,d=att.shape
        q,k,v=[z.reshape(n,t,2,d//2).transpose(1,2) for z in self.qkv(att).chunk(3,-1)]
        ks=(lengths.float()*self.rho).ceil().long().clamp_min(1)
        sparse=self.variant in ['C5','C6']
        maxk=int(ks.max())
        def attend(query,key,value):
            scores=(query@key.transpose(-2,-1))/math.sqrt(d//2)
            scores=scores.masked_fill(~mask[:,None,None,:],float('-inf'))
            if sparse:
                top=scores.topk(maxk,dim=-1).indices
                allowed=(torch.arange(maxk,device=ids.device)[None,None,None,:]<ks[:,None,None,None]).expand_as(top)
                keep=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,top,allowed)
                scores=scores.masked_fill(~keep,float('-inf'))
            return scores.softmax(-1)@value
        # Exact query blocking + checkpoint recomputation bounds peak memory.
        # Every query still attends to all valid keys; no window or truncation.
        block=max(1,min(t,2_000_000//max(1,n*2*t)))
        pieces=[]
        for begin in range(0,t,block):
            query=q[:,:,begin:begin+block]
            pieces.append(checkpoint(attend,query,k,v,use_reentrant=False) if self.training and torch.is_grad_enabled() else attend(query,k,v))
        attended=torch.cat(pieces,2).transpose(1,2).reshape(n,t,d)
        attended=self.attn_out(attended)*m
        joined=torch.cat([route,local,attended],-1)*m
        return joined.sum(1)/lengths[:,None]

class OriginalLTA(nn.Module):
    """T=1 LTA architecture port; separate from method-level detector."""
    def __init__(self,features):
        super().__init__(); self.lstm=nn.LSTM(features,64,batch_first=True); self.tcn_in=nn.Linear(features,64); self.tcn=TCN(64)
        self.attn=nn.MultiheadAttention(64,2,batch_first=True,kdim=64,vdim=64)
        self.head=nn.Sequential(nn.Dropout(.5),nn.Linear(192,1))
    def forward(self,x):
        x=x[:,None,:]; a=self.lstm(x)[0]; b=self.tcn(self.tcn_in(x),torch.ones(x.shape[:2],device=x.device,dtype=torch.bool)); c=self.attn(a,a,a,need_weights=False)[0]
        logit=self.head(torch.cat([a,b,c],-1).flatten(1)).squeeze(-1)
        return torch.stack([torch.zeros_like(logit),logit],-1)
