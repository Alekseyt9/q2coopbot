"""Two observed-only attention hypotheses with zero residual initialization."""
import copy,math
from ppo_combat import torch,nn,network,layers

TEMPORAL='combat_causal_attention_v1'
ENTITY='combat_entity_attention_v1'

def attention_cell(width,outputs,heads):
    mha=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0.,device='cuda')
    result={}
    for i,key in enumerate(('query','key','value')):
        result[key]={'weight':mha.in_proj_weight[i*width:(i+1)*width].detach().cpu().tolist(),'bias':mha.in_proj_bias[i*width:(i+1)*width].detach().cpu().tolist()}
    result['output']={'weight':mha.out_proj.weight.detach().cpu().tolist(),'bias':mha.out_proj.bias.detach().cpu().tolist()}
    result['residual']={'weight':[[0.]*width for _ in range(outputs)],'bias':[0.]*outputs}
    return result

def initialize_attention(model,entity=False,heads=4,window=32):
    assert not any(model.get(k) for k in ('memory','attention','entity_attention'))
    assert model['feature_version']=='combat_features_v4'
    spec={'version':ENTITY if entity else TEMPORAL,'heads':heads}
    if not entity:spec['window']=window
    for name in ('actor','value'):
        width=len(model[name][1]['bias']);cell=attention_cell(width,len(model[name][-1]['bias']),heads)
        if entity:
            token=nn.Linear(68,width,device='cuda')
            cell={'attention':cell,'token':{'weight':token.weight.detach().cpu().tolist(),'bias':token.bias.detach().cpu().tolist()}}
        spec[name]=cell
    return {**copy.deepcopy(model),'entity_attention' if entity else 'attention':spec}

def set_linear(module,spec):
    with torch.no_grad():module.weight.copy_(torch.tensor(spec['weight']));module.bias.copy_(torch.tensor(spec['bias']))

class AttentionBase(nn.Module):
    def __init__(self,base,cell,heads):
        super().__init__();net=network(base);self.encoder=net[:4];self.head=net[4]
        width=len(base[1]['bias']);self.mha=nn.MultiheadAttention(width,heads,batch_first=True,dropout=0.)
        self.residual=nn.Linear(width,len(base[-1]['bias']))
        with torch.no_grad():
            self.mha.in_proj_weight.copy_(torch.cat([torch.tensor(cell[k]['weight']) for k in ('query','key','value')]))
            self.mha.in_proj_bias.copy_(torch.cat([torch.tensor(cell[k]['bias']) for k in ('query','key','value')]))
        set_linear(self.mha.out_proj,cell['output']);set_linear(self.residual,cell['residual'])
    def export(self):
        width=self.mha.embed_dim;cell={}
        for i,key in enumerate(('query','key','value')):
            cell[key]={'weight':self.mha.in_proj_weight[i*width:(i+1)*width].detach().cpu().tolist(),'bias':self.mha.in_proj_bias[i*width:(i+1)*width].detach().cpu().tolist()}
        for key,module in [('output',self.mha.out_proj),('residual',self.residual)]:
            cell[key]={'weight':module.weight.detach().cpu().tolist(),'bias':module.bias.detach().cpu().tolist()}
        return layers(nn.Sequential(*self.encoder,self.head)),cell

class CausalAttention(AttentionBase):
    def __init__(self,base,cell,heads=4,window=32):super().__init__(base,cell,heads);self.window=window
    def forward(self,x):
        embedded=self.encoder(x);length,width=embedded.shape[1:]
        positions=torch.arange(length,device=x.device,dtype=x.dtype)[:,None]
        freq=torch.exp(-math.log(10000)*torch.arange(0,width,2,device=x.device,dtype=x.dtype)/width)
        pe=torch.zeros(length,width,device=x.device,dtype=x.dtype);pe[:,0::2]=torch.sin(positions*freq);pe[:,1::2]=torch.cos(positions*freq)
        tokens=embedded+pe[None,:,:]
        q=torch.arange(length,device=x.device)[:,None];k=torch.arange(length,device=x.device)[None,:]
        mask=(k>q)|(k<q-self.window+1)
        result,_=self.mha(tokens,tokens,tokens,attn_mask=mask,need_weights=False)
        return self.head(embedded)+self.residual(result),tokens
    def single(self,x):return self(x[:,None,:])[0][:,0,:]

def entity_tokens(x):
    assert x.shape[-1]==810
    tokens=[];valid=[]
    def put(role,data,mask):
        t=x.new_zeros((*x.shape[:-1],68));t[...,:data.shape[-1]]=data;t[...,62+role]=1
        tokens.append(t);valid.append(mask)
    put(0,torch.cat((x[...,:25],x[...,786:810]),-1),torch.ones_like(x[...,0],dtype=torch.bool))
    for i in range(8):
        j=73+12*i;data=torch.cat((x[...,j:j+12],x[...,386+5*i:391+5*i],x[...,426+5*i:431+5*i],x[...,466+40*i:506+40*i]),-1)
        put(1,data,x[...,j]!=0)
    for i in range(4):j=170+12*i;put(2,x[...,j:j+12],x[...,j]!=0)
    for i in range(8):j=25+6*i;put(3,x[...,j:j+6],x[...,j]!=0)
    for role,start in enumerate((223,260)):
        for i in range(4):j=start+9*i;put(4+role,x[...,j:j+9],x[...,j]!=0)
    return torch.stack(tokens,-2),torch.stack(valid,-1)

class EntityAttention(AttentionBase):
    def __init__(self,base,cell,heads=4):
        super().__init__(base,cell['attention'],heads);self.token=nn.Linear(68,self.mha.embed_dim);set_linear(self.token,cell['token'])
    def single(self,x):
        encoded=self.encoder(x);tokens,valid=entity_tokens(x);tokens=self.token(tokens).relu()
        attended,_=self.mha(encoded[:,None,:],tokens,tokens,key_padding_mask=~valid,need_weights=False)
        return self.head(encoded)+self.residual(attended[:,0,:])
    def export(self):
        base,cell=super().export();token={'weight':self.token.weight.detach().cpu().tolist(),'bias':self.token.bias.detach().cpu().tolist()}
        return base,{'attention':cell,'token':token}
