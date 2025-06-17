#回归
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from d2l import torch as d2l
from torch.nn.utils import weight_norm
import math
import os
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScale
os.chdir('D:/itransformer/EMD')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model,dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        '''new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )'''
        x = x
        attn=0

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

seq_len,output_attention,enc_in,d_model,dropout,factor,n_heads,d_ff,activation,e_layers,num_class,device=63,None,256,512,0.1,5,256,1024,"gelu",4,70,d2l.try_gpu()
num_epochs,batch_size=1000,256


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self,seq_len,output_attention,enc_in,d_model,dropout,factor,n_heads,d_ff,activation,e_layers,num_calss):
        super(Model, self).__init__()
        self.seq_len=seq_len
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model,dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention),d_model,n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        #self.projection1=nn.Linear(d_model * seq_len, 3200)
        #self.projection2=nn.Linear(3200, 1600)
        self.projection = nn.Linear(d_model * seq_len, num_class)
        self.softmax=nn.Softmax(dim=1)
    def classification(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        #output = self.act(self.projection1(output))
        #output = self.act(self.projection2(output))# (batch_size, seq_length * d_model)
        output = self.projection(output)

        #output=self.softmax(output)# (batch_size, num_classes)
        return output

    def forward(self, x_enc,mask=None):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
    
net=Model(seq_len,output_attention,enc_in,d_model,dropout,factor,n_heads,d_ff,activation,e_layers,num_class)
net.to(device)

def evaluate_loss(net,data_iter,loss):
    total_loss = []
    preds = []
    trues = []
    net.eval()
    with torch.no_grad():
        for X,y in data_iter:
            X=X.to(device)
            y=y.to(device)
            y=y
            out=net(X)
            pred=out.detach().cpu()
            l=loss(pred,y.cpu())
            total_loss.append(l)
            preds.append(out.detach())
            trues.append(y)
    total_loss=np.average(total_loss)
    preds = torch.cat(preds, 0)
    trues = torch.cat(trues, 0)
    predictions = preds.cpu().numpy()  # (total_samples,) int class index for each sample
    print(predictions[0])
    trues = trues.cpu().numpy()
    print(trues[0])
    net.train()
    return total_loss

train_features=torch.transpose(torch.load('train_features.pth'),dim0=1,dim1=2)
train_labels=torch.load('train_labels.pth')
test_features=torch.transpose(torch.load('test_features.pth'),dim0=1,dim1=2)
test_labels=torch.load('test_labels.pth')
train_iter=d2l.load_array((train_features,train_labels),batch_size,is_train=True)
test_iter=d2l.load_array((test_features,test_labels),batch_size,is_train=True)
def train(net,train_iter,test_iter,num_epochs,resume=False):
    start_epoch=0
    Loss=nn.MSELoss()
    trainer=torch.optim.RAdam(net.parameters(),lr=0.001)
    if resume:
        checkpoint=torch.load('checkpoint')
        start_epoch=checkpoint['epoch']+1
        net.load_state_dict(checkpoint['model'])
        trainer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint (epoch{start_epoch-1})')
    else:
        print('=> no checkpoint')
    animator=d2l.Animator(xlabel='epoch',ylabel='loss',yscale='log',xlim=[1,num_epochs],ylim=[0,0.001],legend=['train','test'])
    loss_min=1
    train_save=[]
    test_save=[]
    for epoch in range(start_epoch,num_epochs):
        train_loss=[]
        scaler = GradScale()
        if isinstance(net, torch.nn.Module):
            net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(2)
        for X, y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y=y
            with autocast():
                y_hat = net(X)
                l = Loss(y_hat, y)
                #print(l)
            train_loss.append(l.item())
            trainer.zero_grad()
        # Compute gradients and update parameters
            scaler.scale(l).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(trainer)
            scaler.update()
        train_loss = np.average(train_loss)
        print(train_loss)
        if epoch==0 or (epoch+1)%1==0:
            train_loss=evaluate_loss(net,train_iter,Loss)
            test_loss=evaluate_loss(net,test_iter,Loss)
            train_save.append(train_loss)
            test_save.append(test_loss)
            print(train_loss)
            print(test_loss)
            loss_temp=test_loss
            animator.add(epoch+1,(train_loss,test_loss))
        if(loss_temp<loss_min):
            loss_min=loss_temp
            checkpoint={'epoch':epoch,'model':net.state_dict(),'optimizer':trainer.state_dict()}
            torch.save(checkpoint,'checkpoint')
    train_save=np.array(train_save)
    test_save=np.array(test_save)
    np.save("train_loss",train_save)
    np.save("test_loss",test_save)
train(net,train_iter,test_iter,num_epochs,resume=False)