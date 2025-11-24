import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, drop_out):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size 
        self.heads = heads 
        self.head_dim = embed_size // heads  

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.attn_drop = nn.Dropout(drop_out)
        self.out_drop= nn.Dropout(drop_out)
    def forward(self, values, keys, query, mask = None):
        N = query.shape[0] 
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, -1, self.heads, self.head_dim)
        keys = keys.view(N, -1, self.heads, self.head_dim)
        queries = queries.view(N, -1, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, -1, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, drop_out, num_heads = 4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim*num_heads , v_dim)
        self.attn_dropout = Dropout(drop_out)
        self.proj_dropout = Dropout(drop_out)
    def forward(self, x1, x2, mask=None):

        batch_size, seq_len1, in_dim1 = x1.size()
        _, seq_len2, _ = x2.size()
        q1 = self.proj_q1(x1)
        q1 = q1.view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)#8 4 768 4097
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)#8 4 4097 768
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v2).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, self.v_dim*self.num_heads)#  64 4 49 768
        output = self.proj_o(output)
        output = self.proj_dropout(output)
        output = output.view(batch_size, -1, self.v_dim)
        
        return output


class EnhenceVisionFeature(nn.Module):
    def __init__(self, in_channel, embed_size, v_channel,drop_out):
        super(EnhenceVisionFeature, self).__init__()
        #=========Code organizing in progress===============


    def forward(self, x):
        #=========Code organizing in progress===============
 
        
def swish(x):
    return x * torch.sigmoid(x)  
class MultiFeatureFusion(nn.Module):
    def __init__(self,g_embed_size, g_in_channel,v_in_channel, vision_embed, heads, drop_out, 
                 fusion_feat_ch = 1024,fusion_embed = 16):
         super(MultiFeatureFusion, self).__init__()
         #=========Code organizing in progress===============
         
    def forward(self, gene_feature, vision_feature):
        #=========Code organizing in progress===============
        return ret
       
        