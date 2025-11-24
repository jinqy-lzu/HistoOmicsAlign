import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViT
from feature_fusion import MultiFeatureFusion, EnhenceVisionFeature
from gene_feature import StatisSpitialFusion




def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg =  l1_reg_all(model.enhence_module) + l1_reg_all(model.gene_embed) + l1_reg_all(model.fusion_module)

    return l1_reg

class HistopGeneFusion(nn.Module):
    def __init__(self, gene_nums, img_feat_ch, img_feat_emb, classes, mode):
        super().__init__()
        self.classes = classes
        self.vit_feature_channel = img_feat_ch
        self.vit_feature_embed = img_feat_emb
        self.vision_channel = 768
        self.gene_channel = 64
        self.gene_embed_dim = 512
        self.lstm_layer = 3
        self.gene_drop = 0.3 
        self.vision_drop = 0.3 
        self.fusion_drop = 0.2 
        self.fusion_feature_channel = 64 
        self.fusion_embed = 16 
        self.mode = mode
        self.gene_embed = StatisSpitialFusion(input_size=gene_nums, hidden_size=self.gene_channel,
                                              embed_size = self.gene_embed_dim, 
                                              drop_out=self.gene_drop,num_layers=self.lstm_layer)
        self.enhence_module = EnhenceVisionFeature(self.vit_feature_channel, self.vit_feature_embed,
                                                   self.vision_channel,drop_out=self.vision_drop)
        self.fusion_module = MultiFeatureFusion(self.gene_embed_dim, 
                                                self.gene_channel, 
                                                self.vision_channel,
                                                self.vit_feature_embed, 4,
                                                self.fusion_drop,
                                                fusion_feat_ch= self.fusion_feature_channel,
                                                fusion_embed = self.fusion_embed)
        self.classifier = nn.Linear(self.fusion_embed * self.fusion_feature_channel * 2, self.classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, gene, img):
        img = img.permute(0,2,1)
        vision_f = self.enhence_module(img)
        gene_f = self.gene_embed(gene)
        embed_f = self.fusion_module(gene_f, vision_f)
        embed_f = embed_f.flatten(start_dim = 1)
        predict = self.classifier(embed_f)
        if self.mode == 'survival':
            Y_hat = torch.topk(predict, 1, dim=1)[1]
            Y_prob = F.softmax(predict, dim=1)
            hazards = torch.sigmoid(predict)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, atten
        else:
            predict = self.softmax(predict)
            return predict
        
