import torch
import torch.nn as nn
import numpy as np

class AttnNet(nn.Module):
    # Adapted from https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
    # Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()

        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(p_dropout_atn))
            self.attention_b.append(nn.Dropout(p_dropout_atn))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

class FC_block(nn.Module):
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x