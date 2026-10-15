import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_basics import AttnNet, FC_block

class HERA(nn.Module):
    def __init__(
        self,
        input_feature_size=1024,
        precompression_layer=False,
        feature_size_comp = 512,
        feature_size_attn = 256,
        postcompression_layer=True,
        feature_size_comp_post = 128,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    ):
        super(HERA, self).__init__()

        if precompression_layer:
            self.compression_layer = nn.Sequential(*[FC_block(input_feature_size, feature_size_comp*4, p_dropout_fc=p_dropout_fc),
                                                    FC_block(feature_size_comp*4, feature_size_comp*2, p_dropout_fc=p_dropout_fc),
                                                    FC_block(feature_size_comp*2, feature_size_comp, p_dropout_fc=p_dropout_fc),])

            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = input_feature_size

        self.attention_net = AttnNet(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=1)

        self.post_compression_layer = nn.Sequential(*[FC_block(dim_post_compression, feature_size_comp_post, p_dropout_fc=p_dropout_fc)])

        # Classification head.
        self.classifiers = nn.Linear(feature_size_comp_post, 1)

        # Init weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_attention(self, h):
        A_ = self.attention_net(h)  # h shape is N_tilesxdim
        A_raw = torch.transpose(A_, 1, 0)  # K_attention_classesxN_tiles
        A = F.softmax(A_raw, dim=-1)  # #normalize attentions scores over tiles
        return A_raw, A

    def forward_classification(self, m):
        logits = self.classifiers(m)
        Y_prob = torch.sigmoid(logits)
        return logits, Y_prob

    def forward(self, h):

        #H&E embedding
        h = self.compression_layer(h)

        ### Attention MIL
        A_raw, A = self.forward_attention(h) # 1xN tiles

        ### First-order pooling
        m = A @ h #torch.Size([1, dim_embedding])  # 1x512 [Sum over N(aihi,1), ..., Sum over N(aihi,dim_embedding)]

        # post compression of he embedding
        m = self.post_compression_layer(m)

        logits, Y_prob = self.forward_classification(m)

        return logits, Y_prob, A_raw, m