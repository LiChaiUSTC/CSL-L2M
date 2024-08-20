import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformer_helpers import generate_causal_mask

class FullSongTransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu'):
    super(FullSongTransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dropout = dropout
    self.activation = activation
    self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x, seg_emb):
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)
    seg_emb = self.seg_emb_proj(seg_emb)
    out = x
    for i in range(self.n_layer):
      out += seg_emb
      out = self.decoder_layers[i](out, src_mask=attn_mask)
    return out



