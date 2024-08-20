import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizeEMA(nn.Module):
    def __init__(self, d_latent, n_codes=2048, n_groups=64, decay=0.995, eps=1e-4, restart_threshold=0.99,training=True):
        assert d_latent // n_groups == d_latent / n_groups, f"Unexpected latent dimension: d_latent={d_latent} must be divisible by n_groups={n_groups}"

        super().__init__()

        self.d_latent = d_latent
        self.n_groups = n_groups
        self.dim = d_latent // n_groups
        self.n_codes = n_codes 

        self.decay = decay
        self.eps = eps
        self.threshold = restart_threshold
        self.training=training
        self.init = False

        embed = torch.randn(self.n_codes, self.dim) 
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.ones(self.n_codes))
        self.register_buffer('cluster_sum', embed.clone().detach())

    def forward(self, x, dist=None):
        assert x.shape[-1] == self.n_groups * self.dim, f"Unexpected input shape: expected last dimension to be {self.n_groups * self.dim} but was {x.shape[-1]}"
        x_ = x.reshape(-1, self.dim) 

        if self.training and not self.init:
            self._init_embeddings(x_, dist=dist)

        ### Shared embeddings between groups ###
        # Find nearest neighbors in latent space
        emb_t = self.embedding.t()
        
        distance = (
            x_.pow(2).sum(1, keepdim=True)
            - 2 * x_ @ emb_t
            + emb_t.pow(2).sum(0, keepdim=True)
        ) 
     
        _, embed_idx = (-distance).max(1) 

        embed_onehot = F.one_hot(embed_idx, self.n_codes).type(x_.dtype) 

        quantize = self.embed(embed_idx).view(-1, self.n_groups * self.dim) 
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        codes = embed_idx.view(-1, self.n_groups) 

        if self.training:
            update_metrics = self._ema_update(x_, embed_onehot, dist=dist)
        else:
            update_metrics = {}

        return dict(
            z=quantize,
            diff=diff,
            codes=codes,
            **update_metrics
        )

    def embed(self, idx):
        return F.embedding(idx, self.embedding)

    def _init_embeddings(self, x, dist=None):
        self.init = True
        rand_centers = self._randomize(x)
        self.cluster_sum.data.copy_(rand_centers)
        self.cluster_size.data.fill_(1)
        

    def _randomize(self, x):
        n = x.size(0)
        if n < self.n_codes:
            r = (self.n_codes + n - 1) // n 
            std = 0.01 / np.sqrt(self.dim)
            x = x.repeat(r, 1)
            x += std * torch.randn_like(x)
        return x[torch.randperm(x.size(0))][:self.n_codes]

    def _ema_update(self, x, cluster_assign, dist=None):
        with torch.no_grad():
            cluster_size = cluster_assign.sum(0)
            cluster_sum = cluster_assign.t() @ x

            rand_centers = self._randomize(x)

            self.cluster_size.data.copy_(self.decay*self.cluster_size + (1 - self.decay)*cluster_size)
            self.cluster_sum.data.copy_(self.decay*self.cluster_sum + (1 - self.decay)*cluster_sum)

            used = (self.cluster_size >= self.threshold).float().unsqueeze(-1)

            n = self.cluster_size.sum()
            count = (self.cluster_size + self.eps) / (n + self.n_codes*self.eps) * n

            cluster_centers = self.cluster_sum / count.unsqueeze(-1)
            cluster_centers = used * cluster_centers + (1 - used) * rand_centers
            self.embedding.data.copy_(cluster_centers)

            # Compute metrics
            avg_usage = used.mean()
            usage = used.sum()
            pr = cluster_size / cluster_size.sum()
            entropy = -(pr * (pr + 1e-5).log()).sum()

        return {
            'avg_usage': avg_usage,
            'usage': usage,
            'entropy': entropy
        }


class TransformerEncoderQ(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu',n_codes=2048,n_groups=64):
    super(TransformerEncoderQ, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation
    self.n_codes=n_codes
    self.n_groups=n_groups
    self.vq_embed = VectorQuantizeEMA(self.d_vae_latent, self.n_codes, self.n_groups)
    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )
    self.fc_mu = nn.Linear(d_model, d_vae_latent)

  def forward(self, x,padding_mask=None):
    out = self.tr_encoder(x,src_key_padding_mask=padding_mask)
    hidden_out = out[0, :, :] 
    latent= self.fc_mu(hidden_out)
    return latent,self.vq_embed(latent)



class TransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
    super(TransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None):
    out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
    hidden_out = out[0, :, :] 
    return self.fc_mu(hidden_out)



