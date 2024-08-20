import torch
from torch import nn
import torch.nn.functional as F
from transformer_encoder import TransformerEncoderQ
from transformer_decoder import FullSongTransformerDecoder
from transformer_helpers import (
  weights_init, PositionalEncoding, TokenEmbedding
)

class VQVAE(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    is_training=True, n_codes=2048,n_groups=64
  ):
    super(VQVAE, self).__init__()
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation  

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training
    self.n_codes=n_codes
    self.n_groups=n_groups

    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(dec_d_model, n_token)
    self.encoder = TransformerEncoderQ(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation,n_codes,n_groups
    )

    self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,dropout=dec_dropout, activation=dec_activation)

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
    

  def get_sampled_latent(self, inp, padding_mask=None):
    token_emb = self.token_emb(inp)
    enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    _, encoder_out = self.encoder(enc_inp, padding_mask=padding_mask)
    vqvae_latent = encoder_out['z']
    
    return vqvae_latent

  def forward(self, enc_inp, dec_inp, dec_inp_seq_pos, padding_mask=None):
    enc_bt_size, enc_n_seqs = enc_inp.size(1), enc_inp.size(2) 
    enc_token_emb = self.token_emb(enc_inp) 
    dec_token_emb = self.token_emb(dec_inp) 

    enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1)) 

    enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))  
    dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))  

    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1)) 

    mu, encoder_out = self.encoder(enc_inp, padding_mask=padding_mask) 
    vqvae_latent = encoder_out['z']
    vae_latent_reshaped = vqvae_latent.reshape(enc_bt_size, enc_n_seqs, -1) 
   
    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vqvae_latent.device) 
    for n in range(dec_inp.size(1)):
      for b, (st, ed) in enumerate(zip(dec_inp_seq_pos[n, :-1], dec_inp_seq_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

    dec_out = self.decoder(dec_inp, dec_seg_emb) 
    dec_logits = self.dec_out_proj(dec_out) 
    
    return mu, encoder_out,dec_logits


  def compute_loss(self, encoder_out, beta, dec_logits, dec_tgt):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction='mean'
    ).float()

    vqvae_loss=encoder_out['diff']
    total_loss=recons_loss+beta*encoder_out['diff']

    return {
      'beta': beta,
      'total_loss': total_loss,
      'vqvae_loss': vqvae_loss,
      'recons_loss': recons_loss
    }
