import torch
from torch import nn
import torch.nn.functional as F
from transformer_encoder import TransformerEncoder
from transformer_decoder import FullSongTransformerDecoder
from transformer_helpers import (
  weights_init, PositionalEncoding, TokenEmbedding
)

class CSLL2M(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,n_token_lyric,pad_token_melody,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_pos_emb=128, d_tone_emb=128,
    d_struct_emb=32, d_key_emb=32, d_emotion_emb=32, 
    d_PM_emb=32, d_PV_emb=32, d_PR_emb=32, d_DMM_emb=32, d_AA_emb=32, d_CM_emb=32, 
    d_DM_emb=32, d_DV_emb=32, d_DR_emb=32, d_MCD_emb=32,
    d_ND_emb=32, d_Align_emb=32,
    d_learned_features=128,
    n_pos_cls=55, n_tone_cls=5,
    n_struct_cls=5, n_key_cls=24, n_emotion_cls=3,  #human-labeled musical tags
    n_PM_cls=64, n_PV_cls=64, n_PR_cls=64, n_DMM_cls=64, n_AA_cls=64, n_CM_cls=64, #pitch-related statistical musical attributes: pitch mean (PM), pitch variance (PV), pitch range (PR), direction of melodic motion (DMM), amount of arpeggiation (AA), chromatic motion (CM)
    n_DM_cls=64, n_DV_cls=64, n_DR_cls=64, n_MCD_cls=64,  #duration-related statistical musical attributes: duration mean (DM), duration variance (DV), duration range (DR), prevalence of most common note duration (MCD)
    n_ND_cls=64, n_Align_cls=64,   #rhythm-related and note-number-related statistical musical attributes: note density (ND), fraction of syllables to notes (Align) 
    f_pos=False, f_tone=False,
    f_struct=False, f_key=False, f_emotion=False,
    f_PM=False, f_PV=False, f_PR=False, f_DMM=False, f_AA=False, f_CM=False,
    f_DM=False, f_DV=False, f_DR=False, f_MCD=False,
    f_ND=False, f_Align=False,
    f_leared_features=False,is_training=True,
    use_musc_ctls =False):

    super(CSLL2M, self).__init__()
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
    self.n_token_lyric=n_token_lyric
    self.pad_token_melody=pad_token_melody
    self.is_training = is_training

    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.token_emb_lyric = TokenEmbedding(n_token_lyric, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(dec_d_model, n_token)
    self.encoder_seqLyric = TransformerEncoder(enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation)

    self.f_pos=f_pos
    self.f_tone=f_tone
    self.f_struct=f_struct
    self.f_key=f_key
    self.f_emotion=f_emotion
    self.f_PM=f_PM
    self.f_PV=f_PV
    self.f_PR = f_PR
    self.f_DMM=f_DMM
    self.f_AA=f_AA
    self.f_CM = f_CM
    self.f_DM=f_DM
    self.f_DV=f_DV 
    self.f_DR=f_DR
    self.f_MCD=f_MCD
    self.f_ND = f_ND
    self.f_Align=f_Align
    self.f_leared_features=f_leared_features
    self.use_musc_ctls=use_musc_ctls

    if use_musc_ctls:
      if f_key and f_emotion and f_struct:
        self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb,dropout=dec_dropout, activation=dec_activation)
      if f_key and f_emotion and f_struct and f_Align:
        self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb+d_Align_emb,dropout=dec_dropout, activation=dec_activation)
      if f_key and f_emotion and f_struct and f_Align and f_ND:
        self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb+d_Align_emb+d_ND_emb,dropout=dec_dropout, activation=dec_activation)
      if f_key and f_emotion and f_struct and f_Align and f_ND and f_PM and f_PV and f_PR and f_DMM and f_AA and f_CM:
        self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb+d_Align_emb+d_ND_emb+d_PM_emb+d_PV_emb+d_PR_emb+d_DMM_emb+d_AA_emb+d_CM_emb,dropout=dec_dropout, activation=dec_activation)
      if f_key and f_emotion and f_struct and f_Align and f_ND and f_PM and f_PV and f_PR and f_DMM and f_AA and f_CM and f_DM and f_DV and f_DR and f_MCD:
        self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb+d_Align_emb+d_ND_emb+d_PM_emb+d_PV_emb+d_PR_emb+d_DMM_emb+d_AA_emb+d_CM_emb+d_DM_emb+d_DV_emb+d_DR_emb+d_MCD_emb,dropout=dec_dropout, activation=dec_activation)
      if f_key and f_emotion and f_struct and f_Align and f_ND and f_PM and f_PV and f_PR and f_DMM and f_AA and f_CM and f_DM and f_DV and f_DR and f_MCD and f_leared_features:
         self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_key_emb+d_emotion_emb+d_struct_emb+d_Align_emb+d_ND_emb+d_PM_emb+d_PV_emb+d_PR_emb+d_DMM_emb+d_AA_emb+d_CM_emb+d_DM_emb+d_DV_emb+d_DR_emb+d_MCD_emb+d_learned_features,dropout=dec_dropout, activation=dec_activation)
    else:
      self.decoder = FullSongTransformerDecoder(dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,dropout=dec_dropout, activation=dec_activation)


    if use_musc_ctls:
      if f_key and f_emotion and f_struct:
        self.key_emb=TokenEmbedding(n_key_cls, d_key_emb, d_key_emb)
        self.emotion_emb=TokenEmbedding(n_emotion_cls, d_emotion_emb, d_emotion_emb)
        self.struct_emb=TokenEmbedding(n_struct_cls, d_struct_emb, d_struct_emb)
      if f_key and f_emotion and f_struct and f_Align:
        self.Align_emb = TokenEmbedding(n_Align_cls, d_Align_emb, d_Align_emb)
      if f_key and f_emotion and f_struct and f_Align and f_ND:
        self.ND_emb = TokenEmbedding(n_ND_cls, d_ND_emb, d_ND_emb)
      if f_key and f_emotion and f_struct and f_Align and f_ND and f_PM and f_PV and f_PR and f_DMM and f_AA and f_CM:
        self.PM_emb = TokenEmbedding(n_PM_cls, d_PM_emb, d_PM_emb)
        self.PV_emb=TokenEmbedding(n_PV_cls, d_PV_emb, d_PV_emb)
        self.PR_emb=TokenEmbedding(n_PR_cls, d_PR_emb, d_PR_emb)
        self.DMM_emb=TokenEmbedding(n_DMM_cls, d_DMM_emb, d_DMM_emb)
        self.AA_emb=TokenEmbedding(n_AA_cls, d_AA_emb, d_AA_emb)
        self.CM_emb=TokenEmbedding(n_CM_cls, d_CM_emb, d_CM_emb)
      if f_key and f_emotion and f_struct and f_Align and f_ND and f_PM and f_PV and f_PR and f_DMM and f_AA and f_CM and f_DM and f_DV and f_DR and f_MCD:
        self.DM_emb = TokenEmbedding(n_DM_cls, d_DM_emb, d_DM_emb)
        self.DV_emb=TokenEmbedding(n_DV_cls, d_DV_emb, d_DV_emb)
        self.DR_emb=TokenEmbedding(n_DR_cls, d_DR_emb, d_DR_emb)
        self.MCD_emb=TokenEmbedding(n_MCD_cls, d_MCD_emb, d_MCD_emb)

    if f_pos and f_tone:
      self.pos_emb=TokenEmbedding(n_pos_cls, d_pos_emb, d_pos_emb)
      self.tone_emb=TokenEmbedding(n_tone_cls, d_tone_emb, d_tone_emb)

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
    
  def get_semantic_emb(self, inp, padding_mask=None):
    token_emb = self.token_emb_lyric(inp)
    enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
    latent_lyric= self.encoder_seqLyric(enc_inp, padding_mask=padding_mask)
    return latent_lyric

  def generate(self, inp, dec_seg_emb, struct=None, key=None, emotion=None, ND=None, Align=None, PM=None, PV=None, PR=None, DMM=None, AA=None, CM=None, DM=None, DV=None, DR=None, MCD=None, vqvae_latent=None, pos=None, tone=None, keep_last_only=True):
    token_emb = self.token_emb(inp)
    dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    if self.f_pos and self.f_tone:
      dec_pos_emb=self.pos_emb(pos)
      dec_tone_emb=self.tone_emb(tone)
      dec_seg_emb=dec_seg_emb+dec_pos_emb+dec_tone_emb

    if self.use_musc_ctls:
      if self.f_key and self.f_emotion and self.f_struct:
        dec_key_emb=self.key_emb(key)
        dec_emotion_emb=self.emotion_emb(emotion)
        dec_struct_emb=self.struct_emb(struct)
        dec_seg_emb_cat=torch.cat([dec_seg_emb,dec_key_emb,dec_emotion_emb,dec_struct_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align:
        dec_Align_emb = self.Align_emb(Align)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_Align_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND:
        dec_ND_emb = self.ND_emb(ND)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_ND_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM:
        dec_PM_emb = self.PM_emb(PM)
        dec_PV_emb=self.PV_emb(PV)
        dec_PR_emb=self.PR_emb(PR)
        dec_DMM_emb=self.DMM_emb(DMM)
        dec_AA_emb=self.AA_emb(AA)
        dec_CM_emb=self.CM_emb(CM)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_PM_emb,dec_PV_emb,dec_PR_emb,dec_DMM_emb,dec_AA_emb,dec_CM_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM and self.f_DM and self.f_DV and self.f_DR and self.f_MCD:
        dec_DM_emb = self.DM_emb(DM)
        dec_DV_emb=self.DV_emb(DV)
        dec_DR_emb=self.DR_emb(DR)
        dec_MCD_emb=self.MCD_emb(MCD)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_DM_emb,dec_DV_emb,dec_DR_emb,dec_MCD_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM and self.f_DM and self.f_DV and self.f_DR and self.f_MCD and self.f_leared_features:
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,vqvae_latent],dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    out = self.decoder(dec_inp, dec_seg_emb_cat)
    out = self.dec_out_proj(out)

    if keep_last_only:
      out = out[-1, ...]

    return out


  def forward(self, enc_inp,dec_inp, dec_inp_seq_pos, struct=None, key=None, emotion=None, ND=None, Align=None, PM=None, PV=None, PR=None, DMM=None, AA=None, CM=None, DM=None, DV=None, DR=None, MCD=None, vqvae_latent=None, pos=None, tone=None, padding_mask=None):
    enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2) 
    enc_token_emb = self.token_emb_lyric(enc_inp) 
    dec_token_emb = self.token_emb(dec_inp) 

    enc_token_emb = enc_token_emb.reshape(enc_inp.size(0), -1, enc_token_emb.size(-1)) 

    enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))  
    dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))  

    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1)) 

    lyric_latent = self.encoder_seqLyric(enc_inp, padding_mask=padding_mask) 

    lyric_latent_reshaped = lyric_latent.reshape(enc_bt_size, enc_n_bars, -1) 
   
    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(lyric_latent.device) 
    for n in range(dec_inp.size(1)):
      for b, (st, ed) in enumerate(zip(dec_inp_seq_pos[n, :-1], dec_inp_seq_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = lyric_latent_reshaped[n, b, :]
  
    if self.f_pos and self.f_tone:
      dec_pos_emb=self.pos_emb(pos)
      dec_tone_emb=self.tone_emb(tone)
      dec_seg_emb=dec_seg_emb+dec_pos_emb+dec_tone_emb

    if self.use_musc_ctls:
      if self.f_key and self.f_emotion and self.f_struct:
        dec_key_emb=self.key_emb(key)
        dec_emotion_emb=self.emotion_emb(emotion)
        dec_struct_emb=self.struct_emb(struct)
        dec_seg_emb_cat=torch.cat([dec_seg_emb,dec_key_emb,dec_emotion_emb,dec_struct_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align:
        dec_Align_emb = self.Align_emb(Align)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_Align_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND:
        dec_ND_emb = self.ND_emb(ND)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_ND_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM:
        dec_PM_emb = self.PM_emb(PM)
        dec_PV_emb=self.PV_emb(PV)
        dec_PR_emb=self.PR_emb(PR)
        dec_DMM_emb=self.DMM_emb(DMM)
        dec_AA_emb=self.AA_emb(AA)
        dec_CM_emb=self.CM_emb(CM)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_PM_emb,dec_PV_emb,dec_PR_emb,dec_DMM_emb,dec_AA_emb,dec_CM_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM and self.f_DM and self.f_DV and self.f_DR and self.f_MCD:
        dec_DM_emb = self.DM_emb(DM)
        dec_DV_emb=self.DV_emb(DV)
        dec_DR_emb=self.DR_emb(DR)
        dec_MCD_emb=self.MCD_emb(MCD)
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,dec_DM_emb,dec_DV_emb,dec_DR_emb,dec_MCD_emb],dim=-1)
      if self.f_key and self.f_emotion and self.f_struct and self.f_Align and self.f_ND and self.f_PM and self.f_PV and self.f_PR and self.f_DMM and self.f_AA and self.f_CM and self.f_DM and self.f_DV and self.f_DR and self.f_MCD and self.f_leared_features:
        dec_seg_emb_cat=torch.cat([dec_seg_emb_cat,vqvae_latent],dim=-1)

    else:
      dec_seg_emb_cat = dec_seg_emb

    dec_out = self.decoder(dec_inp, dec_seg_emb_cat) #dec_out:[dec_seqlen,batch_size,dec_d_model]
    dec_logits = self.dec_out_proj(dec_out) #dec_logits:[dec_seqlen,batch_size,n_token]
    
    return dec_logits

  def compute_loss(self, dec_logits, dec_tgt):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.pad_token_melody, reduction='mean'
    ).float()

    return recons_loss