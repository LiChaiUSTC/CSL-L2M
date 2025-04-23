import os, pickle, random
from glob import glob

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

def pickle_load(path):
  return pickle.load(open(path, 'rb'))

def convert_event(event_seq, event2idx, to_ndarr=True):
  if isinstance(event_seq[0], dict):
    event_seq = [event2idx['{}_{}'.format(e['name'], e['value'])] for e in event_seq]
  else:
    event_seq = [event2idx[e] for e in event_seq]

  if to_ndarr:
    return np.array(event_seq)
  else:
    return event_seq

def convert_lyrics(lyric_seq, lyric2idx):
  all_lyric_words=[]
  for lyric_list in lyric_seq:
    seq_lyric_words=[]
    for ly in lyric_list:
      seq_lyric_words.append(lyric2idx[ly])
    all_lyric_words.append(seq_lyric_words)
  return all_lyric_words


class REMIalignedFullSongDataset(Dataset):
  def __init__(self, data_dir, vocab_melody,vocab_lyric, 
               model_enc_seqlen=128, model_dec_seqlen=1280, model_max_seqs=64,
               pieces=[],  pad_to_same=True, use_musc_ctls=True, f_pos=True, f_tone=True,f_leared_features=False,
               appoint_st_seq=None, dec_end_pad_value=None):
    self.vocab_melody = vocab_melody
    self.vocab_lyric = vocab_lyric
    self.read_vocab()

    self.data_dir = data_dir
    self.pieces = pieces
    self.build_dataset()

    self.model_enc_seqlen = model_enc_seqlen
    self.model_dec_seqlen = model_dec_seqlen
    self.model_max_seqs = model_max_seqs

    self.pad_to_same = pad_to_same
    self.use_musc_ctls = use_musc_ctls
    self.f_pos=f_pos
    self.f_tone=f_tone
    self.f_leared_features=f_leared_features

    self.appoint_st_seq = appoint_st_seq
    if dec_end_pad_value is None:
      self.dec_end_pad_value = self.pad_token_melody
    elif dec_end_pad_value == 'EOS':
      self.dec_end_pad_value = self.eos_token
    else:
      self.dec_end_pad_value = self.pad_token_melody

  def read_vocab(self):
    self.event2idx,self.idx2event = pickle_load(self.vocab_melody)
    self.bar_token = self.event2idx['Bar_None']
    self.eos_token = self.event2idx['EOS_None']
    self.pad_token_melody = len(self.event2idx)
    self.vocab_size = len(self.event2idx)+1
    self.lyric2idx,self.idx2lyric = pickle_load(self.vocab_lyric)
    self.seq_lyric=len(self.lyric2idx)
    self.pad_token = len(self.lyric2idx)+1
    self.vocab_size_lyric=len(self.lyric2idx)+2
  
  def build_dataset(self):
    if not self.pieces:
      self.pieces = sorted( glob(os.path.join(self.data_dir, '*.pkl')) )
    else:
      self.pieces = sorted( [os.path.join(self.data_dir, p) for p in self.pieces] )

    self.piece_seq_pos = []

    for i, p in enumerate(self.pieces):
      _,seq_pos, p_evs = pickle_load(p)
      if not i % 200:
        print ('[preparing data] now at #{}'.format(i))
      if seq_pos[-1] == len(p_evs):
        seq_pos = seq_pos[:-1]
      if len(p_evs) - seq_pos[-1] == 2:
        seq_pos = seq_pos[:-1]

      seq_pos.append(len(p_evs))

      self.piece_seq_pos.append(seq_pos)

  def get_sample_from_file(self, piece_idx):
    piece_lyrics,_,piece_evs = pickle_load(self.pieces[piece_idx])
    if len(self.piece_seq_pos[piece_idx]) > self.model_max_seqs and self.appoint_st_seq is None:
      picked_st_seq = random.choice(
        range(len(self.piece_seq_pos[piece_idx]) - self.model_max_seqs)
      )
    elif self.appoint_st_seq is not None and self.appoint_st_seq < len(self.piece_seq_pos[piece_idx]) - self.model_max_seqs:
      picked_st_seq = self.appoint_st_seq
    else:
      picked_st_seq = 0

    piece_seq_pos = self.piece_seq_pos[piece_idx]

    if len(piece_seq_pos) > self.model_max_seqs:
      piece_evs = piece_evs[ piece_seq_pos[picked_st_seq] : piece_seq_pos[picked_st_seq + self.model_max_seqs] ]
      picked_seq_pos = np.array(piece_seq_pos[ picked_st_seq : picked_st_seq + self.model_max_seqs ]) - piece_seq_pos[picked_st_seq]
      n_seqs = self.model_max_seqs
      piece_lyrics=piece_lyrics[picked_st_seq:picked_st_seq + self.model_max_seqs]
    else:
      picked_seq_pos = np.array(piece_seq_pos + [piece_seq_pos[-1]] * (self.model_max_seqs - len(piece_seq_pos)))
      n_seqs = len(piece_seq_pos)
      assert len(picked_seq_pos) == self.model_max_seqs

    return piece_lyrics,piece_evs, picked_st_seq, picked_seq_pos, n_seqs

  def pad_sequence(self, seq, maxlen, pad_value):
    assert pad_value is not None
    seq.extend( [pad_value for _ in range(maxlen- len(seq))] )
    return seq

  def get_attr_classes(self, piece, st_seq):
    ND = pickle_load(os.path.join('data/StatisticalAttributes/ND_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    Align = pickle_load(os.path.join('data/StatisticalAttributes/Align_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    PM = pickle_load(os.path.join('data/StatisticalAttributes/PM_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    MCD = pickle_load(os.path.join('data/StatisticalAttributes/MCD_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    DMM = pickle_load(os.path.join('data/StatisticalAttributes/DMM_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    AA = pickle_load(os.path.join('data/StatisticalAttributes/AA_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    CM = pickle_load(os.path.join('data/StatisticalAttributes/CM_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    PV = pickle_load(os.path.join('data/StatisticalAttributes/PV_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    PR = pickle_load(os.path.join('data/StatisticalAttributes/PR_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    DM = pickle_load(os.path.join('data/StatisticalAttributes/DM_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    DV = pickle_load(os.path.join('data/StatisticalAttributes/DV_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    DR = pickle_load(os.path.join('data/StatisticalAttributes/DR_seq_d64', piece))[st_seq : st_seq + self.model_max_seqs]
    struct_events = pickle_load(os.path.join('data/struct', piece))[st_seq : st_seq + self.model_max_seqs][0]

    struct_dict={'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
    struct=[]
    for s in struct_events:
      struct.append(struct_dict[s])

    ND.extend([0 for _ in range(self.model_max_seqs - len(ND))])
    Align.extend([0 for _ in range(self.model_max_seqs - len(Align))])
    PM.extend([0 for _ in range(self.model_max_seqs - len(PM))])
    MCD.extend([0 for _ in range(self.model_max_seqs - len(MCD))])
    DMM.extend([0 for _ in range(self.model_max_seqs - len(DMM))])
    AA.extend([0 for _ in range(self.model_max_seqs - len(AA))])
    CM.extend([0 for _ in range(self.model_max_seqs - len(CM))])
    DM.extend([0 for _ in range(self.model_max_seqs - len(DM))])
    DV.extend([0 for _ in range(self.model_max_seqs - len(DV))])
    PV.extend([0 for _ in range(self.model_max_seqs - len(PV))])
    PR.extend([0 for _ in range(self.model_max_seqs - len(PR))])
    DR.extend([0 for _ in range(self.model_max_seqs - len(DR))])
    struct.extend([0 for _ in range(self.model_max_seqs - len(struct))])

    assert len(ND) == self.model_max_seqs
    assert len(Align) == self.model_max_seqs
    assert len(PM) == self.model_max_seqs
    assert len(MCD) == self.model_max_seqs
    assert len(DMM) == self.model_max_seqs
    assert len(AA) == self.model_max_seqs
    assert len(CM) == self.model_max_seqs
    assert len(DM) == self.model_max_seqs
    assert len(PV) == self.model_max_seqs
    assert len(DV) == self.model_max_seqs
    assert len(PR) == self.model_max_seqs
    assert len(DR) == self.model_max_seqs
    assert len(struct) == self.model_max_seqs
    return ND, Align,MCD,DMM,AA,CM,PM,DM,PV,DV,PR,DR,struct

  def get_pos_tone_classes(self, piece, st_seq):
    pos_seq = pickle_load(os.path.join('data/Pos_Tone', piece))[0][st_seq : st_seq + self.model_max_seqs]
    tone_seq = pickle_load(os.path.join('data/Pos_Tone', piece))[1][st_seq : st_seq + self.model_max_seqs]
    pos2idx={'a': 0, 'ad': 1, 'ag': 2, 'an': 3, 'b': 4, 'c': 5, 'd': 6, 'df': 7, 'dg': 8, 'e': 9, 'eng': 10, 'f': 11, 'g': 12, 'h': 13, 'i': 14, 'j': 15, 'k': 16, 'l': 17, 'm': 18, 'mq': 19, 'n': 20, 'ng': 21, 'nr': 22, 'nrfg': 23, 'nrt': 24, 'ns': 25, 'nt': 26, 'nz': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 'rr': 32, 'rz': 33, 's': 34, 't': 35, 'tg': 36, 'u': 37, 'ud': 38, 'ug': 39, 'uj': 40, 'ul': 41, 'uv': 42, 'uz': 43, 'v': 44, 'vd': 45, 'vg': 46, 'vi': 47, 'vn': 48, 'vq': 49, 'x': 50, 'y': 51, 'yg': 52, 'z': 53, 'zg': 54}
    tone2idx={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    pos_seq_tokens=[]
    tone_seq_tokens=[]
    seqs=len(pos_seq)
    for s in range(seqs):
      pos_token=[]
      tone_token=[]
      numlyrics=len(pos_seq[s])
      for n in range(numlyrics):
        pos_token.append(pos2idx[pos_seq[s][n]])
        tone_token.append(tone2idx[tone_seq[s][n]])
      pos_seq_tokens.append(pos_token)
      tone_seq_tokens.append(tone_token)
    return pos_seq_tokens,tone_seq_tokens

  def get_encoder_input_data(self, seq_lyrics):
    enc_padding_mask = np.ones((self.model_max_seqs, self.model_enc_seqlen), dtype=bool)
    enc_padding_mask[:, :2] = False
    padded_enc_input = np.full((self.model_max_seqs, self.model_enc_seqlen), dtype=int, fill_value=self.pad_token)
    enc_lens = np.zeros((self.model_max_seqs,))
    ind=0
    for lis in seq_lyrics:
      lis.insert(0,self.seq_lyric)
      enc_lens[ind]=len(lis)
      enc_padding_mask[ind, :len(lis)] = False
      within_seq_events = self.pad_sequence(lis, self.model_enc_seqlen, self.pad_token)
      within_seq_events = np.array(within_seq_events)
      padded_enc_input[ind, :] = within_seq_events[:self.model_enc_seqlen]
      ind=ind+1
    return padded_enc_input, enc_padding_mask, enc_lens

  def get_encoder_melody_input(self, seq_positions, seq_events):
    assert len(seq_positions) == self.model_max_seqs + 1
    enc_padding_mask = np.ones((self.model_max_seqs, self.model_enc_seqlen), dtype=bool)
    enc_padding_mask[:, :2] = False
    padded_enc_input = np.full((self.model_max_seqs, self.model_enc_seqlen), dtype=int, fill_value=self.pad_token_melody)
    enc_lens = np.zeros((self.model_max_seqs,))

    for b, (st, ed) in enumerate(zip(seq_positions[:-1], seq_positions[1:])):
      enc_padding_mask[b, : (ed-st)] = False
      enc_lens[b] = ed - st
      within_seq_events = self.pad_sequence(seq_events[st : ed], self.model_enc_seqlen, self.pad_token_melody)
      within_seq_events = np.array(within_seq_events)

      padded_enc_input[b, :] = within_seq_events[:self.model_enc_seqlen]

    return padded_enc_input, enc_padding_mask, enc_lens

  def __len__(self):
    return len(self.pieces)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    seq_lyrics,melody_events, st_seq, seq_pos, enc_n_seqs = self.get_sample_from_file(idx)
    
    if self.f_tone and self.f_pos:
      pos_seq_tokens,tone_seq_tokens= self.get_pos_tone_classes(os.path.basename(self.pieces[idx]), st_seq)
      pos_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      tone_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      all_pos=[]
      for p in pos_seq_tokens:
        for pp in p:
          all_pos.append(pp)
      all_tone=[]
      for t in tone_seq_tokens:
        for tt in t:
          all_tone.append(tt)
   
      assert len(all_pos)==len(all_tone)
      num_aligns=0
      melody_events2=melody_events[:self.model_dec_seqlen]
      event_lens=len(melody_events2)
      fls=False
      for i in range(event_lens):
        if melody_events2[i]['name']=='EOS':
          fls=True
      if fls:
        for ii in range(event_lens-3):
          pos_expanded[ii]=all_pos[num_aligns]
          tone_expanded[ii]=all_tone[num_aligns]
          if melody_events2[ii]['name']=='ALIGN':
            num_aligns=num_aligns+1
        pos_expanded[event_lens-3]=all_pos[num_aligns-1]
        pos_expanded[event_lens-2]=all_pos[num_aligns-1]
        pos_expanded[event_lens-1]=all_pos[num_aligns-1]
        tone_expanded[event_lens-3]=all_tone[num_aligns-1]
        tone_expanded[event_lens-2]=all_tone[num_aligns-1]
        tone_expanded[event_lens-1]=all_tone[num_aligns-1]
      else:
        for ii in range(event_lens):
          if num_aligns>=len(all_pos):
            num_aligns=len(all_pos)-1
          pos_expanded[ii]=all_pos[num_aligns]
          tone_expanded[ii]=all_tone[num_aligns]
          if melody_events2[ii]['name']=='ALIGN':
            num_aligns=num_aligns+1 
    else:
      pos_seq_tokens,tone_seq_tokens=0,0
      pos_expanded,tone_expanded=0,0

    if self.use_musc_ctls:
      ND, Align,MCD,DMM,AA,CM,PM,DM,PV,DV,PR,DR,struct = self.get_attr_classes(os.path.basename(self.pieces[idx]), st_seq)
      ND_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      Align_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      PM_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      MCD_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      DMM_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      AA_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      CM_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      DM_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      PV_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      DV_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      PR_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      DR_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      struct_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)

      if self.f_leared_features:
        learned_features=pickle_load(os.path.join('data/LearnedFeats', os.path.basename(self.pieces[idx])))[st_seq : st_seq + self.model_max_seqs]
        learned_features_expanded1 = np.zeros((self.model_max_seqs,learned_features.shape[-1]),dtype=np.float32)
        learned_features_expanded1[:learned_features.shape[0]]=learned_features
        learned_features_expanded = np.zeros((self.model_dec_seqlen,learned_features.shape[-1]),dtype=np.float32)
      else:
        learned_features=0
      for i, (b_st, b_ed) in enumerate(zip(seq_pos[:-1], seq_pos[1:])):
        ND_expanded[b_st:b_ed] = ND[i]
        Align_expanded[b_st:b_ed] = Align[i]
        PM_expanded[b_st:b_ed] = PM[i]
        MCD_expanded[b_st:b_ed] = MCD[i]
        DMM_expanded[b_st:b_ed] = DMM[i]
        AA_expanded[b_st:b_ed] = AA[i]
        CM_expanded[b_st:b_ed] = CM[i]
        DM_expanded[b_st:b_ed] = DM[i]
        PV_expanded[b_st:b_ed] = PV[i]
        DV_expanded[b_st:b_ed] = DV[i]
        PR_expanded[b_st:b_ed] = PR[i]
        DR_expanded[b_st:b_ed] = DR[i]
        struct_expanded[b_st:b_ed] = struct[i]
        if self.f_leared_features:
          learned_features_expanded[b_st:b_ed,:]=learned_features_expanded1[i]
        else:
          learned_features_expanded,learned_features_expanded1=0,0
    else:
      ND, Align,MCD,DMM,AA,CM,PM,DM,PV,DV,PR,DR,struct,learned_features = 0,0,0,0,0,0,0,0,0,0,0,0,0,0
      ND_expanded, Align_expanded,MCD_expanded,DMM_expanded,AA_expanded,CM_expanded,PM_expanded,DM_expanded,PV_expanded,DV_expanded,PR_expanded,DR_expanded,struct_expanded,learned_features_expanded,learned_features_expanded1=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    melody_tokens = convert_event(melody_events, self.event2idx, to_ndarr=False)
    seq_pos = seq_pos.tolist() + [len(melody_tokens)]

    seq_lyrics_tokens=convert_lyrics(seq_lyrics, self.lyric2idx)

    enc_inp, enc_padding_mask, enc_lens = self.get_encoder_input_data(seq_lyrics_tokens)
    enc_melody_inp, enc_melody_padding_mask, enc_melody_lens= self.get_encoder_melody_input(seq_pos, melody_tokens)

    length = len(melody_tokens)
    if self.pad_to_same:
      inp = self.pad_sequence(melody_tokens, self.model_dec_seqlen + 1,self.pad_token_melody) 
    else:
      inp = self.pad_sequence(melody_tokens, len(melody_tokens) + 1, pad_value=self.dec_end_pad_value)
    target = np.array(inp[1:], dtype=int)
    inp = np.array(inp[:-1], dtype=int)
    assert len(inp) == len(target)

    if self.use_musc_ctls:
      emotion_dict={'Neutral':0, 'Negative':1, 'Positive':2}
      key_dict={'A':0, 'Ab':1, 'Am':2, 'B':3, 'Bb':4, 'Bbm':5, 'Bm':6, 'C':7, 'C#m':8, 'Cm':9, 'D':10, 'D#m':11, 'Db':12, 'Dm':13, 'E':14, 'Eb':15, 'Em':16, 'F':17, 'F#':18, 'F#m':19, 'Fm':20, 'G':21, 'G#m':22, 'Gm':23}

      key=key_dict[pickle_load(os.path.join('data/Key', os.path.basename(self.pieces[idx])))]
      emotion=emotion_dict[pickle_load(os.path.join('data/Emotion', os.path.basename(self.pieces[idx])))]
      key_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      emotion_expanded = np.zeros((self.model_dec_seqlen,), dtype=int)
      for j in range(self.model_dec_seqlen):
        key_expanded[j]=key
        emotion_expanded[j]=emotion
    else:
      emotion,key=0,0
      emotion_expanded,key_expanded=0,0

    return {
      'id': idx,
      'piece_id': os.path.basename(self.pieces[idx]).replace('.pkl', ''),
      'st_seq_id': st_seq,
      'seq_pos': np.array(seq_pos, dtype=int),
      'enc_input': enc_inp,
      "enc_melody_input":enc_melody_inp,
      'dec_input': inp[:self.model_dec_seqlen],
      'dec_target': target[:self.model_dec_seqlen],
      'ND': ND_expanded,
      'Align': Align_expanded,
      'MCD':MCD_expanded,
      'DMM':DMM_expanded,
      'PM':PM_expanded,
      'AA':AA_expanded,
      'CM':CM_expanded,
      'DM':DM_expanded,
      'pos':pos_expanded,
      'tone':tone_expanded,
      'PV':PV_expanded,
      'DV':DV_expanded,
      'PR':PR_expanded,
      'DR':DR_expanded,
      'struct':struct_expanded,
      'key':key_expanded,
      'emotion':emotion_expanded,
      'learned_feats':learned_features_expanded,
      'global_key':key,
      'global_emotion':emotion,
      'ND_seq': np.array(ND),
      'Align_seq': np.array(Align),
      'PM_seq':np.array(PM),
      'MCD_seq':np.array(MCD),
      'DMM_seq':np.array(DMM),
      'AA_seq':np.array(AA),
      'CM_seq':np.array(CM),
      'DM_seq':np.array(DM),
      'PV_seq':np.array(PV),
      'DV_seq':np.array(DV),
      'PR_seq':np.array(PR),
      'DR_seq':np.array(DR),
      'struct_seq':np.array(struct),
      'learned_feats_seq':learned_features_expanded1,
      'length': min(length, self.model_dec_seqlen),
      'enc_padding_mask': enc_padding_mask,
      'enc_melody_padding_mask': enc_melody_padding_mask,
      'enc_length': enc_lens,
      "enc_melody_lens":enc_melody_lens,
      'enc_n_seqs': enc_n_seqs
    }
