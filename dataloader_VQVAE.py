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
  def __init__(self, data_dir, vocab_melody, 
               model_enc_seqlen=128, model_dec_seqlen=1280, model_max_seqs=64,
               pieces=[],  pad_to_same=True, 
               appoint_st_seq=None, dec_end_pad_value=None):
    self.vocab_melody = vocab_melody
    self.read_vocab()

    self.data_dir = data_dir
    self.pieces = pieces
    self.build_dataset()

    self.model_enc_seqlen = model_enc_seqlen
    self.model_dec_seqlen = model_dec_seqlen
    self.model_max_seqs = model_max_seqs

    self.pad_to_same = pad_to_same

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
    _,_,piece_evs = pickle_load(self.pieces[piece_idx])
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
    else:
      picked_seq_pos = np.array(piece_seq_pos + [piece_seq_pos[-1]] * (self.model_max_seqs - len(piece_seq_pos)))
      n_seqs = len(piece_seq_pos)
      assert len(picked_seq_pos) == self.model_max_seqs

    return piece_evs, picked_st_seq, picked_seq_pos, n_seqs

  def pad_sequence(self, seq, maxlen, pad_value):
    assert pad_value is not None
    seq.extend( [pad_value for _ in range(maxlen- len(seq))] )
    return seq

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

    melody_events, st_seq, seq_pos, enc_n_seqs = self.get_sample_from_file(idx)
    
    melody_tokens = convert_event(melody_events, self.event2idx, to_ndarr=False)
    seq_pos = seq_pos.tolist() + [len(melody_tokens)]

    enc_melody_inp, enc_melody_padding_mask, enc_melody_lens= self.get_encoder_melody_input(seq_pos, melody_tokens)

    length = len(melody_tokens)
    if self.pad_to_same:
      inp = self.pad_sequence(melody_tokens, self.model_dec_seqlen + 1,self.pad_token_melody) 
    else:
      inp = self.pad_sequence(melody_tokens, len(melody_tokens) + 1, pad_value=self.dec_end_pad_value)
    target = np.array(inp[1:], dtype=int)
    inp = np.array(inp[:-1], dtype=int)
    assert len(inp) == len(target)

    return {
      'id': idx,
      'piece_id': os.path.basename(self.pieces[idx]).replace('.pkl', ''),
      'st_seq_id': st_seq,
      'seq_pos': np.array(seq_pos, dtype=int),
      "enc_melody_input":enc_melody_inp,
      'dec_input': inp[:self.model_dec_seqlen],
      'dec_target': target[:self.model_dec_seqlen],
      'length': min(length, self.model_dec_seqlen),
      'enc_melody_padding_mask': enc_melody_padding_mask,
      "enc_melody_lens":enc_melody_lens,
      'enc_n_seqs': enc_n_seqs
    }
