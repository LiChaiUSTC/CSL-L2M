import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from dataloader_CSLL2M import REMIalignedFullSongDataset
from model.CSLL2M import CSLL2M

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from REMIaligned2midi import REMIaligned2midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 64 

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
#vocab_path = config['data']['vocab_path']
#data_split = config['data']['test_split']

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
n_pieces = int(sys.argv[4])
n_samples_per_piece = int(sys.argv[5])

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

########################################
# generation
########################################
def get_semantic_embedding(model, piece_data):
  # reshape
  batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_semantic_emb(
      batch_inp, padding_mask=batch_padding_mask)
  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, seq_lyric_list, ND, Align, PM, DM, PV, DV, PR, DR, MCD, DMM, AA, CM, struct, key, emotion, 
        learned_feats, pos, tone,
        event2idx, idx2event, 
        max_events=12800, primer=None,
        nucleus_p=0.9, temperature=1.2
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  ND_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  Align_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PV_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DV_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  PR_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DR_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  MCD_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  DMM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  AA_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  CM_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  struct_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  key_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  emotion_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  learned_feats_placeholder=torch.zeros(max_events, 1, latents.size(-1)).to(device)
  pos_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  tone_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  if primer is None:
    generated = [event2idx['SEQ_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    if config['model']['f_pos'] and config['model']['f_tone']:
      pos_placeholder[:len(generated), 0] = pos[0]
      tone_placeholder[:len(generated), 0] = tone[0]
    if config['model']['use_musc_ctls']:
      ND_placeholder[:len(generated), 0] = ND[0]
      Align_placeholder[:len(generated), 0] = Align[0]
      PM_placeholder[:len(generated), 0] = PM[0]
      DM_placeholder[:len(generated), 0] = DM[0]
      PV_placeholder[:len(generated), 0] = PV[0]
      DV_placeholder[:len(generated), 0] = DV[0]
      PR_placeholder[:len(generated), 0] = PR[0]
      DR_placeholder[:len(generated), 0] = DR[0]
      MCD_placeholder[:len(generated), 0] = MCD[0]
      DMM_placeholder[:len(generated), 0] = DMM[0]
      AA_placeholder[:len(generated), 0] = AA[0]
      CM_placeholder[:len(generated), 0] = CM[0]
      struct_placeholder[:len(generated), 0] = struct[0]
      key_placeholder[:len(generated), 0] = key
      emotion_placeholder[:len(generated), 0] = emotion
      if config['model']['f_leared_features']:
        learned_feats_placeholder[:len(generated), 0, :] = learned_feats[0]

  target_bars, generated_bars = latents.size(0), 0

  steps = 0
  time_st = time.time()
  last_start_tick = 0
  cur_dur=0
  cur_bar=0
  failed_cnt = 0
  failed_eos=0
  seq_num_align=0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated) == 1:
      dec_input = numpy_to_tensor([generated], device=device).long()
    else:
      dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    if config['model']['f_pos'] and config['model']['f_tone']:
      pos_placeholder[len(generated)-1, 0] = pos[ generated_bars ]
      tone_placeholder[len(generated)-1, 0] = tone[ generated_bars ]
    if config['model']['use_musc_ctls']:
      ND_placeholder[len(generated)-1, 0] = ND[ generated_bars ]
      Align_placeholder[len(generated)-1, 0] = Align[ generated_bars ]
      PM_placeholder[len(generated)-1, 0] = PM[ generated_bars ]
      DM_placeholder[len(generated)-1, 0] = DM[ generated_bars ]
      PV_placeholder[len(generated)-1, 0] = PV[ generated_bars ]
      DV_placeholder[len(generated)-1, 0] = DV[ generated_bars ]
      PR_placeholder[len(generated)-1, 0] = PR[ generated_bars ]
      DR_placeholder[len(generated)-1, 0] = DR[ generated_bars ]
      MCD_placeholder[len(generated)-1, 0] = MCD[ generated_bars ]
      DMM_placeholder[len(generated)-1, 0] = DMM[ generated_bars ]
      AA_placeholder[len(generated)-1, 0] = AA[ generated_bars ]
      CM_placeholder[len(generated)-1, 0] = CM[ generated_bars ]
      struct_placeholder[len(generated)-1, 0] = struct[ generated_bars ]
      key_placeholder[len(generated)-1, 0] = key
      emotion_placeholder[len(generated)-1, 0] = emotion
      if config['model']['f_leared_features']:
        learned_feats_placeholder[len(generated)-1, 0, :] = learned_feats[generated_bars]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_pos=pos_placeholder[:len(generated), :]
    dec_tone=tone_placeholder[:len(generated), :]
    dec_ND = ND_placeholder[:len(generated), :]
    dec_Align = Align_placeholder[:len(generated), :]
    dec_PM = PM_placeholder[:len(generated), :]
    dec_DM = DM_placeholder[:len(generated), :]
    dec_PV = PV_placeholder[:len(generated), :]
    dec_DV = DV_placeholder[:len(generated), :]
    dec_PR = PR_placeholder[:len(generated), :]
    dec_DR = DR_placeholder[:len(generated), :]
    dec_MCD = MCD_placeholder[:len(generated), :]
    dec_DMM = DMM_placeholder[:len(generated), :]
    dec_AA = AA_placeholder[:len(generated), :]
    dec_CM = CM_placeholder[:len(generated), :]
    dec_struct = struct_placeholder[:len(generated), :]
    dec_key = key_placeholder[:len(generated), :]
    dec_emotion = emotion_placeholder[:len(generated), :]
    dec_learned_feats = learned_feats_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_struct, dec_key, dec_emotion, dec_ND, dec_Align, dec_PM, dec_PV, dec_PR, dec_DMM, dec_AA, dec_CM, dec_DM, dec_DV, dec_DR, dec_MCD, dec_learned_feats, dec_pos,dec_tone)
    logits = tensor_to_numpy(logits[0])
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    if len(seq_lyric_list[generated_bars])==seq_num_align:
      if word!=event2idx['SEQ_None']:
        word=event2idx['SEQ_None']
    else:
      if word==event2idx['SEQ_None']:
        logits[-2]=min(logits)
        probs = temperatured_softmax(logits, temperature)
        word = nucleus(probs, nucleus_p)

    word_event = idx2event[word]
    print("generated word_event****:",word_event)

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      start_tick = cur_bar * DEFAULT_BAR_RESOL + event_pos * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
      if not start_tick >= (last_start_tick+cur_dur):
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated
        continue
      else:
        last_start_tick=start_tick
        failed_cnt = 0

    if 'Bar' in word_event and len(generated)>2:
      cur_bar+=1
    if 'SEQ' in word_event:
      generated_bars += 1
      seq_num_align=0
    if 'ALIGN' in word_event:
      seq_num_align=seq_num_align+1
    if 'Note_Duration' in word_event:
      cur_dur=int(word_event.split('_')[-1])
    if generated_bars < target_bars - 1 and word_event == 'EOS_None':
      failed_eos += 1
      print ('[info] error EOS occurs, failed eos:', failed_eos)
      if failed_eos >= 128:
        print ('[FATAL] model stuck, eos error ...')
        return generated
      continue

    if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['SEQ_None'])
      print ('[info] gotten eos')
      break

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1]


########################################
# change attribute classes
########################################
def random_shift_attr_cls(n_samples, upper=6, lower=-6, flag_transfer=False):
  if flag_transfer:
    return np.random.randint(lower, upper, (n_samples,))
  else:
    return np.random.randint(0, 1, (n_samples,))


if __name__ == "__main__":
  dset = REMIalignedFullSongDataset(
    data_dir,config['data']['vocab_path_melody'], config['data']['vocab_path_lyric'],  
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_seqs=config['generate']['max_seqs'],
    #pieces=['红颜一声叹-黄静美-116-E.pkl'],
    pieces=pickle_load(config['data']['test_split']),
    pad_to_same=False, use_musc_ctls=config['model']['use_musc_ctls'], f_pos=config['model']['f_pos'], f_tone=config['model']['f_tone'],f_leared_features=config['model']['f_leared_features']
  )
  pieces = random.sample(range(len(dset)), n_pieces)
  print ('[sampled pieces]', pieces)
  
  mconf = config['model']
  model = CSLL2M(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,dset.vocab_size_lyric,dset.pad_token_melody,
    d_pos_emb=mconf['d_pos_emb'], d_tone_emb=mconf['d_tone_emb'],
    d_struct_emb=mconf['d_struct_emb'], d_key_emb=mconf['d_key_emb'], d_emotion_emb=mconf['d_emotion_emb'], 
    d_PM_emb=mconf['d_PM_emb'], d_PV_emb=mconf['d_PV_emb'], d_PR_emb=mconf['d_PR_emb'], d_DMM_emb=mconf['d_DMM_emb'], d_AA_emb=mconf['d_AA_emb'], d_CM_emb=mconf['d_CM_emb'], 
    d_DM_emb=mconf['d_DM_emb'], d_DV_emb=mconf['d_DV_emb'], d_DR_emb=mconf['d_DR_emb'], d_MCD_emb=mconf['d_MCD_emb'],
    d_ND_emb=mconf['d_ND_emb'], d_Align_emb=mconf['d_Align_emb'],
    d_learned_features=mconf['d_learned_features'],
    n_pos_cls=mconf['n_pos_cls'], n_tone_cls=mconf['n_tone_cls'],
    n_struct_cls=mconf['n_struct_cls'], n_key_cls=mconf['n_key_cls'], n_emotion_cls=mconf['n_emotion_cls'], 
    n_PM_cls=mconf['n_PM_cls'], n_PV_cls=mconf['n_PV_cls'], n_PR_cls=mconf['n_PR_cls'], n_DMM_cls=mconf['n_DMM_cls'], n_AA_cls=mconf['n_AA_cls'], n_CM_cls=mconf['n_CM_cls'], 
    n_DM_cls=mconf['n_DM_cls'], n_DV_cls=mconf['n_DV_cls'], n_DR_cls=mconf['n_DR_cls'], n_MCD_cls=mconf['n_MCD_cls'],
    n_ND_cls=mconf['n_ND_cls'], n_Align_cls=mconf['n_Align_cls'],
    f_pos=mconf['f_pos'], f_tone=mconf['f_tone'],
    f_struct=mconf['f_struct'], f_key=mconf['f_key'], f_emotion=mconf['f_emotion'], 
    f_PM=mconf['f_PM'], f_PV=mconf['f_PV'], f_PR=mconf['f_PR'], f_DMM=mconf['f_DMM'], f_AA=mconf['f_AA'], f_CM=mconf['f_CM'], 
    f_DM=mconf['f_DM'], f_DV=mconf['f_DV'], f_DR=mconf['f_DR'], f_MCD=mconf['f_MCD'],
    f_ND=mconf['f_ND'], f_Align=mconf['f_Align'],
    f_leared_features=mconf['f_leared_features'],
    use_musc_ctls=mconf['use_musc_ctls']).to(device)
  model.eval()
  pretrained_dict=torch.load(ckpt_path, map_location='cpu')

  adjusted_weights = {}
  for k, v in pretrained_dict.items():
    if k in model.state_dict():
      adjusted_weights[k] = v 

  try:
    model.load_state_dict(pretrained_dict, strict=False)
  except:
    model.load_state_dict(adjusted_weights, strict=False)

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for p in pieces:
    # fetch test sample
    p_data = dset[p]
    p_id = p_data['piece_id']
    p_seq_id = p_data['st_seq_id']
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_seqs']-1 ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_seqs']-1 ]
    
    orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    orig_song = word2event(orig_song, dset.idx2event)
    orig_out_file = os.path.join(out_dir, '{}_seq{}_orig'.format(p_id, p_seq_id))

    lyrics=[]
    for lyriclist in p_data['enc_input']:
      for lyric in lyriclist:
        if lyric!=dset.vocab_size_lyric-1 and lyric!=dset.vocab_size_lyric-2:
          lyrics.append(dset.idx2lyric[lyric])
    
    lys, bar_pos, p_evs = pickle_load('data/REMIaligned_events/'+p_id+'.pkl')

    print ('[info] writing to ...', orig_out_file)
    # output reference song's MIDI
    specified_tempo=90 
    _ = REMIaligned2midi(lyrics, orig_song, specified_tempo, orig_out_file + '.mid')
    
    for k in p_data.keys():
      if k not in 'piece_id':
        if not torch.is_tensor(p_data[k]):
          p_data[k] = numpy_to_tensor(p_data[k], device=device)
        else:
          p_data[k] = p_data[k].to(device)

    if mconf['f_pos'] and mconf['f_tone']:
      pos_seq = pickle_load('data/Pos_Tone/'+p_id+'.pkl')[0]
      tone_seq = pickle_load('data/Pos_Tone/'+p_id+'.pkl')[1]
      pos2idx={'a': 0, 'ad': 1, 'ag': 2, 'an': 3, 'b': 4, 'c': 5, 'd': 6, 'df': 7, 'dg': 8, 'e': 9, 'eng': 10, 'f': 11, 'g': 12, 'h': 13, 'i': 14, 'j': 15, 'k': 16, 'l': 17, 'm': 18, 'mq': 19, 'n': 20, 'ng': 21, 'nr': 22, 'nrfg': 23, 'nrt': 24, 'ns': 25, 'nt': 26, 'nz': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 'rr': 32, 'rz': 33, 's': 34, 't': 35, 'tg': 36, 'u': 37, 'ud': 38, 'ug': 39, 'uj': 40, 'ul': 41, 'uv': 42, 'uz': 43, 'v': 44, 'vd': 45, 'vg': 46, 'vi': 47, 'vn': 48, 'vq': 49, 'x': 50, 'y': 51, 'yg': 52, 'z': 53, 'zg': 54}
      tone2idx={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
      pos_tokens=[]
      tone_tokens=[]
      seqs=len(pos_seq)
      for s in range(seqs):
        num_lyrics=len(pos_seq[s])
        for n in range(num_lyrics):
          pos_tokens.append(pos2idx[pos_seq[s][n]])
          tone_tokens.append(tone2idx[tone_seq[s][n]])
    else:
      pos_tokens=None
      tone_tokens=None

    p_latents= get_semantic_embedding(model, p_data)

    flag_trans=False #specified by users
    p_ND_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_Align_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_PM_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_DM_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_PV_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_DV_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_PR_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_DR_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_MCD_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_DMM_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_AA_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)
    p_CM_diff = random_shift_attr_cls(n_samples_per_piece, flag_transfer=flag_trans)

    for samp in range(n_samples_per_piece):
      p_ND = (p_data['ND_seq'] + p_ND_diff[samp]).clamp(0,63).long()
      p_Align = (p_data['Align_seq'] + p_Align_diff[samp]).clamp(0,47).long()
      p_PM = (p_data['PM_seq'] + p_PM_diff[samp]).clamp(0,63).long()
      p_DM = (p_data['DM_seq'] + p_DM_diff[samp]).clamp(0,63).long()
      p_PV = (p_data['PV_seq'] + p_PV_diff[samp]).clamp(0,63).long()
      p_DV = (p_data['DV_seq'] + p_DV_diff[samp]).clamp(0,63).long()
      p_PR = (p_data['PR_seq'] + p_PR_diff[samp]).clamp(0,63).long()
      p_DR = (p_data['DR_seq'] + p_DR_diff[samp]).clamp(0,63).long()
      p_MCD = (p_data['MCD_seq'] + p_MCD_diff[samp]).clamp(0,63).long()
      p_DMM = (p_data['DMM_seq'] + p_DMM_diff[samp]).clamp(0,63).long()
      p_AA = (p_data['AA_seq'] + p_AA_diff[samp]).clamp(0,63).long()
      p_CM = (p_data['CM_seq'] + p_CM_diff[samp]).clamp(0,63).long() 

      p_struct = (p_data['struct_seq']).long()
      p_key = (p_data['global_key']).long()
      p_emotion = (p_data['global_emotion']).long()

      p_learned_feats=p_data['learned_feats_seq']

      print ('[info] piece: {}, seq: {}'.format(p_id, p_seq_id))
      out_file = os.path.join(out_dir, '{}_seq{}_sample{:02d}'.format(p_id, p_seq_id, samp + 1))      
      # generate
      song= generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents,lys, p_ND, p_Align, p_PM, p_DM, p_PV, p_DV, p_PR, p_DR, p_MCD, p_DMM, p_AA, p_CM, p_struct, p_key, p_emotion, 
                                  p_learned_feats, pos_tokens, tone_tokens,
                                  dset.event2idx, dset.idx2event,
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature']
                                )
      song = word2event(song, dset.idx2event)
      _=REMIaligned2midi(lyrics, song, specified_tempo, out_file +'.mid')
      
 
