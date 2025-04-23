import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from model.CSLL2M import CSLL2M
from itertools import chain
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
lyric2idx,idx2lyric = pickle_load(config['data']['vocab_path_lyric'])
event2idx,idx2event = pickle_load(config['data']['vocab_path_melody'])

def convert_lyrics(lyric_seq, ly2idx):
  all_lyric_words=[]
  for lyric_list in lyric_seq:
    seq_lyric_words=[]
    for ly in lyric_list:
      seq_lyric_words.append(ly2idx[ly])
    all_lyric_words.append(seq_lyric_words)
  return all_lyric_words


def pad_sequence(seq, maxlen, pad_value):
  assert pad_value is not None
  seq.extend( [pad_value for _ in range(maxlen- len(seq))] )
  return seq

def get_encoder_input_data(seq_lyrics):
  enc_padding_mask = np.ones((len(seq_lyrics), config['data']['enc_seqlen']), dtype=bool)
  enc_padding_mask[:, :2] = False
  padded_enc_input = np.full((len(seq_lyrics), config['data']['enc_seqlen']), dtype=int, fill_value=len(lyric2idx)+1)
  enc_lens = np.zeros((len(seq_lyrics),))
  ind=0
  for lis in seq_lyrics:
    lis.insert(0,len(lyric2idx))
    enc_lens[ind]=len(lis)
    enc_padding_mask[ind, :len(lis)] = False
    within_seq_events = pad_sequence(lis, config['data']['enc_seqlen'], len(lyric2idx)+1)
    within_seq_events = np.array(within_seq_events)
    padded_enc_input[ind, :] = within_seq_events[:config['data']['enc_seqlen']]
    ind=ind+1
  return padded_enc_input, enc_padding_mask, enc_lens

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
def get_semantic_embedding(model, enc_input,enc_padding_mask):
  # reshape
  batch_inp = enc_input.permute(1, 0).long().to(device)
  batch_padding_mask = enc_padding_mask.bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_semantic_emb(
      batch_inp, padding_mask=batch_padding_mask)
  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, seq_lyric_list,
        event2idx, idx2event, 
        max_events=12800, primer=None,
        nucleus_p=0.9, temperature=1.2
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  
  if primer is None:
    generated = [event2idx['SEQ_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)

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
 
    dec_seg_emb = latent_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb)
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


if __name__ == "__main__":
  seq_lyrics=[['红', '颜', '一', '声', '叹'], ['人', '生', '过', '半', '为', '难'], ['风', '雨', '来', '作', '伴'], ['日', '月', '沧', '桑', '变', '幻'], ['彼', '岸', '花', '无', '岸'], ['天', '涯', '洒', '落', '花', '瓣'], ['忘', '了', '遥', '遥', '无', '期', '的', '尘', '缘'], ['红', '颜', '一', '声', '叹'], ['风', '花', '雪', '月', '谁', '伴'], ['奈', '何', '君', '不', '见'], ['尘', '世', '浮', '华', '三', '千'], ['云', '水', '间', '再', '叹'], ['渐', '逝', '妖', '娆', '容', '颜'], ['终', '是', '红', '颜', '情', '断', '过', '前', '川'], ['昨', '日', '梨', '花', '寒'], ['抹', '泪', '风', '中', '流', '转'], ['那', '千', '回', '思', '念', '是', '寒', '雾', '浓', '烟'], ['三', '生', '三', '世', '的', '温', '柔', '是', '我', '今', '世', '眷', '恋'], ['恍', '然', '一', '梦', '似', '风', '飘', '远'], ['红', '颜', '一', '声', '叹'], ['人', '生', '过', '半', '为', '难'], ['风', '雨', '来', '作', '伴'], ['日', '月', '沧', '桑', '变', '幻'], ['彼', '岸', '花', '无', '岸'], ['天', '涯', '洒', '落', '花', '瓣'], ['忘', '了', '遥', '遥', '无', '期', '的', '尘', '缘'], ['红', '颜', '一', '声', '叹'], ['风', '花', '雪', '月', '谁', '伴'], ['奈', '何', '君', '不', '见'], ['尘', '世', '浮', '华', '三', '千'], ['云', '水', '间', '再', '叹'], ['渐', '逝', '妖', '娆', '容', '颜'], ['终', '是', '红', '颜', '情', '断', '过', '前', '川']]
  merged_lyrics = list(chain.from_iterable(seq_lyrics))
  seq_lyrics_tokens=convert_lyrics(seq_lyrics, lyric2idx)
  enc_inp, enc_padding_mask, enc_lens = get_encoder_input_data(seq_lyrics_tokens)
  
  mconf = config['model']
  model = CSLL2M(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], len(event2idx)+1,len(lyric2idx)+2,len(event2idx),
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

    
  enc_inp = numpy_to_tensor(enc_inp, device=device)
  enc_padding_mask = numpy_to_tensor(enc_padding_mask, device=device)
       
  p_latents= get_semantic_embedding(model, enc_inp,enc_padding_mask)


  for samp in range(n_samples_per_piece):
    out_file = os.path.join(out_dir, 'sample{:02d}'.format(samp + 1))
    # generate
    song= generate_on_latent_ctrl_vanilla_truncate(model, p_latents,seq_lyrics,event2idx, idx2event,nucleus_p=config['generate']['nucleus_p'], temperature=config['generate']['temperature'])
    song = word2event(song, idx2event)
    _=REMIaligned2midi(merged_lyrics, song, 90, out_file +'.mid')
      
 
