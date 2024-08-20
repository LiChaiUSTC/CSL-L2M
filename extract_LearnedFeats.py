import sys, os, random
sys.path.append('./model')

from model.VQVAE import VQVAE
from dataloader_VQVAE import REMIalignedFullSongDataset

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
import pickle
import torch
import yaml
import numpy as np

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']

ckpt_path = sys.argv[2] #pretrained_VQVAE.pt
out_dir = sys.argv[3]

def get_latent_embedding(model, piece_data):
  # reshape
  batch_inp = piece_data['enc_melody_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_melody_padding_mask'].bool().to(device)

  # get leared musical features
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(batch_inp, padding_mask=batch_padding_mask)

  return piece_latents

if __name__ == "__main__":
  dset = REMIalignedFullSongDataset(
    config['data']['data_dir'], config['data']['vocab_path_melody'], 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_seqs=config['generate']['max_seqs'],
    pieces=pickle_load(config['data']['train_split']),
    pad_to_same=False
  )
  pieces = random.sample(range(len(dset)), len(dset))
  print ('[sampled pieces]', pieces)

  mconf = config['model']
  model = VQVAE(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    n_codes=mconf['n_codes'],n_groups=mconf['n_groups']).to(device)
  model.eval()
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

  times = []
  for p in pieces:
    p_data = dset[p]
    p_id = p_data['piece_id']

    p_data['enc_melody_input'] = p_data['enc_melody_input'][ : p_data['enc_n_seqs'] ]
    p_data['enc_melody_padding_mask'] = p_data['enc_melody_padding_mask'][ : p_data['enc_n_seqs'] ]

    for k in p_data.keys():
      if k not in 'piece_id':
        if not torch.is_tensor(p_data[k]):
          p_data[k] = numpy_to_tensor(p_data[k], device=device)
        else:
          p_data[k] = p_data[k].to(device)

    p_latents = get_latent_embedding(model, p_data)
    p_latents=p_latents.cpu().numpy()

    savepath=os.path.join(out_dir, '{}.pkl'.format(p_id))
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    pickle.dump((p_latents), open(savepath, 'wb'))


