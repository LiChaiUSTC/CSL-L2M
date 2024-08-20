import sys, os, time
sys.path.append('./model')
from model.CSLL2M import CSLL2M
from dataloader_CSLL2M import REMIalignedFullSongDataset
from torch.utils.data import DataLoader
from utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
no_kl_steps = config['training']['no_kl_steps']
kl_cycle_steps = config['training']['kl_cycle_steps']
kl_max_beta = config['training']['kl_max_beta']
free_bit_lambda = config['training']['free_bit_lambda']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

ckpt_dir = config['training']['ckpt_dir']
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
log_interval = config['training']['log_interval']
val_interval = config['training']['val_interval']
constant_kl = config['training']['constant_kl']

recons_loss_ema = 0.

def log_epoch(log_file, log_data, is_init=False):
  if is_init:
    with open(log_file, 'w') as f:
      f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

  with open(log_file, 'a') as f:
    f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
      log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
    ))

def compute_loss_ema(ema, batch_loss, decay=0.95):
  if ema == 0.:
    return batch_loss
  else:
    return batch_loss * (1 - decay) + ema * decay

def train_model(epoch, model, dloader, dloader_val, optim, sched):
  model.train()

  print ('[epoch {:03d}] training ...'.format(epoch))
  print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
  st = time.time()
  for batch_idx, batch_samples in enumerate(dloader):
    model.zero_grad()
    batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)  
    batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)     
    batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)    
    batch_inp_seq_pos = batch_samples['seq_pos'].to(device)                 
    batch_dec_lens = batch_samples['length']                                
    batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
    if config['model']['use_musc_ctls']:      
      batch_ND = batch_samples['ND'].permute(1, 0).to(device)
      batch_Align = batch_samples['Align'].permute(1, 0).to(device)  
      batch_PM = batch_samples['PM'].permute(1, 0).to(device) 
      batch_DM = batch_samples['DM'].permute(1, 0).to(device)
      batch_PV = batch_samples['PV'].permute(1, 0).to(device)   
      batch_DV = batch_samples['DV'].permute(1, 0).to(device)   
      batch_PR = batch_samples['PR'].permute(1, 0).to(device)   
      batch_DR = batch_samples['DR'].permute(1, 0).to(device)    
      batch_MCD = batch_samples['MCD'].permute(1, 0).to(device) 
      batch_DMM = batch_samples['DMM'].permute(1, 0).to(device) 
      batch_AA= batch_samples['AA'].permute(1, 0).to(device) 
      batch_CM = batch_samples['CM'].permute(1, 0).to(device) 
      batch_struct = batch_samples['struct'].permute(1, 0).to(device)
      batch_key = batch_samples['key'].permute(1, 0).to(device)    
      batch_emotion = batch_samples['emotion'].permute(1, 0).to(device)
      if config['model']['f_leared_features']:
        batch_vqvae_latent= batch_samples['learned_feats'].permute(1, 0,2).to(device) 
      else:
        batch_vqvae_latent= batch_samples['learned_feats'].to(device) 
    else:      
      batch_ND = batch_samples['ND'].to(device)
      batch_Align = batch_samples['Align'].to(device)  
      batch_PM = batch_samples['PM'].to(device) 
      batch_DM = batch_samples['DM'].to(device)
      batch_PV = batch_samples['PV'].to(device)   
      batch_DV = batch_samples['DV'].to(device)   
      batch_PR = batch_samples['PR'].to(device)   
      batch_DR = batch_samples['DR'].to(device)    
      batch_MCD = batch_samples['MCD'].to(device) 
      batch_DMM = batch_samples['DMM'].to(device) 
      batch_AA= batch_samples['AA'].to(device) 
      batch_CM = batch_samples['CM'].to(device) 
      batch_struct = batch_samples['struct'].to(device)
      batch_key = batch_samples['key'].to(device)    
      batch_emotion = batch_samples['emotion'].to(device)
      batch_vqvae_latent= batch_samples['learned_feats'].to(device) 
    if config['model']['f_pos'] and config['model']['f_tone']:
      batch_pos = batch_samples['pos'].permute(1, 0).to(device)   
      batch_tone = batch_samples['tone'].permute(1, 0).to(device)   
    else: 
      batch_pos = batch_samples['pos'].to(device)   
      batch_tone = batch_samples['tone'].to(device)   
    global trained_steps
    trained_steps += 1

    dec_logits = model(
      batch_enc_inp,batch_dec_inp,
      batch_inp_seq_pos, batch_struct, batch_key,batch_emotion,
      batch_ND,batch_Align,batch_PM,batch_PV,batch_PR,batch_DMM,batch_AA,batch_CM,batch_DM,batch_DV,batch_DR,batch_MCD,batch_vqvae_latent,batch_pos,batch_tone,padding_mask=batch_padding_mask
    )

    losses = model.compute_loss(dec_logits, batch_dec_tgt)
    
    # anneal learning rate
    if trained_steps < lr_warmup_steps:
      curr_lr = max_lr * trained_steps / lr_warmup_steps
      optim.param_groups[0]['lr'] = curr_lr
    else:
      sched.step(trained_steps - lr_warmup_steps)

    # clip gradient & update model
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()

    global recons_loss_ema
    recons_loss_ema = compute_loss_ema(recons_loss_ema, losses.item())
    

    print (' -- epoch {:03d} | batch {:03d}: len: {}\n\t * loss = (RC: {:.4f} ), step = {}, time_elapsed = {:.2f} secs'.format(
      epoch, batch_idx, batch_dec_lens, recons_loss_ema, trained_steps, time.time() - st
    ))

    if not trained_steps % log_interval:
      log_data = {
        'ep': epoch,
        'steps': trained_steps,
        'recons_loss': recons_loss_ema,
        'time': time.time() - st
      }
      log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
      )

    if not trained_steps % val_interval:
      vallosses = validate(model, dloader_val)
      with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
        f.write('[step {}] RC: {:.4f} | [val] | RC: {:.4f} \n'.format(
          trained_steps, 
          recons_loss_ema, 
          np.mean(vallosses)
        ))
      model.train()

    if not trained_steps % ckpt_interval:
      torch.save(model.state_dict(),
        os.path.join(params_dir, 'step_{:d}-RC_{:.3f}.pt'.format(
            trained_steps,
            recons_loss_ema
          ))
      )

  print ('[epoch {:03d}] training completed\n  -- loss = (RC: {:.4f})\n  -- time elapsed = {:.2f} secs.'.format(
    epoch, recons_loss_ema, time.time() - st
  ))
  log_data = {
    'ep': epoch,
    'steps': trained_steps,
    'recons_loss': recons_loss_ema,
    'time': time.time() - st
  }
  log_epoch(
    os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
  )

def validate(model, dloader, n_rounds=8):
  model.eval()
  loss_rec = []

  print ('[info] validating ...')
  with torch.no_grad():
    for i in range(n_rounds):
      print ('[round {}]'.format(i+1))

      for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()

        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)  
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)     
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)    
        batch_inp_seq_pos = batch_samples['seq_pos'].to(device)                                               
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)      
        if config['model']['use_musc_ctls']:      
          batch_ND = batch_samples['ND'].permute(1, 0).to(device)
          batch_Align = batch_samples['Align'].permute(1, 0).to(device)  
          batch_PM = batch_samples['PM'].permute(1, 0).to(device) 
          batch_DM = batch_samples['DM'].permute(1, 0).to(device)
          batch_PV = batch_samples['PV'].permute(1, 0).to(device)   
          batch_DV = batch_samples['DV'].permute(1, 0).to(device)   
          batch_PR = batch_samples['PR'].permute(1, 0).to(device)   
          batch_DR = batch_samples['DR'].permute(1, 0).to(device)    
          batch_MCD = batch_samples['MCD'].permute(1, 0).to(device) 
          batch_DMM = batch_samples['DMM'].permute(1, 0).to(device) 
          batch_AA= batch_samples['AA'].permute(1, 0).to(device) 
          batch_CM = batch_samples['CM'].permute(1, 0).to(device) 
          batch_struct = batch_samples['struct'].permute(1, 0).to(device)
          batch_key = batch_samples['key'].permute(1, 0).to(device)    
          batch_emotion = batch_samples['emotion'].permute(1, 0).to(device)
          if config['model']['f_leared_features']:
            batch_vqvae_latent= batch_samples['learned_feats'].permute(1, 0,2).to(device) 
          else:
            batch_vqvae_latent= batch_samples['learned_feats'].to(device) 
        if config['model']['f_pos'] and config['model']['f_tone']:
          batch_pos = batch_samples['pos'].permute(1, 0).to(device)   
          batch_tone = batch_samples['tone'].permute(1, 0).to(device)  
        else: 
          batch_pos = batch_samples['pos'].to(device)   
          batch_tone = batch_samples['tone'].to(device)   
        dec_logits = model(
          batch_enc_inp,batch_dec_inp,
          batch_inp_seq_pos, batch_struct, batch_key,batch_emotion,
          batch_ND,batch_Align,batch_PM,batch_PV,batch_PR,batch_DMM,batch_AA,batch_CM,batch_DM,batch_DV,batch_DR,batch_MCD,batch_vqvae_latent,batch_pos,batch_tone,padding_mask=batch_padding_mask
        )
        losses = model.compute_loss(dec_logits, batch_dec_tgt)
        if not (batch_idx + 1) % 10:
          print ('batch #{}:'.format(batch_idx + 1), round(losses.item(), 3))

        loss_rec.append(losses.item())
  return loss_rec

if __name__ == "__main__":
  dset = REMIalignedFullSongDataset(
    config['data']['data_dir'], config['data']['vocab_path_melody'], config['data']['vocab_path_lyric'], 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_seqs=config['data']['max_seqs'],
    pieces=pickle_load(config['data']['train_split']),
    pad_to_same=True, use_musc_ctls=config['model']['use_musc_ctls'], f_pos=config['model']['f_pos'], f_tone=config['model']['f_tone'],f_leared_features=config['model']['f_leared_features']
  )
  dset_val = REMIalignedFullSongDataset(
    config['data']['data_dir'], config['data']['vocab_path_melody'], config['data']['vocab_path_lyric'], 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_seqs=config['data']['max_seqs'],
    pieces=pickle_load(config['data']['val_split']),
    pad_to_same=True,use_musc_ctls=config['model']['use_musc_ctls'], f_pos=config['model']['f_pos'], f_tone=config['model']['f_tone'],f_leared_features=config['model']['f_leared_features']
  )
  print ('[info]', '# training samples:', len(dset.pieces))

  dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
  dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

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
  if pretrained_params_path:
    model.load_state_dict( torch.load(pretrained_params_path) )

  model.train()
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print ('[info] model # params:', n_params)

  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=max_lr)
  if pretrained_optim_path:
    optimizer.load_state_dict( torch.load(pretrained_optim_path) )
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, eta_min=min_lr
  )

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(params_dir):
    os.makedirs(params_dir)

  for ep in range(config['training']['max_epochs']):
    train_model(ep+1, model, dloader, dloader_val, optimizer, scheduler)
