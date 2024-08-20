import os, pickle
import numpy as np
from collections import Counter
import math
# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 16

dim=64
data_dir = 'data/REMIaligned_events'
PM_out_dir = 'data/StatisticalAttributes/PM_seq_d'+str(dim)
PV_out_dir = 'data/StatisticalAttributes/PV_seq_d'+str(dim)
PR_out_dir = 'data/StatisticalAttributes/PR_seq_d'+str(dim)
DM_out_dir = 'data/StatisticalAttributes/DM_seq_d'+str(dim)
DV_out_dir = 'data/StatisticalAttributes/DV_seq_d'+str(dim)
DR_out_dir = 'data/StatisticalAttributes/DR_seq_d'+str(dim)
ND_out_dir = 'data/StatisticalAttributes/ND_seq_d'+str(dim)
MCD_out_dir = 'data/StatisticalAttributes/MCD_seq_d'+str(dim)
AA_out_dir = 'data/StatisticalAttributes/AA_seq_d'+str(dim)
CM_out_dir = 'data/StatisticalAttributes/CM_seq_d'+str(dim)
DMM_out_dir = 'data/StatisticalAttributes/DMM_seq_d'+str(dim)
Align_out_dir = 'data/StatisticalAttributes/Align_seq_d'+str(dim)

if __name__ == "__main__":
  pieces = [p for p in sorted(os.listdir(data_dir)) if '.pkl' in p]
  if not os.path.exists(PM_out_dir):
    os.makedirs(PM_out_dir)
  if not os.path.exists(PV_out_dir):
    os.makedirs(PV_out_dir)
  if not os.path.exists(PR_out_dir):
    os.makedirs(PR_out_dir)  
  if not os.path.exists(DM_out_dir):
    os.makedirs(DM_out_dir) 
  if not os.path.exists(DV_out_dir):
    os.makedirs(DV_out_dir) 
  if not os.path.exists(DR_out_dir):
    os.makedirs(DR_out_dir) 
  if not os.path.exists(ND_out_dir):
    os.makedirs(ND_out_dir)
  if not os.path.exists(MCD_out_dir):
    os.makedirs(MCD_out_dir)
  if not os.path.exists(AA_out_dir):
    os.makedirs(AA_out_dir) 
  if not os.path.exists(CM_out_dir):
    os.makedirs(CM_out_dir)
  if not os.path.exists(DMM_out_dir):
    os.makedirs(DMM_out_dir)
  if not os.path.exists(Align_out_dir):
    os.makedirs(Align_out_dir)
    
  all_PV = []
  all_PM=[]
  all_PR=[]
  all_DV=[]
  all_DM=[]
  all_DR=[]
  all_ND = []
  all_Align=[]
  all_MCD=[]
  all_AA=[]
  all_CM=[]
  all_DMM=[]
  for p in pieces:
    all_lyrics,pos_seq,events = pickle.load(open(os.path.join(data_dir, p),'rb'))
    PV=[]
    PM=[]
    PR=[]
    DV=[]
    DM=[]
    DR=[]
    ND=[]
    Align=[]
    MCD=[]
    AA=[]
    CM=[]
    DMM=[]

    seq_pitch=[]
    seq_dur=[]
    jj=0
    nn=0
    num_notes=0
    cur_bar=0
    mm=0
    for ev in events:
      if ev['name']=='Note_Pitch':
        seq_pitch.append(ev['value'])
        num_notes=num_notes+1
      elif ev['name']=='Note_Duration':
        seq_dur.append(ev['value']//TICK_RESOL)
      elif ev['name']=='ALIGN':
        mm=mm+1
      elif ev['name'] == 'Bar' and events[jj-1]['name']!='SEQ' and events[jj+1]['name']!='SEQ' and events[jj+1]['name']=='Beat':
        cur_bar += 1
      elif ev['name'] == 'Beat':
        cur_pos= int(ev['value'])
        nn=nn+1
        if nn==1:
          first_pos=cur_pos
      elif ev['name']=='SEQ' and jj!=0:
        assert len(seq_pitch)==len(seq_dur)
        rec_idx = cur_bar * 64 + cur_pos-first_pos
        PV.append(np.std(seq_pitch))
        PR.append(max(seq_pitch)-min(seq_pitch))
        PM.append(np.mean(seq_pitch))
        DV.append(np.std(seq_dur))
        DM.append(np.mean(seq_dur))
        DR.append(max(seq_dur)-min(seq_dur))
        ND.append(num_notes/rec_idx)
        Align.append(mm/num_notes)
        MCD.append(seq_dur.count(8)/len(seq_dur))
        if len(seq_pitch)>1:
          numpy_pitch=np.array(seq_pitch)
          new_pitch=np.abs(numpy_pitch[1:]-numpy_pitch[:-1]).tolist()
          AA.append((new_pitch.count(0)+new_pitch.count(3)+new_pitch.count(4)+new_pitch.count(7)+new_pitch.count(10)+new_pitch.count(11)+new_pitch.count(12)+new_pitch.count(13)+new_pitch.count(14))/len(new_pitch))
          CM.append(new_pitch.count(1)/len(new_pitch))
          DMM.append(np.sum(numpy_pitch[1:]-numpy_pitch[:-1]>0)/len(new_pitch))
        else:
          AA.append(0)
          CM.append(0)
          DMM.append(0)

        seq_pitch=[]
        seq_dur=[]
        num_notes=0
        cur_bar=0
        mm=0
        nn=0
      jj=jj+1
        
    all_PV.extend(PV)
    all_PM.extend(PM)
    all_PR.extend(PR)
    all_DV.extend(DV)
    all_DM.extend(DM)
    all_DR.extend(DR)
    all_ND.extend(ND)
    all_Align.extend(Align)
    all_MCD.extend(MCD)
    all_AA.extend(AA)
    all_CM.extend(CM)
    all_DMM.extend(DMM)

  nums=len(all_PM)
  ss=int(nums/dim)
  PM_bounds=[]
  PV_bounds=[]
  PR_bounds=[]
  DM_bounds=[]
  DV_bounds=[]
  DR_bounds=[]
  ND_bounds=[]
  MCD_bounds=[]
  AA_bounds=[]
  CM_bounds=[]
  DMM_bounds=[]
  Align_bounds=[]
  for j in range(ss,nums-ss,ss):
    PM_bounds.append(sorted(all_PM)[j])
    PV_bounds.append(sorted(all_PV)[j])
    PR_bounds.append(sorted(all_PR)[j])
    DM_bounds.append(sorted(all_DM)[j])
    DV_bounds.append(sorted(all_DV)[j])
    DR_bounds.append(sorted(all_DR)[j])
    ND_bounds.append(sorted(all_ND)[j])
    MCD_bounds.append(sorted(all_MCD)[j])
    AA_bounds.append(sorted(all_AA)[j])
    CM_bounds.append(sorted(all_CM)[j])
    DMM_bounds.append(sorted(all_DMM)[j])
    Align_bounds.append(sorted(all_Align)[j])
 
  for p in pieces:
    all_lyrics,pos_seq,events = pickle.load(open(os.path.join(data_dir, p),'rb'))
    PV=[]
    PM=[]
    PR=[]
    DV=[]
    DM=[]
    DR=[]
    ND=[]
    Align=[]
    MCD=[]
    AA=[]
    CM=[]
    DMM=[]

    seq_pitch=[]
    seq_dur=[]
    jj=0
    nn=0
    num_notes=0
    cur_bar=0
    mm=0
    for ev in events:
      if ev['name']=='Note_Pitch':
        seq_pitch.append(ev['value'])
        num_notes=num_notes+1
      elif ev['name']=='Note_Duration':
        seq_dur.append(ev['value']//TICK_RESOL)
      elif ev['name']=='ALIGN':
        mm=mm+1
      elif ev['name'] == 'Bar' and events[jj-1]['name']!='SEQ' and events[jj+1]['name']!='SEQ' and events[jj+1]['name']=='Beat':
        cur_bar += 1
      elif ev['name'] == 'Beat':
        cur_pos= int(ev['value'])
        nn=nn+1
        if nn==1:
          first_pos=cur_pos
      elif ev['name']=='SEQ' and jj!=0:
        assert len(seq_pitch)==len(seq_dur)
        rec_idx = cur_bar * 64 + cur_pos-first_pos
        PV.append(np.std(seq_pitch))
        PR.append(max(seq_pitch)-min(seq_pitch))
        PM.append(np.mean(seq_pitch))
        DV.append(np.std(seq_dur))
        DM.append(np.mean(seq_dur))
        DR.append(max(seq_dur)-min(seq_dur))
        ND.append(num_notes/rec_idx)
        Align.append(mm/num_notes)
        MCD.append(seq_dur.count(8)/len(seq_dur))
        if len(seq_pitch)>1:
          numpy_pitch=np.array(seq_pitch)
          new_pitch=np.abs(numpy_pitch[1:]-numpy_pitch[:-1]).tolist()
          AA.append((new_pitch.count(0)+new_pitch.count(3)+new_pitch.count(4)+new_pitch.count(7)+new_pitch.count(10)+new_pitch.count(11)+new_pitch.count(12)+new_pitch.count(13)+new_pitch.count(14))/len(new_pitch))
          CM.append(new_pitch.count(1)/len(new_pitch))
          DMM.append(np.sum(numpy_pitch[1:]-numpy_pitch[:-1]>0)/len(new_pitch))
        else:
          AA.append(0)
          CM.append(0)
          DMM.append(0)

        seq_pitch=[]
        seq_dur=[]
        num_notes=0
        cur_bar=0
        mm=0
        nn=0
      jj=jj+1
    
    PV_cls = np.searchsorted(PV_bounds, np.array(PV)).tolist()
    DV_cls = np.searchsorted(DV_bounds, np.array(DV)).tolist()
    PM_cls = np.searchsorted(PM_bounds, np.array(PM)).tolist()
    DM_cls = np.searchsorted(DM_bounds, np.array(DM)).tolist()
    PR_cls = np.searchsorted(PR_bounds, np.array(PR)).tolist()
    DR_cls = np.searchsorted(DR_bounds, np.array(DR)).tolist()
    ND_cls = np.searchsorted(ND_bounds, np.array(ND)).tolist()
    MCD_cls = np.searchsorted(MCD_bounds, np.array(MCD)).tolist()
    AA_cls = np.searchsorted(AA_bounds, np.array(AA)).tolist()
    CM_cls = np.searchsorted(CM_bounds, np.array(CM)).tolist()
    DMM_cls = np.searchsorted(DMM_bounds, np.array(DMM)).tolist()
    Align_cls = np.searchsorted(Align_bounds, np.array(Align)).tolist()

    
    pickle.dump(PV_cls, open(os.path.join(PV_out_dir, p), 'wb'))
    pickle.dump(DV_cls, open(os.path.join(DV_out_dir, p), 'wb'))
    pickle.dump(PM_cls, open(os.path.join(PM_out_dir, p), 'wb'))
    pickle.dump(DM_cls, open(os.path.join(DM_out_dir, p), 'wb'))
    pickle.dump(PR_cls, open(os.path.join(PR_out_dir, p), 'wb'))
    pickle.dump(DR_cls, open(os.path.join(DR_out_dir, p), 'wb'))
    pickle.dump(ND_cls, open(os.path.join(ND_out_dir, p), 'wb'))
    pickle.dump(MCD_cls, open(os.path.join(MCD_out_dir, p), 'wb'))
    pickle.dump(AA_cls, open(os.path.join(AA_out_dir, p), 'wb'))
    pickle.dump(CM_cls, open(os.path.join(CM_out_dir, p), 'wb'))
    pickle.dump(DMM_cls, open(os.path.join(DMM_out_dir, p), 'wb'))
    pickle.dump(Align_cls, open(os.path.join(Align_out_dir, p), 'wb'))
    
    