# CSL-L2M

This repository provides an implementation of the paper:
**_CSL-L2M_: Controllable Song-Level Lyric-to-Melody Generation based on Conditional Transformer with Fine-Grained Lyric and Musical Controls**
submitted to _AAAI-25_.  

[Demo website](https://sites.google.com/view/csl-l2m/)

## Setup
* Python >= 3.6
* Install required packages
```bash
pip install -r requirements.txt
```

## Preprocessing
* Prepare MIDI dataset with corresponding lyrics  
We provide several examples in the `data/Midi` directory. In the future, we plan to share our precisely annotated parallel lyric-melody dataset, which consists of 10,170 Chinese pop songs with a 4/4 time signature.  
* Convert MIDI to REMI-Aligned 
```bash
python mid2events.py [output dir]  
```
* Acquire separate dictionaries for melodies and lyrics
```bash
python events2words.py   
```
(Note: We provide the dictionaries used in our training, i.e., `data/dictionary_melody.pkl` and `data/dictionary_lyric.pkl`.)  

* Extract part-of-speech (POS) tags and tones from the lyrics  
```bash
python extract_PosTone.py  
```
* Compute statistical musical attribute classes 
```bash
python extract_StatisticalAttributes.py  
```
* Extract learned musical features  
   1. First, train a VQ-VAE model
      ```bash
      python train_VQVAE.py [config file]
      ```
      * e.g.
      ```bash
      python train_VQVAE.py config/VQVAE.yaml
      ```
      (Note: We provide our trained [VQ-VAE](https://drive.google.com/file/d/1xyvK3Hasd8IdBa1m4RPTP30eiGZaJyTj/view?usp=drive_link) checkpoint.)
      
   3. Then, extract the learned musical features from the pre-trained VQ-VAE model
      ```bash
      python extract_LearnedFeats.py [config file] [ckpt path] [output dir]
      ```

## Training
```bash
python train_CSLL2M.py [config file]
```
* e.g.
1. Train CSL-L2M
   ```bash
   python train_CSLL2M.py config/CSLL2M.yaml
   ```
3. Train CSL-L2M with learned musical features
   ```bash
   python train_CSLL2M.py config/CSLL2M_withLearedFeats.yaml
   ```
(Note: Training on different controls can be achieved by specifying the _flag_ parameters of the config file. We provide our trained [CSL-L2M](https://drive.google.com/file/d/1ylVTiDd_fwif2ISQzn9bxxYSkfEzXjQu/view?usp=drive_link) and [CSL-L2M with learned musical features](https://drive.google.com/file/d/1U6xazAovM7Yp5d-DtxrzhsF_3NDDIZs2/view?usp=drive_link) checkpoints.)  


## Generation
```bash
python generate_CSLL2M.py [config file] [ckpt path] [output dir] [num songs] [num samples per song]
```
* e.g.
```bash
python generate_CSLL2M.py config/CSLL2M.yaml pretrained_CSLL2M.pt generated_midis/ 1 5
```

## Acknowledgements
Our code is based on [MuseMorphose](https://github.com/YatingMusic/MuseMorphose/) and [FIGARO](https://github.com/dvruette/figaro/).








