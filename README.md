# CSL-L2M



**_CSL-L2M_: Controllable Song-Level Lyric-to-Melody Generation based on Conditional Transformer with Fine-Grained Lyric and Musical Controls**
submitted to _AAAI-25_.
[<a href="https://sites.google.com/view/csl-l2m/" target="_blank">Demo website</a>]

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
* Extract statistical musical attribute classes 
```bash
python extract_StatisticalAttributes.py  
```
* Extract learned musical features  
   1. Train a VQ-VAE model
      ```bash
      python train_VQVAE.py [config file]
      ```
      * e.g.
      ```bash
      python train_VQVAE.py config/VQVAE.yaml
      ```
      (Note: We provide our trained VQ-VAE(https://drive.google.com/file/d/1xyvK3Hasd8IdBa1m4RPTP30eiGZaJyTj/view?usp=drive_link) checkpoint.)
      
   3. ni












