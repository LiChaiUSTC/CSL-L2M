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
1. Prepare MIDI dataset with corresponding lyrics
   <pre>
  We provide several examples in the `data/Midi` directory. In the future, we plan to share our precisely annotated parallel lyric-melody dataset, which consists of 10,170 Chinese pop songs with a 4/4 time signature.
  </pre>
2. Convert MIDI to REMI-Aligned 
```bash
python mid2events.py [output dir]
```
3. Acquire separate dictionaries for melodies and lyrics
```bash
python events2words.py 
```
(Note: We provide the dictionaries used in our training, i.e., `data/dictionary_melody.pkl` and `data/dictionary_lyric.pkl`)

4. Extract part-of-speech (POS) tags and tones from the lyrics














