import os
import pickle
import numpy as np
import glob
import jieba.posseg as pseg
import pypinyin
from pypinyin import Style

if __name__ == '__main__':
    lines=glob.glob('data/REMIaligned_events/'+'*.pkl')
    for line in lines:
        lyrics,_,_=pickle.load(open(line,'rb'))
        pos_all=[]
        tone_all=[]
        for seq_lyric in lyrics:
            str_lyric = "".join(seq_lyric)
            results=pypinyin.pinyin(str_lyric, style=Style.TONE3, heteronym=False)
            tone_seq=[]
            for ii in range(len(seq_lyric)):
                tone_value=results[ii][0][-1]
                if tone_value.isdigit():
                    tone_seq.append(int(tone_value))
                else:
                    tone_seq.append(5)
            tone_all.append(tone_seq)

            words=pseg.cut(str_lyric) 
            pos_seq=[]
            for word, flag in words:
                for jj in range(len(word)):
                    pos_seq.append(flag)
            pos_all.append(pos_seq)

        savepath=line.replace('REMIaligned_events','Pos_Tone')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        pickle.dump((pos_all,tone_all), open(savepath, 'wb'))





    




       
