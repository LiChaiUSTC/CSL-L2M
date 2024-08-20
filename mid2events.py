import os
import pickle
import numpy as np
import glob
from miditoolkit.midi import parser as mid_parser  
import sys
#from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
# config
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 16

out_dir = sys.argv[1] #data/REMIaligned_events

# define event
def create_event(name, value):
    event = dict()
    event['name'] = name
    event['value'] = value
    return event

def time_to_pos(t):
    return round(t / TICK_RESOL)

if __name__ == '__main__':
    len_list = []
    lines=glob.glob('data/Midi/'+'*.mid')
    for line in lines:
        midi_obj = mid_parser.MidiFile(line,charset='utf-8')
        notes_start_pos = [time_to_pos(j.start) for i in midi_obj.instruments for j in i.notes]
        max_pos = min(max(notes_start_pos) + 1, 2 ** 16)
        pos_to_info = [[None for _ in range(2)] for _ in range(max_pos)]
        assert len(midi_obj.instruments)==1

        cnt = 0
        bar = 0
        measure_length = 64 #BAR_RESOL//TICK_RESOL
        bar_to_pos = [0]
        for j in range(len(pos_to_info)):
            pos_to_info[j][0] = bar
            pos_to_info[j][1] = cnt
            cnt += 1
            if cnt >= measure_length:
                assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(j)
                cnt -= measure_length
                bar += 1
                bar_to_pos.append(bar_to_pos[-1] + measure_length)

        melody_events = [] 
        lyrics_list=[]
        seq_lyrics=[]
        current_bar=None
        jjj=0
        lyrics=midi_obj.lyrics[0].text
        assert len(midi_obj.instruments[0].notes)==len(midi_obj.lyrics)
        melody_events.append(create_event('SEQ', None))
        for note in midi_obj.instruments[0].notes:
            assert note.velocity==126
            lyric=midi_obj.lyrics[jjj].text
            info = pos_to_info[time_to_pos(note.start)]
            if current_bar!=info[0]:
                 melody_events.append(create_event('Bar', None))
            melody_events.append(create_event('Beat', info[1]))
            melody_events.append(create_event('Note_Pitch', note.pitch))
            melody_events.append(create_event('Note_Duration', round((note.end-note.start)/TICK_RESOL)*TICK_RESOL))
            current_bar=info[0]

            if jjj!=len(midi_obj.instruments[0].notes)-1:
                if '*' not in lyric and '*' not in midi_obj.lyrics[jjj+1].text:
                    melody_events.append(create_event('ALIGN', None))
                elif '*' in lyric and '*' not in midi_obj.lyrics[jjj+1].text:
                    melody_events.append(create_event('ALIGN', None))
            else:
                melody_events.append(create_event('ALIGN', None))

            if '.' in lyric and jjj!=len(midi_obj.instruments[0].notes)-1:
                melody_events.append(create_event('SEQ', None))

            if '*' not in lyric:
                if '.' not in lyric:
                    lyrics_list.append(lyric)
                else:
                    lyrics_list.append(lyric.replace('.',''))
                    seq_lyrics.append(lyrics_list)
                    lyrics_list=[]
            else:
                if '.' in lyric:
                    seq_lyrics.append(lyrics_list)
                    lyrics_list=[]

            jjj=jjj+1
        
        melody_events.append(create_event('Bar', None))
        melody_events.append(create_event('SEQ', None))   
        melody_events.append(create_event('EOS', None))

        pos_seq=[]
        for ii in range(len(melody_events)):
            if melody_events[ii]['name']=='SEQ':
                pos_seq.append(ii)

        fn = os.path.basename(line)
        savepath=os.path.join(out_dir, fn.replace('.mid','.pkl'))
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        pickle.dump((seq_lyrics,pos_seq,melody_events), open(savepath, 'wb'))
        
        


