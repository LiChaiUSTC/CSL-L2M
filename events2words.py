import os
import pickle
import numpy as np
import glob

if __name__ == '__main__':
    path_dictionary = os.path.join('data', 'dictionary_lyric.pkl')
    path_dictionary2 = os.path.join('data', 'dictionary_melody.pkl')
    lines=glob.glob('data/REMIaligned_events/'+'*.pkl')
    dic_events = []
    dic_lyrics = []
    for line in lines:
        all_lyrics,pos_seq,events=pickle.load(open(line, 'rb'))
        for event in events:
            dic_events.append('{}_{}'.format(event['name'], event['value']))

        for lyric_list in all_lyrics:
            for lyric in lyric_list:
                dic_lyrics.append(lyric)


    unique_lyrics = sorted(set(dic_lyrics), key=lambda x: (not isinstance(x, int), x))
    lyric2word = {key: i for i, key in enumerate(unique_lyrics)}
    word2lyric = {i: key for i, key in enumerate(unique_lyrics)}
    print(' > num classes:', len(word2lyric))
    pickle.dump((lyric2word, word2lyric), open(path_dictionary, 'wb'))

    unique_events = sorted(set(dic_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    print(' > num classes:', len(word2event))
    pickle.dump((event2word, word2event), open(path_dictionary2, 'wb'))
