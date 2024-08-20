import os, pickle, random, copy
import numpy as np

import miditoolkit

##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 64 


##############################
# containers for conversion
##############################
class ConversionEvent(object):
  def __init__(self, event, is_full_event=False):
    if not is_full_event:
      if 'Note' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Chord' in event:
        self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
      else:
        self.name, self.value = event.split('_')
    else:
      self.name, self.value = event['name'], event['value']
  def __repr__(self):
    return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
  def __init__(self, pitch, bar, position, duration):
    self.pitch = pitch
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
    self.duration = duration

def REMIaligned2midi(lyrics, events, tempo, output_midi_path=None, is_full_event=False):
  events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
  #assert events[0].name == 'SEQ'
  temp_notes = []

  cur_bar = 0
  cur_position = 0
  pp=0
  midi_obj = miditoolkit.midi.parser.MidiFile()
  midi_obj.lyrics=[]
  num_notes=0
  all_numnotes=0
  for i in range(len(events)):
    if 'Note_Pitch' in events[i].name:
      all_numnotes=all_numnotes+1
  #assert all_numnotes>=len(lyrics)

  for i in range(len(events)):
    if events[i].name == 'Bar':
      if i > 1:
        cur_bar += 1
    elif events[i].name == 'Beat':
      cur_position = int(events[i].value)
      assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
    elif 'Note_Pitch' in events[i].name and (i+1) < len(events) and 'Note_Duration' in events[i+1].name:
      temp_notes.append(
        NoteEvent(
          pitch=int(events[i].value), 
          bar=cur_bar, position=cur_position, 
          duration=int(events[i+1].value)
        )
      )
      num_notes=num_notes+1
    elif events[i].name == 'ALIGN':
      if num_notes==1:
        midi_obj.lyrics.append(miditoolkit.Lyric(text=lyrics[pp], time=temp_notes[-1].start_tick))
      elif num_notes>1:
        midi_obj.lyrics.append(miditoolkit.Lyric(text=lyrics[pp], time=temp_notes[-num_notes].start_tick))
        for i in range(num_notes-1):
          midi_obj.lyrics.append(miditoolkit.Lyric(text="*", time=temp_notes[i+1-num_notes].start_tick))
      pp=pp+1
      if pp>len(lyrics)-1:
        pp=len(lyrics)-1
      num_notes=0
    elif events[i].name == 'SEQ' and pp>0:
      midi_obj.lyrics[-1].text += '.'
    elif events[i].name in ['EOS', 'PAD']:
      continue
  
  midi_obj.instruments = [miditoolkit.Instrument(program=0, is_drum=False, name='Piano')]

  for n in temp_notes:
    midi_obj.instruments[0].notes.append(
      miditoolkit.Note(126, n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )

  midi_obj.tempo_changes.append(miditoolkit.TempoChange(tempo, 0))

  if output_midi_path is not None:
    midi_obj.dump(output_midi_path, charset='utf-8')

  return midi_obj
