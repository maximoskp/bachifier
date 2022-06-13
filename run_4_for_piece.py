import os
import music21 as m21
import symbolic_data_processing as sdp
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import generation_functions as gf

folder = 'MIDIs/'
# files = os.listdir( folder )
file = '2826493_2.mid'

p = sdp.SymbolicInfo(folder + file, make12t=False )

s = p.nod_string

print(s)

gf.recompose_midi_from_NOD_string(s,fileName="MIDIs/test_midi.mid",sampling_rounds=1000)