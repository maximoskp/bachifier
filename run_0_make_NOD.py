from functools import total_ordering
import os
import music21 as m21
import symbolic_data_processing as sdp
import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

folder = 'data/WTC_I/'
files = os.listdir( folder )

wtc1_structs = []

for f in files:
    print('trying ', f)
    if f.endswith('.mxl'):
        print('processing...')
        wtc1_structs.append( sdp.SymbolicInfo(folder + f, metadatafile=folder+'metadata.csv' ) )

# make a single string
wtc1_string = ''
for s in wtc1_structs:
    wtc1_string += s.nod_12t_string

text_file = open("data/wtc1.txt", "w")
text_file.write(wtc1_string)
text_file.close()

# '''
# for f in files:
#     if f.endswith('.mxl')
#     s = m21.converter.parse( folder + f )
#     f = s.flat.notes
#     for n in f:
#         print(  )
# '''

# f = files[0]
# print(f)
# s = m21.converter.parse( folder + f )
# f = s.flat.notes
# prev_offset = 0
# s = ''
# for n in f.notes:
#     # print(prev_offset)
#     s += '_(' + str(n.pitch.midi) + ',' + str(n.offset - prev_offset) + ',' + str(n.duration.quarterLength) + ')'
#     prev_offset = n.offset

# # f.show('t')

# def stream_from_NOD_string(s):
#     usplit = s.split('_')
#     s = m21.stream.Score()
#     tm = m21.tempo.MetronomeMark(number=80)
#     s.insert(0, tm)
#     total_offset = 0
#     for u in usplit:
#         csplit = u[1:-1].split(',')
#         if len( csplit ) == 3:
#             n = m21.note.Note(int(csplit[0]))
#             d = m21.duration.Duration(float(csplit[2]))
#             n.duration = d
#             # n.show('t')
#             total_offset += float( csplit[1] )
#             s.insert( total_offset , n )
#     return s
# # end stream_from_NOD_string

# def generate_midi(sc, fileName="test_midi.mid"):
#     # we might want the take the name from uData information, e.g. the uData.input_id, which might preserve a unique key for identifying which file should be sent are response to which user
#     mf = m21.midi.translate.streamToMidiFile(sc)
#     mf.open(fileName, 'wb')
#     mf.write()
#     mf.close()
# # end generate_midi

# m21s = stream_from_NOD_string(s)
# generate_midi(m21s)
