import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle

with open('data/dataset.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

vocab_size = dataset['vocab_size']
max_length = dataset['max_length']
char2idx = dataset['char2idx']
idx2char = dataset['idx2char']

# load model
model_name = 'bachifier'
model = keras.models.load_model( 'models/'+model_name+'_current_best.hdf5' )

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# end sample

def recompose_part(p, sampling_rounds=100, temperature=1.0):
    for i in range(sampling_rounds):
        print('sampling round: ', i)
        # get a random index for what to change within the acceptable range
        idx = 1 + np.random.randint( len(p)-1 )
        # make sure a number is changed
        while p[idx] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            idx = 1 + np.random.randint( len(p)-1 )
        # get input indexes
        i1 = max( 0, idx-max_length-1 )
        i2 = max(i1+max_length, idx-1)
        x_in = [char2idx[j] for j in p[i1:i2]]
        preds = model.predict( np.expand_dims( x_in, axis=0 ) )[0]
        next_index = sample(preds[-1], temperature)
        next_char = idx2char[next_index]
        while next_char not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            next_index = sample(preds[i2-idx], temperature)
            next_char = idx2char[next_index]
        p = p[:idx] + next_char + p[idx+1:]
    # end for
    return p
# end recompose_part

def decode_text(text, dictionary):
    s = ''
    for t in text:
        s += dictionary[t]
    return s
# end decode_text

import music21 as m21
def stream_from_NOD_string(s):
    usplit = s.split('_')
    s = m21.stream.Score()
    tm = m21.tempo.MetronomeMark(number=80)
    s.insert(0, tm)
    total_offset = 0
    for u in usplit:
        csplit = u[1:-1].split(',')
        if len( csplit ) == 3:
            n = m21.note.Note(int(csplit[0]))
            d = m21.duration.Duration(float(csplit[2]))
            n.duration = d
            # n.show('t')
            total_offset += float( csplit[1] )
            s.insert( total_offset , n )
    return s
# end stream_from_NOD_string

def generate_midi(sc, fileName="test_midi.mid"):
    # we might want the take the name from uData information, e.g. the uData.input_id, which might preserve a unique key for identifying which file should be sent are response to which user
    mf = m21.midi.translate.streamToMidiFile(sc)
    mf.open(fileName, 'wb')
    mf.write()
    mf.close()
# end generate_midi

def midi_from_NOD_string(s, fileName="test_midi.mid"):
    m21s = stream_from_NOD_string(s)
    generate_midi(m21s, fileName=fileName)
# end midi_from_NOD_string

def recompose_midi_from_NOD_string(s, fileName="test_midi.mid", sampling_rounds=100, temperature=1.0):
    s1 = recompose_part(s, sampling_rounds=sampling_rounds, temperature=temperature)
    midi_from_NOD_string(s1, fileName=fileName)
# end recompose_midi_from_NOD_string