import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import unidecode
import copy

with open('data/dataset_words.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

vocab_size = dataset['vocab_size']
max_length = dataset['max_length']
word2idx = dataset['word2idx']
idx2word = dataset['idx2word']
# print('word2idx: ', word2idx)
# print('idx2word: ', idx2word)

path_to_file = 'data/wtc1.txt'
text = unidecode.unidecode(open(path_to_file).read())
words = text.split('_')

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
        print('sampling round: ', i, '===================================================')
        # get a random index for what to change within the acceptable range
        idx = 1 + np.random.randint( len(p)-1 )
        i1 = max( 0, idx-max_length )
        i2 = max(i1+max_length, idx)
        x_in = [word2idx[j] for j in p[i1:i2]]
        preds = model.predict( np.expand_dims( x_in, axis=0 ) )[0]
        next_index = sample(preds[i2-idx], temperature)
        next_word = idx2word[next_index]
        print('idx: ', idx)
        print('i1: ', i1)
        print('i2: ', i2)
        print('x_in: ', x_in)
        print('x_dec: ', decode_text(x_in, idx2word))
        print('i2-idx: ', i2-idx)
        print('preds[i2-idx]: ', preds[i2-idx])
        print('next_index: ', next_index)
        print('next_word: ', next_word)
        print('substitutes: ', p[idx])
        print('before: ',  '_'.join(p))
        # p = p[:idx] + next_word + p[idx+1:]
        p[idx] = next_word
        print('after : ',  '_'.join(p))
    # end for
    return p
# end recompose_part

def decode_text(text, dictionary):
    s = ''
    for t in text:
        s += dictionary[t]
    return s
# end decode_text

# p_in = text[45000:46500]
# p_in = text[0:1500]
p_in = words[0:200]
# p_in = '_'.join( p_in.split('_')[1:-1] ).replace('\n', '')
print(p_in)
# print( 'FINDING:', p_in.rfind('') )

p2destroy = copy.deepcopy(p_in)
p_out = recompose_part(p2destroy, sampling_rounds=500, temperature=0.3)
print('p_in: ', '_'.join(p_in))
print('p_out: ', '_'.join(p_out))

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

os.makedirs( 'MIDIs', exist_ok=True )

filename = "MIDIs/generated_p_in.mid"
midi_from_NOD_string('_'.join(p_in), fileName=filename)

filename = "MIDIs/generated_p_out.mid"
midi_from_NOD_string('_'.join(p_out), fileName=filename)