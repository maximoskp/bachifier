import unidecode
import numpy as np
import pickle

path_to_file = 'data/wtc1.txt'

text = unidecode.unidecode(open(path_to_file).read())

# unique contains all the unique characters in the file
unique = sorted(set(text))
print('length of overall string: ', len(text))

# creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(unique)}
idx2char = {i:u for i, u in enumerate(unique)}
print('number of unique: ', len(unique))

# setting the maximum length sentence we want for a single input in characters
max_length = 300

# length of the vocabulary in chars
vocab_size = len(unique)

# indexes
input_text = []
# binary
target_text = []

for f in range(0, len(text)-max_length-1, max_length//3):
    inps = text[f:f+max_length]
    targ = text[f+1:f+max_length+1]

    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])

def decode_text(text, dictionary):
    s = ''
    for t in text:
        s += dictionary[t]
    return s
# end decode_text

print('input shape: ', np.array(input_text).shape)
print('target shape: ', np.array(target_text).shape)

x = np.zeros((len(input_text), max_length))
y = np.zeros((len(target_text), max_length, vocab_size))
for i in range(len(input_text)):
    sentence = input_text[i]
    if len(sentence) == max_length:
        for t in range(len(sentence)):
            x[i, t] = sentence[t]
            y[i, t, target_text[i]] = 1

dataset = {
    'char2idx': char2idx,
    'idx2char': idx2char,
    'max_length': max_length,
    'vocab_size': vocab_size,
    'x': x,
    'y': y
}

with open('data/dataset.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)