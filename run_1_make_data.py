import unidecode
import numpy as np
import pickle

path_to_file = 'data/wtc1.txt'

text = unidecode.unidecode(open(path_to_file).read())
words = text.split('_')

# unique contains all the unique characters in the file
unique = sorted(set(words))
print('length of overall string: ', len(words))

# creating a mapping from unique characters to indices
word2idx = {u:i for i, u in enumerate(unique)}
idx2word = {i:u for i, u in enumerate(unique)}
print('number of unique: ', len(unique))

# setting the maximum length sentence we want for a single input in characters
max_length = 50

# length of the vocabulary in chars
vocab_size = len(unique)

# indexes
input_text = []
# binary
target_text = []

for f in range(0, len(words)-max_length-1, max_length//2):
    inps = words[f:f+max_length]
    targ = words[f+1:f+max_length+1]

    input_text.append([word2idx[i] for i in inps])
    target_text.append([word2idx[t] for t in targ])

def decode_text(text, dictionary):
    s = ''
    for t in text:
        s += dictionary[t]
    return s
# end decode_text

print('input shape: ', np.array(input_text).shape)
print('target shape: ', np.array(target_text).shape)

x = np.zeros((len(input_text), max_length)).astype(np.float32)
y = np.zeros((len(target_text), max_length, vocab_size)).astype(np.bool)
for i in range(len(input_text)):
    sentence = input_text[i]
    if len(sentence) == max_length:
        for t in range(len(sentence)):
            x[i, t] = sentence[t]
            y[i, t, target_text[i]] = 1

dataset = {
    'word2idx': word2idx,
    'idx2word': idx2word,
    'max_length': max_length,
    'vocab_size': vocab_size,
    'x': x,
    'y': y
}

with open('data/dataset_words.pickle', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)