import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os

with open('data/dataset.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

x = dataset['x']
y = dataset['y']
vocab_size = dataset['vocab_size']
max_length = dataset['max_length']

rnd_idxs = np.random.permutation( x.shape[0] )
tr = 2*x.shape[0]//30
v = x.shape[0]//60
te = x.shape[0]//60

print('x.shape: ', x.shape)
print('y.shape: ', y.shape)

x_train = x[ rnd_idxs[:tr] ,:]
y_train = y[ rnd_idxs[:tr] ,:]
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

x_valid = x[ rnd_idxs[tr:tr+v] ,:]
y_valid = y[ rnd_idxs[tr:tr+v] ,:]
print('x_valid.shape: ', x_valid.shape)
print('y_valid.shape: ', y_valid.shape)

x_test = x[ rnd_idxs[tr+v:tr+v+te] ,:]
y_test = x[ rnd_idxs[tr+v:tr+v+te] ,:]
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

batch_size = 64

input_layer = keras.layers.Input(shape=[max_length])

embedding = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=128,
    batch_input_shape=[batch_size, None]
)

lstm1 = keras.layers.LSTM(
    units=512, # LSTM units in this layer
    # input_shape=[max_length, vocab_size],
    return_sequences=True, # for input to the next LSTM
    # return_state=True,
    recurrent_initializer='glorot_uniform'
)

lstm2 = keras.layers.Bidirectional( keras.layers.LSTM(
    units=512, # LSTM units in this layer
    # input_shape=[max_length, vocab_size],
    return_sequences=True,
    # return_state=True,
    # recurrent_activation='sigmoid',
) )

d3 = keras.layers.TimeDistributed( keras.layers.Dense( 256 , activation='selu', input_shape=[vocab_size] ) )
d4 = keras.layers.TimeDistributed( keras.layers.Dense( 128 , activation='selu', input_shape=[vocab_size] ) )

output_layer = keras.layers.Dense(vocab_size, activation='sigmoid' )

z = embedding(input_layer)
z = lstm1(z)
z = lstm2(z)
z = d3(z)
z = d4(z)
model_out = output_layer(z)

model = keras.Model(inputs=[input_layer], outputs=[model_out])

model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='binary_accuracy')
# model.compile(loss='mean_squared_error', optimizer='adam', metrics='cosine_similarity')

# checkpoint ====================================================================
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
model_name = 'bachifier'
os.makedirs( 'models', exist_ok=True )
filepath = 'models/'+model_name+'_epoch{epoch:02d}_valLoss{val_loss:.6f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
filepath_current_best = 'models/'+model_name+'_current_best.hdf5'
checkpoint_current_best = ModelCheckpoint(filepath=filepath_current_best,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='min')
if os.path.exists('/models/'+model_name+'_logger.csv'):
    os.remove('/models/'+model_name+'_logger.csv')
csv_logger = CSVLogger('models/'+model_name+'_logger.csv', append=True, separator=';')
# checkpoint ====================================================================

# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_valid,y_valid), verbose=1)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1000, validation_data=(x_valid,y_valid), callbacks=[checkpoint, checkpoint_current_best, csv_logger], verbose=1)