import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import os

with open('data/bach_images.pickle', 'rb') as handle:
    x = pickle.load(handle)

rows = x.shape[1]
columns = x.shape[2]

rnd_idxs = np.random.permutation( x.shape[0] )
tr = 2*x.shape[0]//3
v = x.shape[0]//6
te = x.shape[0]//6

x_train = x[ rnd_idxs[:tr] ,:]
print('x_train.shape: ', x_train.shape)

x_valid = x[ rnd_idxs[tr:tr+v] ,:]
print('x_valid.shape: ', x_valid.shape)

x_test = x[ rnd_idxs[tr+v:tr+v+te] ,:]
print('x_test.shape: ', x_test.shape)

latent_size = 100
encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[rows, columns]),
    keras.layers.Dense(700, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization()
])

latent = keras.models.Sequential([
    keras.layers.Dense(latent_size)
])

decoder = keras.models.Sequential([
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(750, activation='relu'),
    keras.layers.Dense(rows*columns, activation='sigmoid'),
    keras.layers.Reshape([rows,columns])
])

simple_ae = keras.models.Sequential([encoder, latent, decoder])

encoder.summary()
latent.summary()
decoder.summary()
simple_ae.summary()

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

simple_ae.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[rounded_accuracy])

# checkpoint ====================================================================
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
model_name = 'bachifier_img'
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

history = simple_ae.fit(x_train, x_train, validation_data=(x_valid, x_valid), 
                      epochs=1000, batch_size=32, callbacks=[checkpoint, checkpoint_current_best, csv_logger])