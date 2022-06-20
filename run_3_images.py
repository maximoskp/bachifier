import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle
import copy
import matplotlib.pyplot as plt

with open('data/bach_images.pickle', 'rb') as handle:
    x = pickle.load(handle)

rows = x.shape[1]
columns = x.shape[2]

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

model_name = 'bachifier_img'
model = keras.models.load_model( 'models/'+model_name+'_current_best.hdf5', custom_objects={"rounded_accuracy": rounded_accuracy } )

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

os.makedirs('figs', exist_ok=True)

def plot_image(image):
    plt.imshow(image, cmap='binary')
    plt.axis('off')

def columnwise_probability(img):
    new_img = np.zeros( img.shape )
    for j in range( img.shape[1] ):
        if np.sum( img[:,j] )!= 0:
            new_img[:,j] = img[:,j]/np.sum( img[:,j] )
    return new_img

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    probas = columnwise_probability(preds)
    probas = np.squeeze( probas )
    row_max = np.argmax(np.max(probas, axis=1))
    col_max = np.argmax(np.max(probas, axis=0))
    return [row_max, col_max]

def column_reduce(img, row_max, col_max, div=4):
    new_img = copy.deepcopy( img )
    new_img[ 0, :, col_max ] = new_img[ 0, :, col_max ]/div
    new_img[ 0, row_max, col_max ] = 0
    return new_img


def iterative_reconstruction(model, image, iterations=10, filename='test'):
    img = copy.deepcopy( image )
    target_num_notes = np.sum(img)
    fig = plt.figure(figsize=(1, (iterations+1)*1.5))
    plt.subplot(iterations+1,1 , 1)
    plot_image( np.reshape(img, (rows, columns)) )
    for image_index in range(iterations):
        img = model.predict( img )
        # sampled_list = []
        # achieved_num_notes = 0
        # [r, c] = sample( img, temperature=0.1 )
        # while achieved_num_notes <= target_num_notes:
        #     while [r,c] in sampled_list:
        #         [r, c] = sample( img, temperature=0.1 )
        #     sampled_list.append( [r, c] )
        #     img = column_reduce(img, r, c)
        #     achieved_num_notes += 1
        # for s in sampled_list:
        #     img[0, s[0], s[1]] = 1
        plt.subplot(iterations+1, 1, 2 + image_index)
        plot_image( np.reshape(img, (rows, columns)) )
    plt.savefig('figs/' + filename + '.png', dpi=300)

img = x_test[0:1,:,:]
# sparsity = np.sum(img)/img.size
# print('initial sparsity: ', sparsity)
iterative_reconstruction(model, img, iterations=50)