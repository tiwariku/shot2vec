'''
This module contians functions and classes directly related to trainign and
predicting shot2vec RNN models. Data processing is handled separately
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import random


class KerasBatchGenerator(object):
    '''
    generates batches for Keras to train neural networks
    should I grab the batches randomly?
    '''
    def __init__(self, data, num_steps, batch_size, vocabulary, pad_id,
                 seed=0):
        '''
        in:
            data: a list of games in id format
            num_steps: the max length of a game (will pad)
            batch_size: the number of games in a batch
            vocabulary: the number of distinct words in the vocabulary
            pad_id: the id to use after the game has ended
            seed: optional, seed for the sampler, default = 0
        '''
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.pad_id = pad_id
        self.seed = seed
        self.rand = random
        self.rand.seed(seed)

    def generate(self):
        '''
        generator that yields training data for the keras rnn. uses
        random.sample which will lead to double counting of games eventually
        '''
        #input is just the number of steps in each in, and the batch size
        xdata = np.zeros((self.batch_size, self.num_steps)) + self.pad_id
        #output will be one-hots of dimension vocabulary
        ydata = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        ydata[:, :, self.pad_id] = 1
        while True:#never terminate
            for i, game in enumerate(self.rand.sample(self.data,
                                                      self.batch_size)):
                t_len_game = len(game)
                xdata[i, :t_len_game] = game
                temp_y = game[1:]
                ydata[i, :t_len_game-1, :] = to_categorical(temp_y,
                                                            num_classes=self.vocabulary)
            yield xdata, ydata

    def reset_sampler(self):
        '''
        resets the sampler to self.seed
        '''
        self.rand = random
        self.rand.seed(self.seed)

def make_LSTM_RNN(vocabulary, hidden_size, num_steps, use_dropout=True):
    model = Sequential()
    model.add(layers.Embedding(vocabulary,
                               hidden_size,
                               input_length=num_steps))
    model.add(layers.LSTM(hidden_size, return_sequences=True))
    model.add(layers.LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.TimeDistributed(layers.Dense(vocabulary,
                                                  activation='softmax')))
    return model

def make_prediction_model_file(weights_file, vocabulary, hidden_size):
    model_predicting = make_LSTM_RNN(vocabulary, hidden_size, None)
    model_predicting.load_weights(weights_file)
    return model_predicting

def next_probs(seed_list, model_predictining):
    '''
    seed_list is the game so far in event format
    '''
    model_predictining.reset_states()
    for seed in seed_list[:-1]:
        model_predictining.predict([seed,], verbose=0)
    probs_vector = model_predictining.predict([seed_list[-1],],
                                          verbose=1)[0][0]
    #probs = {}
    #for i, prob in enumerate(probs_vector):
    #    probs[id_2_event[i]]=prob
    return probs_vector
