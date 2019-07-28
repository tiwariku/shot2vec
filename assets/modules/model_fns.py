'''
This module contians functions and classes directly related to trainign and
predicting shot2vec RNN models. Data processing is handled separately
'''

import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np

class KerasBatchGenerator(object):
    '''
    generates batches for Keras to train neural networks
    should I grab the batches randomly?
    '''
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = [play for game in data
                     for play in game]
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        while True:#never terminate
            #input is just the number of steps in each in, and the batch size
            x = np.zeros((self.batch_size, 
                          self.num_steps))
            #output will be one-hots of dimension vocabulary
            y = np.zeros((self.batch_size, 
                          self.num_steps, 
                          self.vocabulary))
            for i in range(self.batch_size):
                #if I would run over the edge, reset idx
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0
                x[i,:] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps+1]
                #make the one-hots for the y training data
                y[i,:,:] = to_categorical(temp_y, 
                                          num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

class KerasBatchGenerator_rand(object):
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

def make_LSTM_RNN(vocabulary, hidden_size, num_steps, dropout=.5):
    '''
    in: vocabulary, the number of distinct words in the corpus
        hidden_size, size of LSTM layers
        num_steps, size of input, None for arbitrary
        dropout, the dropout parameter
    out: model, the keras LSTM RNN
    '''
    model = Sequential()
    model.add(layers.Embedding(vocabulary,
                               hidden_size,
                               input_length=num_steps))
    model.add(layers.LSTM(hidden_size, return_sequences=True))
    model.add(layers.LSTM(hidden_size, return_sequences=True))
    model.add(layers.Dropout(dropout))
    model.add(layers.TimeDistributed(layers.Dense(vocabulary,
                                                  activation='softmax')))
    return model

def make_prediction_model_file(weights_file, vocabulary, hidden_size):
    '''
    in: weights_file, file with weights from training, include extension
        vocabulary, used to determine size of embedding layers
        hidden_size, size of LSTM layers
    out:
        model_predicting, a shot2vec model that can accept arbitrary length
                          input
    '''
    model_predicting = make_LSTM_RNN(vocabulary, hidden_size, None)
    model_predicting.load_weights(weights_file)
    return model_predicting

def next_probs(seed_list, model_predictining):
    '''
    in:
        seed_list: the game so far in id format
    out:
        vector of probabilities, index corresponds to id
    '''
    model_predictining.reset_states()
    for seed in seed_list[:-1]:
        print(seed)
        model_predictining.predict([seed,], verbose=0)
    probs_vector = model_predictining.predict([seed_list[-1],],
                                              verbose=0)[0][0]
    return probs_vector
