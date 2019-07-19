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
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.current_idx = 0
        self.skip_step = skip_step
        
    def generate(self):
        #input is just the number of steps in each in, and the batch size
        x = np.zeros((self.batch_size, 
                      self.num_steps))
        #output will be one-hots of dimension vocabulary
        y = np.zeros((self.batch_size, 
                      self.num_steps, 
                      self.vocabulary))
        while True:#never terminate
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
