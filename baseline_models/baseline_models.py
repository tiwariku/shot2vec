'''
This is a baseline model to predict future plays based on the assumption that
plays are sampled at random from a probability distribution. The distribution
is determined by performing a 'fit' to find the percent of total plays occupied
by each type of play.

@tiwariku
2019-07-23
'''

from collections import defaultdict
import numpy as np
from sklearn import base
import data_processing as dp

class UncorrelatedEstimator(base.BaseEstimator, base.RegressorMixin):
    '''
    This sklean estimator predicts the probability of the next_play using iid
    assumption with play probabilities given by their frequency of occurence in
    the corpus
    '''
    def __init__(self):
        '''
        initialize instance with unfilled default dict for the idd distribution
        '''
        self.probs = defaultdict(float)
        self.keys = []
        self.probs = []

    def fit(self, Xdata, ydata):
        '''
        in: Xdata, a list of plays
            ydata, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution
        '''
        count = defaultdict(int)
        total = 0
        for play in Xdata:
            count[play] += 1
            total += 1
        tempdict = {}

        for play in count:
            tempdict[play] = count[play]/total

        self.keys, self.probs = [], []
        for key, prob in tempdict.items():
            self.keys.append(key)
            self.probs.append(prob)

        return self

    def predict(self, Xdata):
        '''
        in: Xdata, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions
        sample from the IDI
        '''
        return [np.random.choice(self.keys, p=self.probs) for x in Xdata]

class MarkovEstimator(base.BaseEstimator, base.RegressorMixin):
    '''
    This sklean estimator predicts the probability of the next_play using by
    sampling from the distribution of plays that have historically followed
    this type of play (Markovian in current play)
    '''
    def __init__(self):
        '''
        initialize instance with unfilled default dict for the idd distribution
        '''
        self.probs = defaultdict(float)
        self.keys = []
        self.probs = []

    def fit(self, Xdata, ydata):
        '''
        in: Xdata, a list of plays
            ydata, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution
        '''
        count = defaultdict(int)
        total = 0
        for play in Xdata:
            count[play] += 1
            total += 1
        tempdict = {}

        for play in count:
            tempdict[play] = count[play]/total

        self.keys, self.probs = [], []
        for key, prob in tempdict.items():
            self.keys.append(key)
            self.probs.append(prob)

        return self

    def predict(self, Xdata):
        '''
        in: Xdata, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions
        sample from the IDI
        '''
        return [np.random.choice(self.keys, p=self.probs) for x in Xdata]

if __name__ == '__main__':
    UE = MarkovEstimator()
    STOP = 100000
    CORPUS = dp.flatten_games(dp.unpickle_it('../assets/corpi/full_coords_bin_10'))
    CORPUS = CORPUS[:STOP]
    print(f'length of corpus: {len(CORPUS)}')
    X = CORPUS[:-1]
    Y = CORPUS[1:]
    UE.fit(X, Y)
    print('Fit complete')
    Y_PRED = UE.predict(X)
    print('Predictions complete')
    VAL_ACC = sum(np.array(Y_PRED) == np.array(Y))/len(Y)
    print(f'Uncorrelated validation accuracy: {VAL_ACC}')
