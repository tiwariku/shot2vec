'''
This is a baseline model to predict future plays based on the assumption that
plays are sampled at random from a probability distribution. The distribution
is determined by performing a 'fit' to find the percent of total plays occupied
by each type of play.

@tiwariku
2019-07-23
'''

from collections import defaultdict
from collections import Counter
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
        tempdict = Counter(Xdata)
        total = sum(tempdict.values())
        self.keys = list(tempdict.keys())
        self.probs = np.array(list(tempdict.values()))/total
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
        self.keys_dict = {}
        self.probs_dict = {}

    def fit(self, Xdata, ydata):
        '''
        in: Xdata, a list of plays
            ydata, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution
        '''
        play_next_plays = defaultdict(list)
        for i, play in enumerate(Xdata):
            play_next_plays[play].append(ydata[i])

        for play, next_plays in play_next_plays.items():
            temp_counter = Counter(next_plays)
            num = sum(temp_counter.values())
            self.keys_dict[play] = list(temp_counter.keys())
            self.probs_dict[play] = np.array(list(temp_counter.values()))/num
        return self

    def predict(self, Xdata):
        '''
        in: Xdata, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions
        sample from the IDI
        '''
        return np.array([np.random.choice(self.keys_dict[play],
                                          p=self.probs_dict[play])
                         for play in Xdata])

if __name__ == '__main__':
    ESTIMATOR = UncorrelatedEstimator()
    STOP = 400000
    CORPUS_FILENAME = '../assets/corpi/full_coords_bin_10'
    CORPUS = dp.flatten_games(dp.unpickle_it(CORPUS_FILENAME))
    CORPUS = CORPUS[:STOP]
    print(f'Corpus: {CORPUS_FILENAME}\nEstimator: {ESTIMATOR}\nsize: {STOP}')
    X = CORPUS[:-1]
    Y = CORPUS[1:]
    ESTIMATOR.fit(X, Y)
    print('Fit complete')
    Y_PRED = ESTIMATOR.predict(X)
    print('Predictions complete')
    VAL_ACC = sum(np.array(Y_PRED) == np.array(Y))/len(Y)
    print(f'Validation accuracy: {VAL_ACC}')
