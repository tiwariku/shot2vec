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
        self.xtrain = []

    def fit(self, xdata, ydata):
        '''
        in: xdata, a list of plays
            ydata, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution
        '''
        self.xtrain = xdata
        tempdict = Counter(ydata)
        total = sum(tempdict.values())
        self.keys = list(tempdict.keys())
        self.probs = np.array(list(tempdict.values()))/total
        return self

    def predict(self, xdata):
        '''
        in: xdata, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions
        sample from the IDI
        '''
        return [np.random.choice(self.keys, p=self.probs) for x in xdata]

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

    def fit(self, xdata, ydata):
        '''
        in: xdata, a list of plays
            ydata, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution
        '''
        play_next_plays = defaultdict(list)
        for i, play in enumerate(xdata):
            play_next_plays[play].append(ydata[i])

        for play, next_plays in play_next_plays.items():
            temp_counter = Counter(next_plays)
            num = sum(temp_counter.values())
            self.keys_dict[play] = list(temp_counter.keys())
            self.probs_dict[play] = np.array(list(temp_counter.values()))/num
        return self

    def predict(self, xdata):
        '''
        in: xdata, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions
        sample from the IDI
        '''
        ypred = []#np.zeros(len(xdata))
        for play in xdata:
            if play in self.keys_dict.keys():
                ypred.append(np.random.choice(self.keys_dict[play],
                                              p=self.probs_dict[play]))
            else:
                ypred.append('UNKNOWN PLAY')
        return ypred

def test_it(num_games, corpus_filename, estimator):
    '''
    in: num_games, number of games from the corpus to use in the whole
                   dataset
        corpus_filename, path (excluding suffix) to pkl file of the corpus
        estimator, the estimator class to use
    '''
    print(f'Corpus: {corpus_filename}\nEstimator: {estimator}')
    corpus = dp.unpickle_it(CORPUS_FILENAME)[:num_games]
    train_data, test_data = dp.train_test_split(corpus)
    print(f'\tTraining on {len(train_data)} games')
    print(f'\tTesting on {len(test_data)} games')
    train_data = dp.flatten_games(train_data)
    test_data = dp.flatten_games(test_data)
    x_train = train_data[:-1]
    y_train = train_data[1:]
    x_test = test_data[:-1]
    y_test = test_data[1:]
    estimator.fit(x_train, y_train)
    print('\tFit complete')
    y_pred = estimator.predict(x_test)
    print('\tPredictions complete')
    val_acc = sum(np.array(y_pred) == np.array(y_test))/len(y_test)
    print(f'\tValidation accuracy: {val_acc}\n\n')


if __name__ == '__main__':
    ESTIMATORS = [UncorrelatedEstimator(), MarkovEstimator()]
    STOP = 3000
    CORPUS_FILENAME = '../assets/corpi/corpus_zone'
    for ESTIMATOR in ESTIMATORS:
        test_it(STOP, CORPUS_FILENAME, ESTIMATOR)
