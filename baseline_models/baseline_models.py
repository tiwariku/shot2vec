'''
This is a baseline model to predict future plays based on the assumption that
plays are sampled at random from a probability distribution. The distribution
is determined by performing a 'fit' to find the percent of total plays occupied
by each type of play.

@tiwariku
2019-07-23
'''

from sklearn import base
import numpy as np
from collections import defaultdict
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
    
    def fit(self, X, y):
        '''
        in: X, a list of plays
            y, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution 
        '''
        count = defaultdict(int)
        total = 0
        for play in X:
            count[play] += 1
            total += 1
        for play in count:
            self.probs[play] = count[play]/total
        return self
    
    def predict(self, X):
        '''
        in: X, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions 
        sample from the IDI
        '''
        keys, probs  = [], []
        for key, prob in self.probs.items():
            keys.append(key)
            probs.append(prob)
        return [np.random.choice(keys, p=probs) for x in X]

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
    
    def fit(self, X, y):
        '''
        in: X, a list of plays
            y, another list of plays, unused in this mode
        out: self, for whatever reason
        build the iid distribution 
        '''
        count = defaultdict(int)
        total = 0
        for play in X:
            count[play] += 1
            total += 1
        for play in count:
            self.probs[play] = count[play]/total
        return self
    
    def predict(self, X):
        '''
        in: X, a list of plays (for which the next lay should be predicted)
        out: y, a list of iid predictions 
        sample from the IDI
        '''
        keys, probs  = [], []
        for key, prob in self.probs.items():
            keys.append(key)
            probs.append(prob)
        return [np.random.choice(keys, p=probs) for x in X]

if __name__ == '__main__':
    corpus = dp.flatten_games(dp.unpickle_it('../../assets/corpi/coarse_corpus'))
    print(f'length of corpus: {len(corpus)}')
    
    X = corpus[:-1] 
    y = corpus[1:]
    UE = UncorrelatedEstimator()
    UE.fit(X, y)
    correct_est = [y[i]==y_esti for i, y_esti in enumerate(UE.predict(X))]
    print(f'Uncorrelated validation accuracy: {sum(correct_est)/len(correct_est)}')
    

