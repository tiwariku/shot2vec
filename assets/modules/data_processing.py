'''
This module handles manipulating the queried data from the NHL api into
the form accepted by the shot2vec models
@tiwariku
2019-07-22
'''
import pickle
import in_out as io

def pickle_it(name, obj):
    '''
    in: name, the indented filename, without the pkl suffix
        obj, the object to pickle
    pickles the file with context handled properly for writing
    '''
    with open(f'{name}.pkl', 'wb') as pkfile:
        pickle.dump(obj, pkfile, pickle.HIGHEST_PROTOCOL)

def unpickle_it(name):
    '''
    in: name, the file to unpickle, without the pkl suffix
    out: the loaded file
    '''
    with open(f'{name}.pkl', 'rb') as pkfile:
        return pickle.load(pkfile)

def game_to_plays(game, strip_fn=lambda x: x):
    '''
    in: game, a game_json as returned by the NHL api
        strip_fn (optional), a function to strip the event dictionaries
    out:
        game_plays: a list of strings representing each play in the game. The
                    strings can be converted back to dictionaries using
                    literal_eval from the ast package
    '''
    game_plays = [str(strip_fn(play)) for play in
                  game['liveData']['plays']['allPlays']]
    return game_plays

def strip_name_only(play):
    '''
    in: an event dictionary in format given by nhl api
    out: just the event type
    '''
    stripped_play = {}
    stripped_play['Type'] = play['result']['event']
    return stripped_play

def get_corpus(start_year=2010, stop_year=2019, strip_fn=strip_name_only):
    '''
    in: start_year, first year to download
        stop_year,  last year to downloads
        strip_fn,   optional, a function that takes a play and returns a dict
                    of a stripped version, default, just the type
    out: corpus,    a list of games, where each 'game' is a list of plays, and
                    hte plays are string represenations of dicts
    >>>perhaps this belongs in the model module
    >>>downloads (or reads cache) all game data form start_year to stop_year
       converts games to plays
    '''
    games, corpus = [], []
    for year in range(start_year, stop_year+1):
        games.extend(io.get_year_games(year))
    for game in games:
        corpus.append(game_to_plays(game, strip_fn=strip_fn))
    return corpus

def flatten_games(corpus):
    '''
    in: corpus, a list of games
    out: games in one long list
    '''
    return [play for game in corpus for play in game]

def make_vocabulary(corpus, pad_play=None, verbose=False):
    '''
    in: corpus, a list of games
    out: play_to_id, a dictionary for encoding plays
         id_to_play, a dictionary for decoding plays
         vocabulary, int, the number of distinct words
    '''
    corpus = flatten_games(corpus)
    if pad_play:
        corpus.append(pad_play)
    distinct_plays = list(set(corpus))
    vocabulary = len(distinct_plays)
    play_to_id = {}
    for play in distinct_plays:
        play_to_id[play] = distinct_plays.index(play)
        id_to_play = dict(zip(play_to_id.values(), play_to_id.keys()))
    if verbose:
        print(play_to_id)
        print(vocabulary)
    return play_to_id, id_to_play, vocabulary

def train_test_split(games, train_frac=0.8, verbose=False):
    '''
    in: games, a list of games. Each game is a list of events, serialized as
               strings
        train_frac, the fraction of the corpus to use for traing

    out: train, test, two lists of games split using train_frac
    '''
    if verbose:
        print(f'Making train test split. train_frac = {train_frac}')
    split_ind = int(len(games)*train_frac)
    train = games[:split_ind]
    test = games[split_ind:]
    return train, test

def corpus_to_keras(corpus_filename, pad_play=None):
    '''
    in: corpus_filename, the name of the corpus file for training/testing
    out: train_data, list of games for testing
         valid_data, will be data for hyperparameter tuning validation, right
                     now just an list with an empty list in it
         test_data, data for in-epoch/RNN weight update validation
         vocabulary, the number of distinct words
         play_to_id, dictionary to convert plays to ids
         id_to_play, dictionary to convert id to plays, play is a string, which
                     must be further converted to a dict using literal_eval
    '''
    #preprocessing including stripping must be done in game_to_plays, which is
    #in get_corpus
    corpus = unpickle_it(corpus_filename)

    #building word to index dictionary and vocabulary
    play_to_id, id_to_play, vocabulary = make_vocabulary(corpus,
                                                         pad_play=pad_play,
                                                         verbose=0)

    #convert to ids
    corpus_id = [[play_to_id[play] for play in game] for game in corpus]

    #train test split
    train_data, test_data = train_test_split(corpus_id, verbose=False)

    #flatten training (testing) data to list of events
    #train_data = flatten_games_to_events(train_data)
    #test_data = flatten_games_to_events(test_data)
    valid_data = [[]]

    return (train_data,
            valid_data,
            test_data,
            vocabulary,
            play_to_id,
            id_to_play
           )
