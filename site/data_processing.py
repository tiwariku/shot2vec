'''
This module handles manipulating the queried data from the NHL api into
the form accepted by the shot2vec models
@tiwariku
2019-07-22
'''
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

def flatten_games(corpus):
    '''
    in: corpus, a list of games
    out: games in one long list
    '''
    return [play for game in corpus for play in game]

def make_vocabulary(corpus, pad_play=None):
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
    return play_to_id, id_to_play, vocabulary
