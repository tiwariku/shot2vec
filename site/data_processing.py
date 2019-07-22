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
        game_plays: a list of dictionaries representing each play in the game
    '''
    game_plays = [strip_fn(play) for play in
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
