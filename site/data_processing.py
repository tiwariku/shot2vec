import numpy as np
import json

def game_json_to_event_dicts(game_json):
    '''
    in: game_json, a string with the json for the game
    out: a list dictionaries each representing an play
    '''
    d = game_json#json.loads(game_json) 
    if 'liveData' in d.keys():
        return [play for play in d['liveData']['plays']['allPlays']]
    else:
        return []


def strip_event(play):
    '''
        in: full json for a play from nhl api
        out: bare bones dictionary with some features
    '''
    def getcoords(play_coordinates, sbin = 10):
        if not ('x' in play_coordinates.keys() 
                or 'y' in play_coordinates.keys()):
            return None
        x = None
        y = None
        if 'x' in play_coordinates.keys():
            x = int(play_coordinates['x']/sbin)*sbin
        if 'y' in play_coordinates.keys():
            y = int(play_coordinates['y']/sbin)*sbin
        if x or y:
            return (x, y)
        else:
            return None
    event = {}
    event['Type'] = ''.join(play['result']['event'].split())
    event['Coords'] = getcoords(play['coordinates'])
    event['periodTime'] = (int(play['about']['periodTime'][0:2])*60+
                           int(play['about']['periodTime'][3:]))
    return event

def strip_game_events(plays):
    '''
    in game, list of plays in nhl api json format
    out game, list of events in stripped dictionary format
    '''
    events = [strip_event(play) for play in plays]
    make_delta_time(events)
    return events

def make_delta_time(game_events):
    '''
    in: a game's worth of plays
    out: modified list with feature expansion for time since last play
    '''
    def bin_DT(deltaTime):
        if deltaTime == 0:
            return -1
        else:
            return int(np.log2(deltaTime))
    for i, play in enumerate(game_events):
        if play['periodTime'] == 0:
            play['DeltaTime'] = -1
        else:
            play['DeltaTime'] = bin_DT(play['periodTime']-
                                       game_events[i-1]['periodTime'])
def make_event_string(event):
    '''
    in: stripped event
    out: string
    '''
    coordstring = 'nocoords'
    if event['Coords']:
        coordstring = '{}_{}'.format(event['Coords'][0],
                                     event['Coords'][1])
    return '{}_{}_{}'.format(event['Type'],
                                coordstring,
                                event['DeltaTime'])
