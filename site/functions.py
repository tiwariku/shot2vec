import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import base64
from numpy.random import randint 
from requests_futures.sessions import FuturesSession

import pickle
import re

import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import data_processing as dp

import model_fns as mf


#function
def load_games_list(filename='corpus.txt', verbose = False):
    '''
    in: filename, the corpus
    out: list of list of events in each game
    '''
    if verbose:
        print('Reading corpus')
    with open(filename,'r') as f:
        return [game.split() for game in f.readlines()]

def strip_aZ(games_list):
    '''
    in: list of lists of games
    out: stripped to just ^a-zA-Z
    '''
    print('Stripping games_list of non-[^a-zA-Z] characters')
    regex = re.compile('[^a-zA-Z]')
    games_list = [[regex.sub('',event) for event in game] for game in games_list]
    #print(games_list[:3][:5])
    return games_list

def make_vocabulary(games_list, verbose=False):
    '''
    in: list of lists of games
    out:
        vocabulary, the number of distinct events
            and
        event_2_ind, a lookup dictionary for the index of each event
    '''
    if verbose:
        print('Constructing vocabular and event2id dictionary')

    #full events list to make vocabulary and ids
    events = flatten_games_to_events(games_list)
    #select distinct
    distinct_events = list(set(events))
    vocabulary = len(distinct_events)
    #make id dictionary
    event_2_id = {}
    for event in distinct_events:
        event_2_id[event] = distinct_events.index(event)
    id_2_event = dict(zip(event_2_id.values(), event_2_id.keys()))
    return vocabulary, event_2_id, id_2_event

def games_list_to_ids(games_list, event_2_id, verbose=False):
    '''
    in: games_list, list of lists of events in string format
        event_2_id, id dictionary constructed from full vocabulary
    out:
        games_list, list of lists of events in id format
    '''
    if verbose:
        print('Encoding a list of games by event id')
    return [[event_2_id[event] for event in game] for game in games_list]

def id_list_to_event(id_list, id_2_event, verbose=False):
    '''
    in: id_list a list of events in id format
        id_2_event: lookup dictionary
    out:
        event_list: a list of events in event format
    '''
    return [id_2_event[idd] for idd in id_list ]

def train_test_split_games_list(games_list, train_frac = .8, verbose=False):
    if verbose:
        print(f'Making train test split. train_frac = {train_frac}')
    split_ind = int(len(games_list)*train_frac)
    train = games_list[:split_ind]
    test = games_list[split_ind:]
    return train, test

def flatten_games_to_events(games_list):
    return [event for game in games_list
                     for event in game]

def load_data(filename):
    '''
    in: filename, a .txt file whose lines are nhl games where events
    are represented by strings separated by spaces
    '''
    games_list = load_games_list('corpus.txt',
                                 verbose=True)[:200]

    ###This is to make things fast for now
    games_list = strip_aZ(games_list)

    #building word to index dictionary and vocabulary
    vocabulary, event_2_id, id_2_event = make_vocabulary(games_list,
                                                         verbose=True)

    #convert to ids
    games_list = games_list_to_ids(games_list,
                                   event_2_id,
                                   verbose=True)

    #train test split
    train_data, test_data = train_test_split_games_list(games_list,
                                                        verbose=True)
    #flatten training (testing) data to list of events
    train_data = flatten_games_to_events(train_data)
    test_data = flatten_games_to_events(test_data)
    valid_data = None

    reversed_dictionary = None

    return (
            train_data,
            valid_data,
            test_data,
            vocabulary,
            reversed_dictionary,
            event_2_id,
            id_2_event
           )
     
def getcoords(event_json, sbin=10):
    play_coordinates = event_json['coordinates']
    x, y = None, None
    if 'x' in play_coordinates.keys():
        x = int(play_coordinates['x']/sbin)*sbin
    if 'y' in play_coordinates.keys():
        y = int(play_coordinates['y']/sbin)*sbin
    return (x, y)

def play_dict_to_plottable(event_json, opacity=1):
    ev_dict = {'name':'',
               'mode':'markers',
               'opacity':opacity,
               'marker':{'size':12, 'color':'black'}}
    ev_dict['text'] = [event_json['result']['event'],]
    x,y = getcoords(event_json, sbin=1)
    ev_dict['x'] = [x,]
    ev_dict['y'] = [y,]
    return ev_dict

def play_dict_to_string(event_json):
    return event_json['result']['event']

def get_example_data(window=10):
    def to_data_dict(event):
        event = event.split('_')
        ev_dict = {'name':'',
                   'mode':'markers',
                   'opacity':1,
                   'marker':{'size':12, 'color':'black'}}
        ev_dict['x'] = [int(event[1]),]
        ev_dict['y'] = [int(event[2]),]
        ev_dict['text'] = [event[0],]
        return ev_dict


    game_data = get_random_game()[-window:]
    data = [play_dict_to_plottable(ejson, 
        opacity=i/window) for i, ejson in enumerate(game_data)]
   #events = ['Shot_10_20_5', 'Shot_50_-20_2',]
   #data = [to_data_dict(event) for event in events]
    return data

def get_random_game():
    session = FuturesSession(max_workers=1) 
    year = '2013'
    game_type = '02'
    game_num = 5
    game_number = str(game_num).zfill(4)
    game_ID = f'{year}{game_type}{game_number}'
    url = f'https://statsapi.web.nhl.com/api/v1/game/{game_ID}/feed/live'
    future = session.get(url)
    d = future.result().json()
    if 'liveData' in d.keys():
        return [play for play in d['liveData']['plays']['allPlays']]
    else:
        print(d)

def make_rink_fig(n_steps, game_json):
    with open("assets/rink.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    #add the prefix that plotly will want when using the string as source
    encoded_image = "data:image/png;base64," + encoded_string

    #print(n_steps)
    #print(game_json)

    window = 30
    cut = n_steps - window
    plays = dp.game_json_to_event_dicts(game_json)[:n_steps]
    data = [play_dict_to_plottable(play, max(i-cut,0)/window) for i, play in enumerate(plays)]
    #print(data)
    data[-1]['marker']['size']=30

    layout_dict = dict(#title='',
                       showlegend=False,
                       #clickmode='event+select',
                       images=[dict(
                                    source=encoded_image,
                                    xref="x",
                                    yref="y",
                                    x=-100,
                                    y=42.5,
                                    sizex=200,
                                    sizey=85,
                                    sizing="stretch",
                                    layer='below')]
                      )
    layout = go.Layout(**layout_dict)
    style_dict = {'height':85,
                  'width':200,
                 }
    fig = go.Figure(data=data,
                    layout=layout,
                    )
    fig.update_xaxes(showgrid=False,
                     zeroline=False,
                     range=[-100, 100],
                     tickvals=[])
    fig.update_yaxes(showgrid=False,
                     zeroline=False,
                     range=[-42.5, 42.5],
                     tickvals=[])
    return fig

def get_recent_plays_string(n_steps, game_json):
    '''
    returns a string of the 5 most recent plays
    '''
    plays = dp.game_json_to_event_dicts(game_json)[max(0, n_steps-5):n_steps]
    string = ', '.join([play_dict_to_string(play) for play in plays])
    return string

def _temp_load_coarse_variable(filename, assetdir='./assets/'):
    '''
    loads a pickle object if it extists
    '''
    with open('{}{}.pkl'.format(assetdir, filename), 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_probs(n_steps, game_json, model_predicting):
    '''
    In:
        events, list of in-game events in original json format
    Out:
        id_2_event, decoding dictionary
        probs, currently None, list of probabilites where index=id
    '''

    events_id = get_events_id(n_steps, game_json)

    #load model
    #probs = mf.next_probs([0, 6,7,8], model_predicting)
    probs = events_id
    return 'RNN input: {}'.format(probs)

def make_probs_html(events):
    '''
    in: 
        id_2_event,  decoding dictionary
        probs, vector of probabilities
    out: 
        html P saying chance for next play to be a goal
    '''
    #probs = get_probs(events)
    return html.H2('Chance for goal: {}'.format(.3))

def get_events_id(n_steps, game_json):
    event_2_id = _temp_load_coarse_variable('event_2_id')
    plays = dp.game_json_to_event_dicts(game_json)[:n_steps]
    ###encode the list of jsons to list of ids 
    ##strip off to intermediate dictionary
    events_strip = dp.strip_game_events(plays)
    #encode dictionary to long to strings
    events_str = [dp.make_event_string(event) for event in events_strip]
    #string to coarse strings
    regex = re.compile('[^a-zA-Z]')
    events_str = [regex.sub('', event) for event in events_str]
    #encode strings to ids
    events_id = [event_2_id[event] for event in events_str]
    return events_id
 

id_2_event = _temp_load_coarse_variable('id_2_event')
event_2_id = _temp_load_coarse_variable('event_2_id')
vocabulary = _temp_load_coarse_variable('vocabulary')
model = mf.make_prediction_model_file('./assets/model-30.hdf5',
                                   vocabulary,
                                   hidden_size=20)
