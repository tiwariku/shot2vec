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
