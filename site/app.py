'''
shot2vec webapp
@tiwariku
'''
#import base64
from ast import literal_eval
#import uuid
import dash
from dash.dependencies import Input, Output, State
#import dash_core_components as dcc
import dash_html_components as html
#import plotly.graph_objs as go
#from numpy.random import randint
#from flask_caching import Cache
import functions as fn
import model_fns as mf
import in_out as io
import data_processing as dp

GAME_JSON = io.get_game_response().json()

EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
CORPUS_FILENAME = '../assets/corpi/full_coords_bin_10'
ASSET_DIR = './assets/coords_bin10/'
MODEL_WEIGHTS = ASSET_DIR+'model_weights.hdf5'
VOCABULARY = dp.unpickle_it(ASSET_DIR+'vocabulary')
PLAY_TO_ID = dp.unpickle_it(ASSET_DIR+'play_to_id')
ID_TO_PLAY = dp.unpickle_it(ASSET_DIR+'id_to_play')
HIDDEN_SIZE = 30
MODEL_PREDICTING = mf.make_prediction_model_file(MODEL_WEIGHTS,
                                                 VOCABULARY,
                                                 hidden_size=HIDDEN_SIZE)



TITLE = html.H1(children='shot2vec')

HOCKEY_RINK = html.Div([html.H2(id='rink_div', children='Recent Plays'),
                        html.P(id='rink placeholder', children='Hi World'),
                        #dcc.Graph(id='rink_plot', figure=None
                        #fn.make_rink_fig(1, temp_game_json),
                       ])

GET_GAME_BUTTON = html.Button(id='get game',
                              children='Get game data',
                              n_clicks=0)

STEP_FORWARD_BUTTOM = html.Button(id='step forward',
                                  children='Step forward',
                                  n_clicks=1)

BUTTONS = html.Div(children=[GET_GAME_BUTTON, STEP_FORWARD_BUTTOM],
                   style={'columnCount':2}
                  )


LAYOUT_KIDS = [TITLE, BUTTONS, HOCKEY_RINK]
LAYOUT = html.Div(LAYOUT_KIDS)

APP = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
APP.layout = LAYOUT
# callbacks
@APP.callback(Output(component_id='rink_plot', component_property='figure'),
              [Input('step forward', 'n_clicks')],
              state=[State(component_id='game_json',
                           component_property='children')])
def step_forward(n_steps, game_json_str):
    '''
    Thsi callback updates the 'rink'fig' with the game json
    '''

    return fn.make_rink_fig(n_steps, temp_game_json)#io.get_game_response().json())

@APP.callback(Output(component_id='recent plays',
                     component_property='children'),
              [Input('step forward', 'n_clicks')],
              state=[State(component_id='game_json',
                           component_property='children')])
def update_recent_plays(n_steps, game_json_str):
    '''
    This callback updates the recent plays list with the most recent plays
    '''
    #window = 5
    temp_game_json = literal_eval(game_json_str)
    plays = fn.get_recent_plays_string(n_steps, temp_game_json)
    return str(plays)#', '.join()


@APP.callback(Output(component_id='probs',
                     component_property='children'),
              [Input('step forward', 'n_clicks')],
              state=[State(component_id='game_json',
                           component_property='children')])
def update_probs(n_steps, game_json_str):
    '''
    This function calculates the new probabilities
    '''
    temp_game_json = literal_eval(game_json_str)
    return str(fn.get_probs(n_steps, temp_game_json, MODEL_PREDICTING))




if __name__ == '__main__':
    APP.run_server(debug=True)
