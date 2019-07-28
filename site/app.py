'''
shot2vec webapp
@tiwariku
'''
#import base64
#from ast import literal_eval
#import uuid
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
#import plotly.graph_objs as go
#from numpy.random import randint
#from flask_caching import Cache
import functions as fn
import model_fns as mf
import in_out as io
import data_processing as dp
import corpse_maker as cm


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
STRIPPER = cm.strip_name_and_coords

TITLE = html.H1(children='shot2vec')

HOCKEY_RINK = html.Div([html.H2(id='rink_div', children='Recent Plays'),
                        html.Div(dcc.Graph(id='rink_plot',
                                           figure=fn.make_rink_fig(None),),
                                 style={'width':700})
                       ],
                      )

GET_GAME_BUTTON = html.Button(id='get game',
                              children='Get game data',
                              n_clicks=0)

STEP_FORWARD_BUTTOM = html.Button(id='step forward',
                                  children='Step forward',
                                  n_clicks=3)

BUTTONS = html.Div(children=[GET_GAME_BUTTON, STEP_FORWARD_BUTTOM],
                   style={'columnCount':2}
                  )

STORE = dcc.Store(id='my-store')
GAME_PLAYS = dcc.Store(id='game-plays', data=None)

DEBUG_OUTPUT = html.Div(id='debug',
                        children='Hi, World')

LAYOUT_KIDS = [TITLE, BUTTONS, HOCKEY_RINK, DEBUG_OUTPUT, STORE, GAME_PLAYS]
LAYOUT = html.Div(LAYOUT_KIDS)

APP = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
APP.layout = LAYOUT

# callbacks
@APP.callback(Output(component_id='rink_plot', component_property='figure'),
              [Input(component_id='game-plays',
                     component_property='data')]
             )
def update_rink_fig(plays):
    """
    This callback updates the 'rink fig' with the game json, which is stored in
    my-store under property 'data'
    """
    return fn.make_rink_fig(plays)

@APP.callback(Output(component_id='my-store', component_property='data'),
              [Input(component_id='get game', component_property='n_clicks')]
             )
def store_game_json(n_clicks):
    """
    callback function to store the selected game's response from the NHL API in
    my-store as property 'data' in JSON format
    """
    game_json = None
    if n_clicks > 0:
        game_json = io.get_game_response().json()
    return game_json

@APP.callback(Output(component_id='game-plays', component_property='data'),
              [Input(component_id='step forward',
                     component_property='n_clicks')],
              state=[State(component_id='my-store',
                           component_property='data')]
             )
def get_current_plays(n_clicks, game_json):
    """
    makes a truncated list of game plays (so far) into store game-plays
    """
    if game_json:
        return dp.game_to_plays(game_json, cast_fn=lambda x: x)[:n_clicks]
    return None

@APP.callback(Output(component_id='step forward',
                     component_property='n_clicks'),
              [Input(component_id='get game', component_property='n_clicks')])
def reset_stepper(n_clicks):
    """
    resets to the step forward start of the game when 'get game' is clicked
    """
    return 0*n_clicks+3

@APP.callback(Output(component_id='debug', component_property='children'),
              [Input(component_id='game-plays', component_property='data')]
             )
def debug_display(data):
    """
    Callback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    if data:
        seed_list = [PLAY_TO_ID[str(STRIPPER(play))] for play in data]
        return mf.next_probs(seed_list, MODEL_PREDICTING)
    return str(PLAY_TO_ID)

if __name__ == '__main__':
    APP.run_server(debug=True)
