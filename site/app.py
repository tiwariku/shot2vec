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
                        dcc.Graph(id='rink_plot',
                                  figure=fn.make_rink_fig(1, GAME_JSON),)
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

STORE = dcc.Store(id='my-store')

DEBUG_OUTPUT = html.Div(id='debug',
                        children='Hi, World')

LAYOUT_KIDS = [TITLE, BUTTONS, HOCKEY_RINK, DEBUG_OUTPUT, STORE]
LAYOUT = html.Div(LAYOUT_KIDS)

APP = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
APP.layout = LAYOUT
# callbacks
@APP.callback(Output(component_id='rink_plot', component_property='figure'),
              [Input('step forward', 'n_clicks')],
              state=[State(component_id='my-store',
                           component_property='data')]
             )
def update_rink_fig(n_steps, game_json):
    '''
    This callback updates the 'rink fig' with the game json, which is stored in
    my-store under property 'data'
    '''
    return fn.make_rink_fig(n_steps, game_json)#io.get_game_response().json())

@APP.callback(Output(component_id='my-store', component_property='data'),
              [Input(component_id='get game', component_property='n_clicks')])
def store_game_json(n_clicks):
    """
    callback function to store the selected game's response from the NHL API in
    my-store as property 'data' in JSON format
    """
    game_json = None
    if n_clicks > 0:
        game_json = io.get_game_response().json()
    return game_json

@APP.callback(Output(component_id='debug', component_property='children'),
              [Input(component_id='my-store', component_property='data')])
def debug_display(data):
    """
    Callback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    if data:
        return str(data['copyright'])
    return 'No data yet'


if __name__ == '__main__':
    APP.run_server(debug=True)
