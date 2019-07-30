'''
shot2vec webapp
@tiwariku
'''
#import base64
#from ast import literal_eval
#import uuid
#from datetime import datetime as dt
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
from baseline_models import MarkovEstimator


EXTERNAL_STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://stackpath.bootstrapcdn.com/bootstrap'
                        +'/4.3.1/css/bootstrap.min.css']
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
STRIPPER = cm.strip_name_zone
BASE_MODEL_DIR = '../baseline_models/markov_zone'
BASE_MODEL = MarkovEstimator()
BASE_MODEL = dp.unpickle_it(BASE_MODEL_DIR)

TITLE = html.Div(html.Div(children=html.H1(children='shot2vec'),
                          className='col-3 offset-1'
                         ),
                 className='row',
                )

HOCKEY_RINK = html.Div(html.Div(dcc.Graph(id='rink_plot',
                                          figure=fn.make_rink_fig(None),
                                          #style={'width':800,
                                          #       'height':450,
                                          #      },
                                         ),
                                className='col-8 offset-1'
                               ),
                       className='row'
                      )

GET_DATE = dcc.DatePickerSingle(id='date-picker',
                                date=None,
                                #style={'width':200},
                               )
GAME_DROPDOWN = dcc.Dropdown(id='game-dropdown',
                             options=[],
                             #style={'width':200}
                            )

STEP_FORWARD_BUTTOM = html.Button(id='step forward',
                                  children='Next play',
                                  n_clicks=3,
                                  #style={'width':200},
                                 )

BUTTONS = html.Div(children=[html.Div(children=GET_DATE,
                                      className='col-1 offset-1'),
                             html.Div(children=GAME_DROPDOWN,
                                      className='col-4 offset-1'),
                             html.Div(STEP_FORWARD_BUTTOM,
                                      className='col-1 offset-1')
                            ],
                   className='row'
                  )

RECENT_PLAYS = html.Div(
    html.Div(children=dcc.Graph(id='recent-table',
                                figure=fn.serve_recent_plays_table()
                               ),
             className='col-8 offset-1',
            ),
    className='row',
    )

STORE = dcc.Store(id='my-store')
GAME_PLAYS = dcc.Store(id='game-plays', data=None)

DEBUG_OUTPUT = html.Div(id='debug',
                        children='Hi, World')

LAYOUT_KIDS = [TITLE,
               BUTTONS,
               HOCKEY_RINK,
               RECENT_PLAYS,
               DEBUG_OUTPUT, STORE, GAME_PLAYS]
LAYOUT = html.Div(LAYOUT_KIDS, className='contianer')

APP = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
APP.layout = LAYOUT

# callbacks
@APP.callback(Output(component_id='recent-table', component_property='figure'),
              [Input(component_id='game-plays', component_property='data')]
             )
def update_recent_table(plays):
    """
    in: plays, the recent plays list from game_plays stored
    out: new table, wrapped in a figure, for display
    """
    return fn.serve_recent_plays_table(plays)

@APP.callback(Output(component_id='rink_plot', component_property='figure'),
              [Input(component_id='game-plays',
                     component_property='data')]
             )
def update_rink_fig(plays):
    """
    This callback updates the 'rink fig' with the game json, which is stored in
    my-store under property 'data'
    """
    goal_probs = None
    if plays:
        play_str = str(STRIPPER(plays[-1]))
        goal_probs = BASE_MODEL.goal_probs(play_str)
    return fn.make_rink_fig(plays, goal_probs)

@APP.callback(Output(component_id='my-store', component_property='data'),
              [Input(component_id='game-dropdown',
                     component_property='value')]
             )
def store_game_json(game_id):
    """
    callback function to store the selected game's response from the NHL API in
    my-store as property 'data' in JSON format
    """
    game_json = io.get_game_response(game_id=game_id).json()
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
              [Input(component_id='game-dropdown', component_property='value')])
def reset_stepper(game_id):
    """
    resets to the step forward start of the game when 'get game' is clicked
    """
    start_at = 3
    if game_id:
        return 0*game_id +start_at
    return start_at

@APP.callback(Output(component_id='game-dropdown', component_property='options'),
              [Input(component_id='date-picker', component_property='date')]
             )
def update_dropdown(date):
    """
    Callback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    if date:
        schedule_json = io.get_schedule_json(date)
        return dp.schedule_to_game_list(schedule_json)
    return []

@APP.callback(Output(component_id='debug', component_property='children'),
              [Input(component_id='date-picker', component_property='date')]
             )
def debug_display(date):
    """
    allback for debuging app.. accepts some input and displays it in a Div at
    the bottom of the page
    """
    #    seed_list = [PLAY_TO_ID[str(STRIPPER(play))] for play in data]
    #    return mf.next_probs(seed_list, MODEL_PREDICTING)
    if date:
        schedule_json = io.get_schedule_json(date)
        return str(dp.schedule_to_game_list(schedule_json))
    return 'No date yet'

if __name__ == '__main__':
    APP.run_server(debug=True)
