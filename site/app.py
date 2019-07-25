#! /usr/local/bin/python3 
import in_out as io
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import base64
from numpy.random import randint 
from requests_futures.sessions import FuturesSession
import functions as fn
import model_fns as mf
session = FuturesSession(max_workers=2)


##external asses
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## the model and other static global variables
id_2_event = fn._temp_load_coarse_variable('id_2_event')
event_2_id = fn._temp_load_coarse_variable('event_2_id')
vocabulary = fn._temp_load_coarse_variable('vocabulary')
hidden_size = 20
model_predicting = mf.make_prediction_model_file('./assets/model-30.hdf5',
                                       vocabulary,
                                       hidden_size=20)
global temp_game_json
temp_game_json = io.get_game_response().json()


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
## html elements
game_data = html.Div(id='game_json',
                     style={'display': 'none'},
                     children='')#io.get_game_response().json())#io.get_game_response().content)

title = html.H1(children='Shot2Vec')

hockey_rink = html.Div([html.H2(id='rink_div', children='Recent Plays'),
                        dcc.Graph(id='rink_plot',
                                  figure=fn.make_rink_fig(1, 
                                      temp_game_json),
                        #style={'height':255, 'width':600},
                       )])

event_list = html.Div(id='recent plays Div', 
                      children=[html.H4(id='recent plays H4', 
                                        children='Recent plays'),
                                html.P(id='recent plays', 
                                       children='No plays yet')])


get_game_button = html.Button(id='get game',
                              children='Get game data',
                              n_clicks=0)

step_forward_buttom = html.Button(id='step forward',
                                  children='Step forward',
                                  n_clicks=1)

buttons = html.Div(children=[get_game_button, step_forward_buttom],
                   style={'columnCount':2}
                  )

probs = html.Div(id='probs Div', 
                 children = [html.H4(id='probs H5', children='Next event:'),
                             html.P(id='probs', children='RNN OUTPUT')])


layout_kids = [title, buttons, hockey_rink, event_list, game_data, probs]
layout = html.Div(layout_kids)
app.layout = layout
# callbacks
@app.callback(Output(component_id='game_json', 
                     component_property='children'),
              [Input('get game', 'n_clicks')],)
def update_game_json(n_clicks):
    return ''#io.get_game_response().content#html.Div('BYE!{}'.format(n_clicks))

@app.callback(Output(component_id='rink_plot', component_property='figure'),
        [Input('step forward', 'n_clicks')],
        state=[State(component_id='game_json', component_property='children')])
def step_forward(n_steps, game_json):
    return fn.make_rink_fig(n_steps, temp_game_json)#io.get_game_response().json())

@app.callback(Output(component_id='recent plays', 
                     component_property='children'), 
              [Input('step forward', 'n_clicks')],
              )
def update_recent_plays(n_steps):
    window = 5
    plays = fn.get_recent_plays_string(n_steps, temp_game_json)
    return str(plays)#', '.join()


@app.callback(Output(component_id='probs', 
                     component_property='children'), 
              [Input('step forward', 'n_clicks')],
              )
def update_probs(n_steps):
    return str(fn.get_probs(n_steps, temp_game_json, model_predicting))




if __name__=='__main__':
    app.run_server(debug=True)
