#! /usr/local/bin/python3 
import in_out as io
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
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


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
## html elements
title = html.H1(children='Shot2Vec')
hockey_rink = html.Div([html.H2(id='rink_div', children='Recent Plays'),
                        dcc.Graph(id='rink_plot',
                                  figure=fn.make_rink_fig(),
                        #style={'height':255, 'width':600},
                       )])

game_data = html.Div(id='game_json',
                     style={'display': 'none'})

get_game_button = html.Button(id='get game',
                              children='Get game data',
                              n_clicks=0)

step_forward_buttom = html.Button(id='step forward',
                                  children='Step forward',
                                  n_clicks=1)

buttons = html.Div(children=[get_game_button, step_forward_buttom],
                   style={'columnCount':2}
                  )

layout_kids = [title, buttons, hockey_rink, game_data]
layout = html.Div(layout_kids)
app.layout = layout
# callbacks
@app.callback(
    Output(component_id='game_json', component_property='children'),
    [Input('get game', 'n_clicks')],
    #state=[State(component_id='game_json', component_property='style')]
    )
def update_game_json(n_clicks):
    return io.get_game_json()#html.Div('BYE!{}'.format(n_clicks))



if __name__=='__main__':
    #print(fn.get_probs(fn.get_random_game()[:40]))
    app.run_server(debug=True)
