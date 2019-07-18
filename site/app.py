#! /usr/local/bin/python3 
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import base64
from numpy.random import randint 
from requests_futures.sessions import FuturesSession
import functions as fn
session = FuturesSession(max_workers=2)


##external asses
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


##callbacks


## html elements
title = html.H1(children='Shot2Vec')
hockey_rink = html.Div([html.H2('Recent Plays'),
                        dcc.Graph(id='rink_plot',
                                  figure=fn.make_rink_fig(),
                        #style={'height':240, 'width':600},
                       )])
predictions = html.H2(id='predictions',
                      children='Predictions:')

layout_kids = [title, hockey_rink, predictions]
layout = html.Div(layout_kids)


if __name__=='__main__':
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    print(fn.get_probs(fn.get_random_game()))
    app.layout = layout 
    app.run_server(debug=True)
