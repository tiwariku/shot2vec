import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import base64
from numpy.random import randint 
from requests_futures.sessions import FuturesSession


#function
def getcoords(event_json, sbin=10):
    play_coordinates = event_json['coordinates']
    x, y = None, None
    if 'x' in play_coordinates.keys():
        x = int(play_coordinates['x']/sbin)*sbin
    if 'y' in play_coordinates.keys():
        y = int(play_coordinates['y']/sbin)*sbin
    return (x, y)

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

    def json_to_plottable(event_json, opacity=1):
        ev_dict = {'name':'',
                   'mode':'markers',
                   'opacity':opacity,
                   'marker':{'size':12, 'color':'black'}}
        ev_dict['text'] = [event_json['result']['event'],]
        x,y = getcoords(event_json, sbin=1)
        ev_dict['x'] = [x,]
        ev_dict['y'] = [y,]
        return ev_dict

    game_data = get_random_game()[-window:]
    data = [json_to_plottable(ejson,opacity=i/window) for i, ejson in enumerate(game_data)]
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

def make_rink_fig():
    with open("assets/rink.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    #add the prefix that plotly will want when using the string as source
    encoded_image = "data:image/png;base64," + encoded_string
    data = get_example_data(window=40)
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
    
def get_probs():
    return None
