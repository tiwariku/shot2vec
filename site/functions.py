'''
functions used by shot2vec site
@tiwariku
2019-07-26
'''
#from ast import literal_eval
import base64
import plotly.graph_objs as go
#import data_processing as dp

with open("assets/rink2.png", "rb") as image_file:
    ENCODED_STRING = base64.b64encode(image_file.read()).decode()
#add the prefix that plotly will want when using the string as source
RINK_IMAGE_ENCODED = "data:image/png;base64," + ENCODED_STRING


def _getcoords(event_json):
    '''
    in:
        event_json in nhl api dict format
    '''
    play_coordinates = event_json['coordinates']
    x_coord, y_coord = None, None
    if 'x' in play_coordinates.keys():
        x_coord = play_coordinates['x']
    if 'y' in play_coordinates.keys():
        y_coord = play_coordinates['y']
    return (x_coord, y_coord)

def _play_dict_to_plottable(event_json, opacity=1):
    '''
    in:
        event_json in nhl api format
    out:
        ev_dict with appropriate keys for input to plotly figure
    '''
    ev_dict = {'name':'',
               'mode':'markers',
               'opacity':opacity,
               'marker':{'size':12, 'color':'black'}}
    ev_dict['text'] = [event_json['result']['event'],]
    x_coord, y_coord = _getcoords(event_json)
    ev_dict['x'] = [x_coord,]
    ev_dict['y'] = [y_coord,]
    return ev_dict

def make_rink_fig(plays):
    '''
    in:
        n_steps, the integer numbero of steps to display
        game_json, json from nhl api of the game
    '''

    #print(n_steps)
    #print(game_json)
    window = 10
    opac = .2
    data = {}
    if plays:
        data = [_play_dict_to_plottable(play, opac) for i, play in enumerate(plays)]
        #print(data)
        if len(data) > window:
            data = data[-window:]

        data[-1]['marker']['size'] = 30
        data[-1]['opacity'] = 1

    layout_dict = dict(title='',
                       showlegend=False,
                       #clickmode='event+select',
                       images=[dict(source=RINK_IMAGE_ENCODED,
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
