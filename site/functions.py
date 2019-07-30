'''
functions used by shot2vec site
@tiwariku
2019-07-26
'''
#from ast import literal_eval
import base64
from collections import defaultdict
import plotly.graph_objs as go
#import data_processing as dp

with open("assets/rink_mine.png", "rb") as image_file:
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
    marker_dict = defaultdict(lambda: 'hourglass')
    marker_dict['Faceoff'] = 'circle'
    marker_dict['Shot'] = 'star-diamond'
    marker_dict['Missed Shot'] = 'star-diamond-open'
    marker_dict['Blocked Shot'] = 'star-diamond-open-dot'
    marker_dict['Goal'] = 'star'
    marker_dict['Hit'] = 'x'
    marker_dict['Penalty'] = 'cross'
    marker_dict['Takeaway'] = 'triangle-up'
    marker_dict['Giveaway'] = 'triangle-down'
    p_type = event_json['result']['event']
    ev_dict = {'name':'',
               'mode':'markers',
               'opacity':opacity,
               'marker':{'size':10,
                         'color':'black',
                         'symbol':marker_dict[p_type],
                         'line':{'width':2}
                        }
              }
    ev_dict['text'] = [p_type,]
    x_coord, y_coord = _getcoords(event_json)
    ev_dict['x'] = [x_coord,]
    ev_dict['y'] = [y_coord,]
    return ev_dict

def _zone_prob_traces(plist, cscale=None):
    """
    in:
        plist, list of length 3, where p[0] is the probability for the left zone
               p[1] is hte probability for the neutral zone and p[2] is the
               probability for the right zone. Probability is used to set the
               opacity in the traces
    out:
        traces: list of traces for plotting
    """
    if not cscale:
        cscale = [[0, 'rgb(255, 255, 255)'], [1, 'rgb(255,0,0)']]
    assert len(plist) == 3
    ybound = [-42.5, 42.5]#, 50, 60]
    xbounds = [[-100, -25], [-25, 25], [25, 100]]
    traces = []
    for i, prob in enumerate(plist):
        zvals = [[prob,]]#, [0], [1]]
        xbound = xbounds[i]
        traces.append(go.Heatmap(x=xbound,
                                 y=ybound,
                                 z=zvals,
                                 type='heatmap',
                                 showscale=False,
                                 colorscale=cscale,
                                 zmin=0,
                                 zmax=1,
                                 hoverinfo='none',
                                 opacity=1))
    return traces

def _plays_to_traces(plays, window=10, bg_opacity=.2):
    """
    helper for make_rink_fig
    """
    data = []
    if plays:
        data = [_play_dict_to_plottable(play, bg_opacity)
                for i, play in enumerate(plays)]
        if len(data) > window:
            data = data[-window:]
        data[-1]['marker']['size'] = 20
        data[-1]['opacity'] = 1
    return data

def _make_rink_fig_layout():
    """
    helper to clean up make_rink_fig
    """
    layout_dict = dict(#title='Recent Plays',
                       showlegend=False,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       #clickmode='event+select',
                       images=[dict(source=RINK_IMAGE_ENCODED,
                                    xref="x",
                                    yref="y",
                                    x=-100,
                                    y=42.5,
                                    sizex=200,
                                    sizey=85,
                                    sizing="stretch",
                                    opacity=1,
                                    layer='above')],
                       autosize=True,
                       margin=dict(l=0, r=0, t=0, b=0),
                       #displayModeBar=False,
                      )
    return go.Layout(**layout_dict)

def make_rink_fig(plays, prob_list=None):
    """
    in:
        plays, list of plays, each in nhl api format
        prop_list, 3-vector of probability for each zone
    out:
        fig, the go.Figure object
    """
    def _curve_prob(prob):
        """
        scaling for appearance
        """
        exp = 2.71828
        prob_c = .03
        num = -exp**(-prob/prob_c)+1
        denom = -exp**(-1/prob_c)+1
        return num/denom

    layout = _make_rink_fig_layout()
    fig = go.Figure(data=None,
                    layout=layout,)

    if not prob_list:
        prob_list = [0, 0, 0]
    #print(prob_list)
    prob_list = [_curve_prob(prob) for prob in prob_list]
    #print('\t'+str(prob_list))

    traces = []
    traces.extend(_zone_prob_traces(prob_list))
    traces.extend(_plays_to_traces(plays))
    for trace in traces:
        fig.add_trace(trace)
    fig.update_xaxes(showgrid=False,
                     scaleanchor='y',
                     scaleratio=1,
                     zeroline=False,
                     range=[-100, 100],
                     tickvals=[])
    fig.update_yaxes(showgrid=False,
                     zeroline=False,
                     range=[-42.5, 42.5],
                     tickvals=[])
    return fig

def _get_recent_table_cells(plays, stop=5):
    """
    in:
        plays, a list of plays, each in nhl api format
        stop, integer number of event descriptions to return
    out:
        list of event descriptions
    """
    if len(plays) > stop:
        t_plays = plays[-stop:][::-1]
    else:
        t_plays = plays[::-1]
    descs = [play['result']['description'] for play in t_plays]
    coords = [_getcoords(play) for play in t_plays]
    return dict(values=[descs, coords])

def serve_recent_plays_table(plays=None):
    """
    in:
        plays: list of plays, each in nhl api's format
        return: go.Table object with the 5 most recent plays
    """
    layout_dict = dict(title='Recent Plays',
                       margin=dict(l=0, r=0, t=30, b=0))
    header = dict(values=['Play', '(x,y)'])
    cells = dict(values=[[], []])
    if plays:
        cells = _get_recent_table_cells(plays)
    return go.Figure(data=[go.Table(header=header,
                                    cells=cells)
                          ],
                     layout=layout_dict
                    )
