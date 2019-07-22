'''
This module handles queries to and from the NHL api. It is used by the website
as well as by the model traning/development pipeline
@tiwariku
2019-07-22
'''

from requests_futures.sessions import FuturesSession
from ediblepickle import checkpoint

def get_game_response(game_number='0400', year='2015', game_type='02',
                      future=None):
    '''
    in: game_ID, year, game_type, optional future, a FutureSession get respnose
        for the game. future overrides the specified game details
    out: response from query
    '''
    if not future:
        session = FuturesSession(max_workers=1)
        future = get_game_future(session,
                                 game_number=game_number,
                                 year=year,
                                 game_type=game_type)
    return future.result()

def get_game_future(session, game_number='0400', year='2015', game_type='02'):
    '''
    in: session, a FuturesSession object for making the asynchronous requests,
        game_number, year, game_type in string format
    out: future associated with request
    '''
    game_id = f'{year}{game_type}{game_number}'
    url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live'
    return session.get(url)

@checkpoint(key=lambda args, kwargs: (str(args)+'.p').encode(),
            work_dir='./cache'.encode(),
            refresh=False)
def get_year_games(year, game_type='02', max_workers=10, verbose=False):
    '''
    in: year, the year to query
        game_type, '02' = regular season, '01' = playoff
    out:
        games, a list of dictionaries. Each dictionary is the json of hte
        associated game
    '''
    year = str(year).zfill(4)
    session = FuturesSession(max_workers=max_workers)
    max_games = 1200
    game_numbers = range(1, max_games)
    futures = []
    for game_number in game_numbers:
        game_number = str(game_number).zfill(4)
        futures.append(get_game_future(session, game_number=game_number,
                                       year=year, game_type=game_type))
    if verbose:
        print(f'Got {year} futures')
    games = []
    for i, future in enumerate(futures):
        game_json = future.result().json()
        if 'liveData' in game_json.keys():
            games.append(game_json)
        else:
            print(f'No game data for game {i+1}')
        if verbose and i%100 == 0:
            print(f'\t game {i+1}')
    return games

def cache_seasons(start_year=2010, end_year=2019):
    '''
    in: start_year, first year to download
        stop_year, last year to downloads
    >>>saves season result gsons to cache using get_year_games
    '''
    for year in range(start_year, end_year+1):
        get_year_games(year, verbose=1)
