from requests_futures.sessions import FuturesSession

def get_game_json(game_number='0400', year='2015', game_type='02'):
    '''
    in: game_ID, year, game_type
    out: response from query in json format
    '''
    session = FuturesSession(max_workers=1)
    game_ID = f'{year}{game_type}{game_number}'
    url = f'https://statsapi.web.nhl.com/api/v1/game/{game_ID}/feed/live'
    future = session.get(url)
    print(type(future.result()))
    return str(future.result().json())


