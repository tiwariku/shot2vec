'''
Module for generating corpi
@tiwariku
2019-07-23
'''
import data_processing as dp

def strip_name_only(play):
    '''
    in: an event dictionary in format given by nhl api
    out: just the event type
    '''
    stripped_play = {}
    stripped_play['Type'] = play['result']['event']
    return stripped_play

def strip_name_and_coords(play, bin_size=10):
    '''
    in:     play, event dictionary in NHL API format
            bin_size, the size of the coordinate bins to use
    out: stripped event dictionary with event type and coordinates
    '''
    stripped_play = {}
    stripped_play['Type'] = play['result']['event']
    keys = play['coordinates'].keys()
    if 'x' in keys and 'y' in keys:
        coords = (play['coordinates']['x']//bin_size*bin_size,
                  play['coordinates']['y']//bin_size*bin_size)
        stripped_play['Coordinates'] = coords
    return stripped_play

if __name__ == '__main__':
    START_YEAR = 2010
    STOP_YEAR = 2018
    BIN_SIZE = 20
    CORPUS_FILENAME = f'full_coords_bin_{BIN_SIZE}'
    STRIP_FN = lambda x: strip_name_and_coords(x, bin_size=BIN_SIZE)
    CORPUS = dp.get_corpus(start_year=START_YEAR,
                           stop_year=STOP_YEAR,
                           strip_fn=STRIP_FN)
    #print('USING ONLY 300 GAMES')
    dp.pickle_it(CORPUS_FILENAME, CORPUS)
