from gioconda._matrix import matrix_class
from gioconda._file import *


# File Utilities
script_folder = lambda: os.path.abspath(os.path.join(inspect.getfile(sys._getframe(1)), os.pardir))
source_folder = os.path.dirname(os.path.realpath(__file__))
test_data_path = os.path.join(source_folder, 'test_data.csv')

join = os.path.join

def read_csv(file_name, delimiter = ',', header = False, log = True):
    print('loading data') if log else None 
    matrix = read_csv_(file_name, delimiter = delimiter, log = log)
    data = matrix_class()
    data._add_matrix(matrix, header)
    print('data loaded!\n') if log else None
    return data

def read_xlsx(file_name, header = False, log = True):
    print('loading data') if log else None 
    matrix = read_xlsx_(file_name, log = log)
    data = matrix_class()
    data._add_matrix(matrix, header)
    print('data loaded!\n') if log else None
    return data

def read_pickle(path, log = True):
    import pickle
    print("reading pickle", path) if log else None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("pickle read!\n") if log else None
    return data
