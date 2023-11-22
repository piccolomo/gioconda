import pickle
from gioconda._matrix import matrix_class
from gioconda._data import nl
import os, sys
import inspect 

# File Utilities
script_folder = lambda: os.path.abspath(os.path.join(inspect.getfile(sys._getframe(1)), os.pardir))
source_folder = os.path.dirname(os.path.realpath(__file__))
test_data_path = os.path.join(source_folder, 'test_data.csv')

join = os.path.join

def read_lines(path, log = True):
    print("reading text lines in", path) if log else None
    with open(path, 'r', encoding = "utf-8") as file:
        text = file.readlines()
    text = [line for line in text if line != nl]
    print("text lines read!\n") if log else None
    return text

def split_lines(lines, delimiter = ','):
    return [line.replace("\n", "").split(delimiter) for line in lines]

def read(file_name, delimiter = ',', header = False, log = True):
    "Random Doc"
    lines = read_lines(file_name, log = log)
    print('loading data') if log else None
    matrix = split_lines(lines, delimiter)
    data = matrix_class()
    data._add_matrix(matrix, header)
    print('data loaded!\n') if log else None
    return data

# def write_pickle(path, object):
#     print("writing pickle")
#     with open(path, 'wb') as f:
#         pickle.dump(object, f)
#     print("pickle written!\n")

# def read_pickle(path):
#     print("reading pickle", path)
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     print("pickle read!\n")
#     return data
