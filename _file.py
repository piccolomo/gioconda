import pickle
import plotext as plx
from bongo._matrix import matrix_class
from bongo._data import nl
import os

# File Utilities
bongo_folder = os.path.dirname(os.path.realpath(__file__))
test_data_path = plx.join_paths(bongo_folder, "test_data.csv")

base_folder = plx.parent_folder(plx.script_folder())
data_folder =  plx.join_paths(base_folder, "data")


def get_data_path(name):
    return name if os.path.isfile(name) else plx.join_paths(data_folder, name + ".csv")

def read_lines(file_name, log = True):
    path = get_data_path(file_name)
    print("reading text lines in", path) if log else None
    with open(path, 'r', encoding = "utf-8") as file:
        text = file.readlines()
    text = [line for line in text if line != nl]
    print("text lines read!\n") if log else None
    return text

def split_lines(lines, delimiter = ','):
    return [line.replace("\n", "").split(delimiter) for line in lines]

def read_data(file_name, delimiter = ',', header = False, log = True):
    lines = read_lines(file_name, log = log)
    print('loading data') if log else None
    matrix = split_lines(lines, delimiter)
    data = matrix_class()
    data.add_matrix(matrix, header)
    print('data loaded!\n') if log else None
    return data

data = read_data(test_data_path, header = True, log = False)

data.to_datetime('d1', '%Y-%m-%d', 'years')
data.to_datetime('d2', '%Y-%m-%d', 'years')
data.to_numerical('n1')
data.to_numerical('n2')
data.to_categorical('c1')
data.to_categorical('c2')
data.to_categorical('c3')


def get_pickle_path(name):
    return name if os.path.isfile(name) else plx.join_paths(data_folder, name + ".pickle") 

def write_pickle(path, object):
    path = get_pickle_path(path)
    print("writing pickle")
    with open(path, 'wb') as f:
        pickle.dump(object, f)
    print("pickle written!\n")

def read_pickle(path):
    path = get_pickle_path(path)
    print("reading pickle", path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("pickle read!\n")
    return data

# def write(path, data, delimiter = ','):
#     print("writing data to", path)
#     text = '\n'.join([delimiter.join(map(str, line)) for line in data])
#     with open(path, 'w', encoding = "utf-8") as file:
#         file.write(text)
#     print("data saved!\n")

# def read_pandas(path, length = None):
#     print("pandas is reading", path)
#     data = pd.read_csv(path, nrows = length)
#     data = data.fillna("nan")
#     print("csv read!\n")
#     return data
    
# class memorize: # it memorise the arguments of a function, when used as its decorator, to reduce computational time
#     def __init__(self, f):
#         self.f = f
#         self.memo = {}
#     def __call__(self, *args):
#         if not args in self.memo:
#             self.memo[args] = self.f(*args)
#         return self.memo[args]
