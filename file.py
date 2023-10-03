import pickle
import plotext as plx
from bongo.matrix import matrix_class
from bongo.methods import nl
import os

# File Utilities
bongo_folder = os.path.dirname(os.path.realpath(__file__))
test_data_path = plx.join_paths(bongo_folder, "test_data.csv")

base_folder = plx.parent_folder(plx.script_folder())
data_folder =  plx.join_paths(base_folder, "data")
get_data_path = lambda name: plx.join_paths(data_folder, name + ".csv")
get_pickle_path = lambda name: plx.join_paths(data_folder, name + ".pickle")


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
    table = matrix[1:] if header else matrix
    data = matrix_class()
    data.set_matrix(table)
    data.set_names(matrix[0]) if header else None
    print('data loaded!\n') if log else None
    return data

data_test = read_data(test_data_path, header = True)

data_test.to_datetime('d1', '%Y-%m-%d', 'years')
data_test.to_datetime('d2', '%Y-%m-%d', 'years')
data_test.to_float('n1')
data_test.to_float('n2')



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
    
# def read_pickle(path):
#     print("reading pickle", path)
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     print("pickle read!\n")
#     return data

# def write_pickle(path, object):
#     print("writing pickle")
#     with open(path, 'wb') as f:
#         pickle.dump(object, f)
#     print("pickle written!\n")

# class memorize: # it memorise the arguments of a function, when used as its decorator, to reduce computational time
#     def __init__(self, f):
#         self.f = f
#         self.memo = {}
#     def __call__(self, *args):
#         if not args in self.memo:
#             self.memo[args] = self.f(*args)
#         return self.memo[args]
