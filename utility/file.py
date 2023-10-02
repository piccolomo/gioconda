import pickle
import plotext as plx
from utility.matrix import matrix_class

# File Utilities
base_folder = plx.parent_folder(plx.script_folder())
data_folder =  plx.join_paths(base_folder, "data")
data_csv = plx.join_paths(data_folder, "data.csv")
data_pickle = plx.join_paths(data_folder, "data.pickle")

def get_data_path(file_name):
    return plx.join_paths(data_folder, file_name + ".csv")

def read_lines(file_name):
    path = get_data_path(file_name)
    print("reading text lines in", path)
    with open(path, 'r', encoding = "utf-8") as file:
        text = file.readlines()
    print("text lines read!\n")
    return text

def split_lines(lines, delimiter = ','):
    return [line.replace("\n", "").split(delimiter) for line in lines]

def read_data(file_name, delimiter = ',', header = False):
    lines = read_lines(file_name)
    print('loading data')
    matrix = split_lines(lines, delimiter)
    table = data[1:] if header else matrix
    data = matrix_class()
    data.set_matrix(matrix)
    data.set_names(data[0]) if header else None
    print('data loaded!\n')
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
