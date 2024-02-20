from copy import deepcopy as copy
import math
import random, string
import datetime as dt
import numpy as np
import shutil
from scipy.interpolate import interp1d

#from matplotlib import pyplot as plt

# System
tw = lambda: shutil.get_terminal_size()[0]
th = lambda: shutil.get_terminal_size()[1]

# Nan
NaN = np.nan
NaT = np.datetime64('NaT')

is_nan = lambda el, nans: el is None or (isinstance(el, str) and el in nans) or (is_number(el) and math.isnan(el)) or (isinstance(el, np.datetime64) and np.isnan(el)) or (isinstance(el, np.timedelta64) and np.isnan(el))

is_number = lambda el: isinstance(el, float) or isinstance(el, int)

def remove_nan(x, y):
    a = np.transpose([x, y])
    a = a[~np.array([any(np.isnan(el)) for el in a])]
    return np.transpose(a)

# Data
transpose = lambda data, length = 1: [[]] * length if data == [] else list(map(list, zip(*data)))
vectorize = lambda method, data: np.vectorize(method)(data) if len(data) > 0 else np.array([])

def ordering(data):
    # data_sorted = sorted(data)
    # return [data_sorted.index(el) for el in data]
    return np.argsort(data)

def smooth(x, y, length = 100, window = 3):
    l, m, M = len(x), np.min(x), np.max(x); s = M - m; window = 4 * s / length
    points = round(l / length)
    points = 10
    print('length', length)
    print('window', window)
    print('points', points)
    def indices(x0):
        distances = np.abs(x - x0)
        firsts = np.unique(np.sort(distances))[ : points]
        return (distances <= window) | np.isin(distances, firsts)
    average = lambda x0: np.mean(y[indices(x0)])
    xn = np.linspace(m, M, length)
    yn = np.array([average(x0) for x0 in xn])
    return xn, yn
    

# String
sp = ' '
vline = '│'
nl = '\n'
delimiter = sp * 2 + 1 * vline + sp * 0

bold = lambda string: '\x1b[1m' + string + '\x1b[0m'
pad = lambda string, length: string + sp * (length - len(string))

def tabulate(data, header = None, decimals = 1):
    cols = len(data[0]) if len(data) > 0 else 0; rows = len(data); Cols = range(cols)
    to_string = lambda el: str(round(el, decimals)) if is_number(el) else str(el)
    data = vectorize(to_string, data)
    data = np.concatenate([[header], data], axis = 0) if header is not None and len(data) > 0 else data if len(data) > 0 else [header]
    dataT = np.transpose(data)
    prepend_delimiter = lambda el: delimiter + el
    dataT = [vectorize(prepend_delimiter, dataT[i]) if i != 0 else dataT[i] for i in Cols]
    lengths = [max(vectorize(len, data)) for data in dataT]
    Cols = np.array(Cols)[np.cumsum(lengths) <= tw()]
    dataT = [vectorize(lambda el: pad(el, lengths[i]), dataT[i]) for i in Cols]
    data = np.transpose(dataT)
    lines = [''.join(line) for line in data]
    if len(lines) > 0 and header is not None:
        lines[0] = bold(lines[0])
    out = nl.join(lines)
    return out

def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# import functools
# def mem(func):
#     memo = {}
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         key = args, frozenset(kwargs.items())
#         if key not in memo:
#             memo[key] = func(*args, **kwargs)
#         return memo[key]
#     return wrapper
