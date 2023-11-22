from functools import lru_cache as mem
from copy import deepcopy as copy
from math import isnan
import random, string
import datetime as dt
import numpy as np
import shutil
#from matplotlib import pyplot as plt

# System
tw = lambda: shutil.get_terminal_size()[1]
th = lambda: shutil.get_terminal_size()[0]

# Values
nan = np.nan
nat = np.datetime64('NaT')

is_nan = lambda el: el is None or (isinstance(el, str) and el == 'nan') or (is_number(el) and isnan(el)) or (isinstance(el, np.datetime64) and np.isnan(el))
is_number = lambda el: isinstance(el, float) or isinstance(el, int)
are_nan = lambda data: np.array([is_nan(el) for el in data], dtype = np.bool_)
are_not_nan = lambda data: np.array([not is_nan(el) for el in data], dtype = np.bool_)

# Data
#transpose = lambda data: list(map(list, zip(*data)))
vectorize = lambda method, data: np.vectorize(method)(data) if len(data) > 0 else np.array([])

# String
sp = ' '
vline = '│'
nl = '\n'
delimiter = sp * 2 + 1 * vline + sp * 0

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
    #lines[0] = plx.colorize(lines[0], style = 'bold') if header is not None else lines[0]
    out = nl.join(lines)
    return out

def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# Datetime
def string_to_datetime64(string, form):
    return np.datetime64(dt.datetime.strptime(string, form)) if string != 'nan' else nat

strings_to_datetime64 = lambda data, form: np.array([string_to_datetime64(el, form) for el in data], dtype = np.datetime64)

def mean_datetime64(dates):
    std = dates[0].item() + dt.timedelta(seconds = np.mean(dates_to_seconds(dates)))
    return np.datetime64(std)

def median_datetime64(dates):
    std = dates[0].item() + dt.timedelta(seconds = np.median(dates_to_seconds(dates)))
    return np.datetime64(std)

def std_datetime64(dates):
    std = dt.timedelta(seconds = np.std(dates_to_seconds(dates)))
    return np.timedelta64(std)

def dates_to_seconds(dates):
    dates = [(el - dates[0]).item().total_seconds() for el in dates]
    return dates

div = [1, 60, 60, 24, 30.44, 12]
div = list(map(float, np.cumprod(div)))
forms = ['seconds', 'minutes', 'hours', 'days', 'months', 'years']

time_to_string = lambda date, form: date.strftime(form)

def timedelta64_to_number(delta, form):
    delta = delta.item().total_seconds()
    index = forms.index(form)
    return delta / div[index]

timedelta64_to_string = lambda delta, form: str(round(timedelta64_to_number(delta, form), 1))

def random_datetime64(mean, std, form, delta_form):
    mean = string_to_datetime64(mean, form).item().timestamp()
    index = forms.index(delta_form)
    std = std * div[index]
    res = random.normalvariate(mean, std)
    return dt.datetime.fromtimestamp(res).strftime(form)

# import numpy as np
# from tabulate import tabulate
# from tabulate import SEPARATING_LINE as hline
# import datetime as dt
# from scipy import stats
# import matplotlib.pyplot as plt
# import plotext as plx
# import pandas as pd

# join = lambda data: [el for row in data for el in row]
# intersect = lambda data1, data2: [el for el in data1 if el in data2]
# mean = lambda data: np.mean(data)
# std = lambda data: np.std(data)
# unique = lambda data: list(set(data))
# is_list = lambda data: isinstance(data, (list, range))
# normalize = lambda data: [100 * el / sum(data) for el in data]

# def custom_sort(item):
#     return (1, item) if isinstance(item, int) else (0, item)

# def linspace(lower, upper, length = 10): # it returns a lists of numbers from lower to upper with given length
#     slope = (upper - lower) / (length - 1) if length > 1 else 0
#     return [lower + x * slope for x in range(length)]

# def datetime_linspace(lower, upper, length):
#     dates = pd.date_range(lower, upper, length)
#     return [el.to_pydatetime() for el in dates]

# def correlate(data):
#     M, s, l = max(data), sum(data), len(data)
#     a = s / l
#     return 100 - 100 * abs(M - s) / abs(a - s)

# def correlate_numerical(x, y):
#     return stats.spearmanr(x, y).statistic
#     #return stats.pearsonr(x, y)

# def correlate_categorical(x, y):
#     pass

# def cramers(confusion_matrix):
#     confusion_matrix = np.array(confusion_matrix)
#     chi2 = stats.chi2_contingency(confusion_matrix)[0]
#     s = confusion_matrix.sum()
#     phi2 = chi2 / s
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2 - ((k - 1) * (r - 1))/(s - 1))    
#     rcorr = r - ((r - 1) ** 2) / (s - 1)
#     kcorr = k - ((k - 1) ** 2 )/ (s - 1)
#     d = min( (kcorr - 1), (rcorr - 1))
#     return np.sqrt(phi2corr / d) if d != 0 else n

# correct_index = lambda r, R: 0 if r < -R else r + R if r < 0 else R if r > R else r
# correct_left_index = lambda r, R: 0 if r is None else correct_index(r, R)
# correct_right_index = lambda r, R: R if r is None else correct_index(r, R)
# correct_range = lambda r, R: range(R) if r is None else intersect(unique([correct_index(el, R) for el in r]), range(R))
# index_to_range = lambda r, R: range(0, correct_right_index(r, R)) if r >= 0 else range(correct_right_index(r, R), R) #if r < 0 else correct_range(r, R)






# headers = ['c1', 'c2', 'c3', 'd1', 'd2', 'n1', 'n2']
# footers = ['c1', 'c2', 'c3', 'd1', 'd2', 'n1', 'n2']

# print(tabulate(data, headers, footers, decimals = 1))

# def tabulate_data(data, decimals = 1, grid = False, headers = None):
#     style = 'rounded_grid' if grid else 'rounded_outline'
#     float_format = '.' + str(decimals) + 'f'
#     headers = list(headers) if headers is not None else []
#     return tabulate(data, headers = headers, tablefmt = style, floatfmt = float_format)


