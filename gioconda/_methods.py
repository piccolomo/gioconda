from functools import lru_cache as mem
from copy import deepcopy as copy
from math import isnan
import random, string
import datetime as dt
import numpy as np
import shutil
#from matplotlib import pyplot as plt

# System
tw = lambda: shutil.get_terminal_size()[0]
th = lambda: shutil.get_terminal_size()[1]

# Nan
nan = np.nan
nat = np.datetime64('NaT')

is_nan = lambda el: el is None or (isinstance(el, str) and el == 'nan') or (is_number(el) and isnan(el)) or (isinstance(el, np.datetime64) and np.isnan(el)) or (isinstance(el, np.timedelta64) and np.isnan(el))
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

def timedelta64_to_number(delta, delta_form):
    #delta = delta.item().timestamp()
    delta = delta.item().total_seconds()
    index = forms.index(delta_form)
    return delta / div[index]

def timedelta64_to_numbers(data, form):
    return vectorize(lambda delta: timedelta64_to_number(delta, form), data)
    

timedelta64_to_string = lambda delta, form: str(round(timedelta64_to_number(delta, form), 1))

def random_datetime64(mean, std, form, delta_form):
    mean = string_to_datetime64(mean, form).item().timestamp()
    index = forms.index(delta_form)
    std = std * div[index]
    res = random.normalvariate(mean, std)
    return dt.datetime.fromtimestamp(res).strftime(form)
