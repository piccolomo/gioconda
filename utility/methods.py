import numpy as np
from tabulate import tabulate
from tabulate import SEPARATING_LINE as hline
import datetime as dt
from scipy import stats
from copy import deepcopy as copy
import plotext as plt
import pandas as pd

transpose = lambda data: list(map(list, zip(*data)))
join = lambda data: [el for row in data for el in row]
intersect = lambda data1, data2: [el for el in data1 if el in data2]
to_integer = lambda el: int(el) if el != 'nan' else el   
mean = lambda data: np.mean(data)
mode = lambda data: stats.mode(data).mode
median = lambda data: np.median(data)
std = lambda data: np.std(data)
unique = lambda data: list(set(data))
is_list = lambda data: isinstance(data, (list, range))

def linspace(lower, upper, length = 10): # it returns a lists of numbers from lower to upper with given length
    slope = (upper - lower) / (length - 1) if length > 1 else 0
    return [lower + x * slope for x in range(length)]

def datetime_linspace(lower, upper, length):
    dates = pd.date_range(lower, upper, length)
    return [el.to_pydatetime() for el in dates]

def correlate(data):
    data = np.array(data)
    s, l = sum(data), len(data)
    data0 = np.array([s / l] * l)
    return 100 * np.linalg.norm(data - data0) / np.linalg.norm(data)

#correct_index = lambda r, R: max(0, min(correct_index_sign(r, R), R))

correct_index = lambda r, R: r if r >= 0 else r + R
correct_left_index = lambda r, R: 0 if r is None else correct_index(r, R)
correct_right_index = lambda r, R: R if r is None else correct_index(r, R)
correct_range = lambda r, R: range(R) if r is None else intersect(unique([correct_index(el, R) for el in r]), range(R))
index_to_range = lambda r, R: range(0, correct_right_index(r, R)) if r >= 0 else range(correct_right_index(r, R), R) #if r < 0 else correct_range(r, R)


nl = '\n'
n = 'nan'

def tabulate_data(data, decimals = 1, grid = False, headers = None):
    style = 'rounded_grid' if grid else 'rounded_outline'
    float_format = '.' + str(decimals) + 'f'
    headers = list(headers) if headers is not None else []
    return tabulate(data, headers = headers, tablefmt = style, floatfmt = float_format)


def dates_to_seconds(dates):
    ref = [el for el in dates if el != n][0]
    dates = [(el - ref).total_seconds() if el != n else n for el in dates]
    return dates

def mean_datetime(dates):
    return dates[0] + dt.timedelta(seconds = mean(dates_to_seconds(dates)))

def median_datetime(dates):
    return dates[0] + dt.timedelta(seconds = median(dates_to_seconds(dates)))

def mode_datetime(dates):
    return dates[0] + dt.timedelta(seconds = mode(dates_to_seconds(dates)))

def std_datetime(dates):
    return dt.timedelta(seconds = std(dates_to_seconds(dates)))

div = [1, 60, 60, 24, 30.44, 12]
div = np.cumprod(div)
forms = ['seconds', 'minutes', 'hours', 'days', 'months', 'years']

def timedelta_to_number(delta, form):
    delta = delta.total_seconds()
    index = forms.index(form)
    return delta / div[index]



