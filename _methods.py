from collections import Counter
import numpy as np
import plotext as plx
from copy import deepcopy as copy
from datetime import datetime as dt


is_nan = lambda el: (isinstance(el, str) and el == 'nan') or (not isinstance(el, str) and np.isnan(el))
are_nan = lambda data: np.array(np.vectorize(is_nan)(data))

sp = ' '
vline = '│'
nl = '\n'
delimiter = sp * 2 + 1 * vline + sp * 0

def tabulate(data, header = None, footer = None, decimals = 1):
    data = [header] + data if header is not None else data
    data = data + [footer] if footer is not None else data
    data = [[round(el, decimals) if is_numerical(el) else el for el in line] for line in data]
    t = np.transpose(data)
    cols = len(t); rows = len(data)
    Cols = range(cols)
    t = [[(delimiter if i != 0 else sp) + el for el in t[i]] for i in Cols]
    ls = [max([len(el) for el in col]) for col in t]
    Cols = np.array(Cols)[np.cumsum(ls) <= plx.tw()]
    t = [[pad(el, ls[i]) for el in t[i]] for i in Cols]
    d = transpose(t)
    lines = [''.join(line) for line in d]
    lines[0] = plx.colorize(lines[0], style = 'bold') if header is not None else lines[0]
    lines[-1] = plx.colorize(lines[-1], style = 'bold') if footer is not None else lines[-1]
    out = nl.join(lines)
    return out

is_numerical = lambda el: isinstance(el, float) or isinstance(el, int)
transpose = lambda data: list(map(list, zip(*data)))
pad = lambda string, length: string + sp * (length - len(string))


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
# mode = lambda data: stats.mode(data).mode
# median = lambda data: np.median(data)
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


# n = np.nan
# isnan = np.isnan

# def dates_to_seconds(dates):
#     ref = [el for el in dates if el != n][0]
#     dates = [(el - ref).total_seconds() if el != n else n for el in dates]
#     return dates

# def mean_datetime(dates):
#     return dates[0] + dt.timedelta(seconds = mean(dates_to_seconds(dates)))

# def median_datetime(dates):
#     return dates[0] + dt.timedelta(seconds = median(dates_to_seconds(dates)))

# def mode_datetime(dates):
#     return dates[0] + dt.timedelta(seconds = mode(dates_to_seconds(dates)))

# def std_datetime(dates):
#     return dt.timedelta(seconds = std(dates_to_seconds(dates)))

# div = [1, 60, 60, 24, 30.44, 12]
# div = list(map(float, np.cumprod(div)))
# forms = ['seconds', 'minutes', 'hours', 'days', 'months', 'years']

# time_to_string = lambda date, form: date.strftime(form)

# def timedelta_to_number(delta, form):
#     delta = delta.total_seconds()
#     index = forms.index(form)
#     return delta / div[index]

# timedelta_to_string = lambda delta, form: str(round(timedelta_to_number(delta, form), 1))



# headers = ['c1', 'c2', 'c3', 'd1', 'd2', 'n1', 'n2']
# footers = ['c1', 'c2', 'c3', 'd1', 'd2', 'n1', 'n2']




# # print(tabulate(data, headers, footers, decimals = 1))

# # def tabulate_data(data, decimals = 1, grid = False, headers = None):
# #     style = 'rounded_grid' if grid else 'rounded_outline'
# #     float_format = '.' + str(decimals) + 'f'
# #     headers = list(headers) if headers is not None else []
# #     return tabulate(data, headers = headers, tablefmt = style, floatfmt = float_format)


