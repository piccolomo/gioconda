import numpy as np
import plotext as plx
from copy import deepcopy as copy
import datetime as dt
import stats
from math import isnan
from functools import lru_cache as mem
from matplotlib import pyplot as plt

class data_class():
    def __init__(self, data = [], name = '', index = 'none'):
        self.set_data(data)
        self.set_name(name)
        self.set_type()
        self.set_index(index)
        self.set_forms()

    def set_data(self, data = []):
        self.data = np.array(data, dtype = object)
        self.update_length()

    def set_name(self, name = 'none'):
        self.name = name

    def set_type(self, type = 'mixed'):
        self.type = type

    def set_index(self, index = 'none'):
        self.index = index

    def set_forms(self, form = '%d/%m/%Y', delta_form = 'years'):
        self.form = form
        self.delta_form = delta_form
        
    def update_length(self):
        self.rows = len(self.data)
        self.Rows = np.arange(self.rows)


    def get(self, row, string = False):
        el = self.data[row]
        el = self.to_string(el) if string else el
        return el

    def get_section(self, rows = None, nan = True, string = False):
        rows = self.correct_rows(rows)
        data = self.data[rows]
        data = data if nan else data[are_not_nan(data)]
        data = self.to_strings(data) if string else data
        return data
    
    def correct_rows(self, rows):
        return np.array([row for row in rows if row in self.Rows]) if isinstance(rows, list) else self.Rows if rows is None else rows
    

    @mem(maxsize = None)
    def counts(self, norm = False):
        #c =  Counter(self.data)
        u, v = np.unique(self.data, return_counts = 1)
        t = sum(v)
        c = {u[i] : v[i] for i in range(len(u))}
        c = {key: (100 * count / t) for key, count in c.items()} if norm else c
        return c

    def count(self, value, norm = False):
        counts = self.counts(norm)
        return counts[value] if value in counts else 0
    
    def count_nan(self, norm = False):
        c = np.count_nonzero(are_nan(self.data))
        return nan if self.rows == 0 else 100 * c / self.rows if norm else c

    def unique(self, nan = True):
        return np.array(list(self.counts().keys()), dtype = self.data.dtype)

    def distinct(self, nan = True):
        return len(self.unique(nan))

    def mode(self):
        unique = self.unique()
        return unique[0] if len(unique) > 0 else nan


    def to_categorical(self):
        self.data = self.to_strings(self.data)
        self.update_length()
        self.set_type('categorical')
        return self

    def to_numerical(self):
        self.data = self.to_numbers(self.data)
        self.update_length()
        self.set_type('numerical')
        return self

    def to_numbers(self, data):
        return np.vectorize(np.float64)(data)

    def to_type_none(self):
        self.set_data(self.data, 'none')
        self.set_type('none')
        return self

    def to_datetime(self, form = '%d/%m/%Y', delta_form = 'years'):
        self.data = strings_to_datetime64(self.data, form)
        self.update_length()
        self.set_type('datetime')
        self.set_forms(form, delta_form)
        return self


    def is_mixed(self):
        return self.type == 'mixed'

    def is_categorical(self):
        return self.type == 'categorical'

    def is_non_categorical(self):
        return not self.is_categorical()

    def is_numerical(self):
        return self.type == 'numerical'
    
    def is_datetime(self):
        return self.type == 'datetime'
    
    def is_countable(self):
        return self.is_numerical() or self.is_datetime()
    
    def is_uncountable(self):
        return not self.is_countable()
    
    
    
    def strip(self):
        self.apply(lambda string: string.strip()) if self.is_categorical() else print('not categorical')

    def replace(self, old, new):
        self.apply(lambda string: string.replace(old, new)) if self.is_categorical() else print('not categorical')

    def apply(self, function, *args):
        data = np.vectorize(function, *args)(self.data)
        self.set_data(data)


        
    def min(self, string = False):
        data = self.get_section(nan = False); l = len(data)
        m = nan if l == 0 or self.is_uncountable() else np.min(data)
        return m
    
    def max(self):
        data = self.get_section(nan = False); l = len(data)
        m = nan if l == 0 or self.is_uncountable() else np.max(data)
        return m

    def span(self):
        m, M = self.min(), self.max()
        s = M - m
        return s if not self.is_datetime() else s
    
    def std(self):
        data = self.get_section(nan = False); l = len(data)
        return nan if l == 0 or self.is_uncountable() else std_datetime64(data) if self.is_datetime() else np.std(data)
    
    def density(self):
        std = self.std().item().total_seconds() if self.is_datetime() else self.std()
        span = self.span().item().total_seconds() if self.is_datetime() else self.span()
        return 100 * std / span if span != 0 else np.inf

    def mean(self):
        data = self.get_section(nan = False); l = len(data)
        return nan if l == 0 or self.is_uncountable() else mean_datetime64(data) if self.is_datetime() else np.mean(data)

    def median(self):
        data = self.get_section(nan = False); l = len(data)
        return nan if l == 0 or self.is_uncountable() else median_datetime64(data) if self.is_datetime() else np.median(data)

    def get_numerical_data(self):
        data = self.get_section(nan = False)
        data = [el.item().timestamp() for el in data] if self.is_datetime() else data
        return data if self.is_countable() else []
    
    def multiply(self, k):
        self.set_data(k * self.data) if self.is_numerical() else print('not numerical')
        

    def tabulate_counts(self, norm = False, length = 10):
        header = [self.name, 'count']
        counts = list(self.counts(norm = norm).items())
        table = tabulate(counts, header = header) + nl
        return table

    def print_counts(self, norm = False, length = 10):
        print(self.tabulate_counts(norm, length))
        
    @mem(maxsize = None)
    def basic_info(self):
        return {'name': self.name, 'index': self.index, 'type': self.type, 'rows': self.rows, 'nan': self.count_nan(), 'unique': self.distinct()}

    @mem(maxsize=None)
    def numerical_info(self):
        info = {'min': self.min(), 'max': self.max(), 'span': self.span(), 'nan': self.count_nan(1), 'mean': self.mean(), 'median': self.median(), 'mode': self.mode(), 'std': self.std(), 'density': self.density()}
        return {k : self.to_string(info[k]) for k in info.keys()}
    
    def info(self):
        info = self.basic_info()
        info.update(self.numerical_info()) if self.is_countable() else None
        return info

    def plot(self, bins = 100):
        plt.figure(0, figsize = (15, 8)); plt.clf()
        bins = min(bins, len(self.unique())) if self.is_countable() else None
        plt.hist(self.get_section(nan = False), bins = bins) if self.is_countable() else None
        plt.bar(self.counts().keys(), self.counts().values()) if self.is_uncountable() else None
        plt.xlabel(self.name); plt.ylabel('count')
        plt.xticks(rotation = 90) if self.is_countable() else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()
    
    
    def tabulate_info(self):
        info = self.info()
        table = [info.keys(), info.values()]
        table = tabulate(transpose(table))
        return table + nl

    def print(self):
        print(self.tabulate_info())

    def get_sample_data(self, length = 10):
        m = min(self.rows, length)
        start = [self.to_string(self.data[i]) for i in range(0, m)]
        end = [self.to_string(self.data[i]) for i in range(-m, 0)]
        out = ', '.join(start)
        out += ' ... ' if self.rows > length else ''
        out += ', '.join(end) if self.rows > 2 * length else ''
        return out

    def to_string(self, el):
        return 'nan' if is_nan(el) else el.item().strftime(self.form) if isinstance(el, np.datetime64) else timedelta64_to_string(el, self.delta_form) if isinstance(el, np.timedelta64) else str(round(el, 2)) if is_number(el) else str(el)

    def to_strings(self, data):
        return np.array([self.to_string(el) for el in data])

    def __str__(self):
        return self.tabulate_info() + nl + sp + self.get_sample_data()

    def __repr__(self):
        return str(self)


        

    def equal(self, value):
        data = self.get_section(string = 1) if isinstance(value, str) and self.is_datetime() else self.data
        return are_nan(self.data) if is_nan(value) else self.data == value

    def not_equal(self, value):
        return ~ self.equal(value)

    def higher(self, value, equal = False):
        return self.data >= value if equal else self.data > value

    def lower(self, value, equal = False):
        return ~self.higher(value, not equal)


    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new.set_name(self.name)
        new.set_type(self.type)
        new.set_index(self.index)
        return new

    def part(self, a = None, b = None):
        a = 0 if a is None else a
        b = self.rows if b is None else b
        return self.subset(np.arange(a, b))
    
    def subset(self, rows):
        new = self.empty()
        new.set_data(self.get_section(rows))
        return new


nan = np.nan
nat = np.datetime64('NaT')

is_nan = lambda el: el is None or (isinstance(el, str) and el == 'nan') or (is_number(el) and isnan(el)) or (isinstance(el, np.datetime64) and np.isnan(el))
is_number = lambda el: isinstance(el, float) or isinstance(el, int)
are_nan = lambda data: np.array([is_nan(el) for el in data], dtype = np.bool_)
are_not_nan = lambda data: np.array([not is_nan(el) for el in data], dtype = np.bool_)

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


sp = ' '
vline = '│'
nl = '\n'
delimiter = sp * 2 + 1 * vline + sp * 0

def tabulate(data, header = None, footer = None, decimals = 1):
    cols = len(data[0]); rows = len(data); Cols = range(cols)
    data = np.concatenate([[header], data], axis = 0) if header is not None else data
    to_string = lambda el: str(round(el, decimals)) if is_number(el) else str(el)
    data = np.vectorize(to_string)(data)
    dataT = np.transpose(data)
    prepend_delimiter = lambda el: delimiter + el
    dataT = [np.vectorize(prepend_delimiter)(dataT[i]) if i != 0 else dataT[i] for i in Cols]
    lengths = [max(np.vectorize(len)(data)) for data in dataT]
    Cols = np.array(Cols)[np.cumsum(lengths) <= plx.tw()]
    dataT = [np.vectorize(pad)(dataT[i], lengths[i]) for i in Cols]
    data = transpose(dataT)
    lines = [''.join(line) for line in data]
    lines[0] = plx.colorize(lines[0], style = 'bold') if header is not None else lines[0]
    lines[-1] = plx.colorize(lines[-1], style = 'bold') if footer is not None else lines[-1]
    out = nl.join(lines)
    return out



transpose = lambda data: list(map(list, zip(*data)))
pad = lambda string, length: string + sp * (length - len(string))





#     def select(self, value, data):
#         new = data.empty()
#         rows = self.equal(value)
#         new.set_data(data.get(rows))
#         return new

#     def cross_count(self, value, data, norm = False):
#         s = self.select(value, data)
#         c = s.not_nan() if s.is_categorical() else s.mean()
#         return 100 * c / self.rows if norm else c

#     def cross_counts(self, data, norm = False, nan = True):
#         u = self.unique(nan)
#         v = [self.cross_count(el, data, norm) for el in u]
#         c = transpose([u, v])
#         return sorted(c, key = lambda el: data.min() if isinstance(el[1], str) else el[1], reverse = True)

#     def cross_unique(self, data, nan = True):
#         return [el[0] for el in self.cross_counts(data, 0, nan)]
3
    # def get_numpy_type(self, type):
    #     return object if type == 'mixed' else '<U3' if type == 'categorical' else np.float64 if type == 'numerical' else np.datetime64 if type == 'datetime' else None
