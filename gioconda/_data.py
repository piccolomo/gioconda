from gioconda._methods import *
from matplotlib import pyplot as plt
from random import sample
from sklearn.preprocessing import StandardScaler as ss

metrics = ['min', 'max', 'mean', 'sum', 'length', 'mode', 'mode_frequency', 'median', 'std', 'sem']

class data_class():
    def __init__(self, data = [], name = '', index = 'none'):
        self._set_data(data)
        self._set_name(name)
        self._set_type()
        self._set_index(index)
        self._set_forms()
        self.set_nans()

    def _set_data(self, data = []):
        self._data = np.array(data)
        self._order = self._data.argsort()
        self._update_length()

    def _set_name(self, name = 'none'):
        self._name = name

    def rename(self, name):
        self._set_name(name)
        return self


    def _set_type(self, type = 'categorical'):
        self._type = type

    def _set_index(self, index = 'none'):
        self._index = index

    def _set_forms(self, form = '%d/%m/%Y', delta_form = 'years'):
        self._form = form
        self._delta_form = delta_form

    def set_nans(self, nans = ['nan']):
        self._nans = nans


    def string_to_datetime64(self, string):
        return np.datetime64(dt.datetime.strptime(string, self._form)) if self.is_not_nan(string) else NaT

    def strings_to_datetime64(self, strings):
        return np.array([self.string_to_datetime64(el) for el in strings], dtype = np.datetime64)

    def timedelta64_to_number(self, delta):
        if self.is_nan(delta):
            return NaN
        #delta = delta.item().timestamp()
        delta = delta.item().total_seconds()
        index = forms.index(self._delta_form)
        return delta / div[index]

    def timedelta64_to_numbers(self, data):
        return np.array([timedelta64_to_number(el) for el in data])

    def timedelta64_to_string(self, delta):
        return str(round(self.timedelta64_to_number(delta), 1))
    
    def _update_length(self):
        self._rows = len(self._data)
        self._Rows = np.arange(self._rows)


    def get(self, row, string = False):
        el = self._data[row]
        el = self._to_string(el) if string else el
        return el

    def get_section(self, rows = None, nan = True, string = False):
        rows = self._correct_rows(rows)
        data = self._data[rows] if len(rows) > 0 else np.array([])
        data = data if nan else data[self._are_not_nan(data)]
        data = self._to_strings(data) if string else data
        return data
    
    def _correct_rows(self, rows = None):
        rows = self._Rows if rows is None else np.array(rows)
        rows = np.array([r for r in self._Rows[:len(rows)] if rows[r]]) if rows.dtype == np.bool_ else rows
        rows =  rows[rows < self._rows]
        return rows

    #@mem(maxsize = None)
    def counts(self, norm = False, nan = True, string = False):
        u, v = np.unique(self._data, return_counts = 1)
        not_nans = self._are_not_nan(u)
        u = u if nan else u[not_nans]
        v = v if nan else v[not_nans]
        t = sum(v)
        c = {u[i] : v[i] for i in range(len(u))}
        c = {key: (100 * count / t) for key, count in c.items()} if norm else c
        c = {self._to_string(k): c[k] for k in c} if string else c
        return c

    def count(self, value, norm = False, string = False):
        counts = self.counts(norm, string = string)
        return counts[value] if value in counts else 0
        #return 100 * count / self.rows if norm else count
    
    def count_nan(self, norm = False):
        c = np.count_nonzero(self._are_nan(self._data))
        return NaN if self._rows == 0 else 100 * c / self._rows if norm else c

    #@mem
    def unique(self, nan = True, string = False):
        return list(self.counts(False, nan = nan, string = string).keys())

    def distinct(self, nan = True):
        return len(self.unique(nan))

    def mode(self):
        unique = self.unique(0)
        return unique[0] if len(unique) > 0 else NaN

    def mode_frequency(self):
        return  self.counts(1)[self.mode()] if self._rows > 0 else NaN


    def to_categorical(self):
        self._data = self._to_strings(self._data)
        self._update_length()
        self._set_type('categorical')
        return self

    def to_numerical(self, dictionary = None):
        self._data = self._to_numbers(dictionary)
        self._update_length()
        self._set_type('numerical')
        self.set_nans(self._nans)
        return self

    def _to_numbers(self, dictionary = None):
        if self.is_datetime():
            return  np.array([self.timedelta64_to_number(el) if not self.is_nan(el) else nan for el in self._data - self.min()])
        elif dictionary is None:
            return np.array([np.float64(el) if not self.is_nan(el) else NaN for el in self._data])
        else:
            #return np.array([dictionary[el] if el in dictionary and not is_nan(el) else nan for el in self._data])
            return np.array([dictionary[el] if el in dictionary else el for el in self._data])


    def to_datetime(self, form = '%d/%m/%Y', delta_form = 'years'):
        self._set_forms(form, delta_form)
        self._data = self.strings_to_datetime64(self._data)
        self._update_length()
        self._set_type('datetime')
        self.set_nans(self._nans)
        return self


    def is_mixed(self):
        return self._type == 'mixed'

    def is_categorical(self):
        return self._type == 'categorical'

    def is_non_categorical(self):
        return not self.is_categorical()

    def is_numerical(self):
        return self._type == 'numerical'
    
    def is_datetime(self):
        return self._type == 'datetime'
    
    def is_countable(self):
        return self.is_numerical() or self.is_datetime()
    
    def is_uncountable(self):
        return not self.is_countable()
    
    
    
    def strip(self):
        self._apply(lambda string: string.strip()) if self.is_categorical() else print('not categorical')

    def replace_string(self, old, new):
        return self._apply(lambda string: string.replace(old, new))

    def replace(self, old, new):
        return self._apply(lambda el: new if el == old else el)
        
    def replace_nan(self, metric = 'mean'):
        new = self.mode() if self.is_categorical() else self.get_metric(metric) if isinstance(metric, str) and metric in metrics else (self.string_to_datetime64(metric) if self.is_datetime() else metric)
        self._data = np.array([new if self.is_nan(el) else el for el in self._data])
        self._clear()

    def _apply(self, function):
        data = vectorize(function, self._data)
        self._set_data(data)
        return self

    def len(self):
        return self._rows

    def sum(self):
        data = self.get_section(nan = False); l = len(data)
        return NaN if l == 0 or self.is_uncountable() else mean_datetime64(data) if self.is_datetime() else np.sum(data)
        
    def min(self, string = False):
        data = self.get_section(nan = False); l = len(data)
        m = NaN if l == 0 or self.is_uncountable() else np.min(data)
        return m
    
    def max(self):
        data = self.get_section(nan = False); l = len(data)
        m = NaN if l == 0 or self.is_uncountable() else np.max(data)
        return m

    def span(self):
        m, M = self.min(), self.max()
        s = M - m
        return self.timedelta64_to_number(s) if self.is_datetime() else s

    def max_diff(self):
        data = self.get_section(nan = False); 
        data = np.diff(np.sort(data))
        data = data[np.nonzero(data)]
        l = len(data)
        m = NaN if l == 0 or self.is_uncountable() else np.max(data)
        return m

    def std(self):
        data = self.get_section(nan = False); l = len(data)
        return NaN if l == 0 or self.is_uncountable() else self.timedelta64_to_number(std_datetime64(data)) if self.is_datetime() else np.std(data)
    
    def sem(self):
        return self.std() / np.sqrt(self._rows)
    
    def density(self):
        std = np.nan if self.is_nan(self.std()) else self.std() if self.is_datetime() else self.std()
        span = np.nan if self.is_nan(self.span()) else  self.span() if self.is_datetime() else self.span()
        return np.nan if self.is_nan(std) or self.is_nan(span) else 100 * std / span if span != 0 else np.inf

    def mean(self):
        data = self.get_section(nan = False); l = len(data)
        return NaN if l == 0 or self.is_uncountable() else mean_datetime64(data) if self.is_datetime() else np.mean(data)

    def median(self):
        data = self.get_section(nan = False); l = len(data)
        return NaN if l == 0 or self.is_uncountable() else median_datetime64(data) if self.is_datetime() else np.median(data)

    def percentile(self, q):
        data = self.get_section(nan = False); l = len(data)
        return NaN if l == 0 or self.is_uncountable() else np.percentile(data, q)

    def iqr(self):
        return self.percentile(74) - self.percentile(25)

    def get_metric(self, metric = 'mean', string = False):
        functions = [self.min, self.max, self.mean, self.sum, self.__len__, self.mode, self.mode_frequency, self.median, self.std, self.sem]
        res = functions[metrics.index(metric)]()
        return self._to_string(res) if string else res 

    def _get_numerical_data(self):
        data = self.get_section(nan = False)
        data = [el.item().timestamp() for el in data] if self.is_datetime() else data
        return data if self.is_countable() else []

    def _get_max_bins(self):
        s = self.span()
        m = self.max_diff()
        return 2 if self.is_nan(s) or self.is_nan(m) else max(2, int(np.ceil(s / m)))
    
    def multiply(self, k):
        self._set_data(k * self._data) if self.is_numerical() else print('not numerical')
        

    def _tabulate_counts(self, norm = False, nan = True, length = 10):
        header = [self._name, 'count']
        counts = list(self.counts(norm = norm, nan = nan).items())
        counts = sorted(counts, key = lambda el: el[1], reverse = 1)[: length]
        table = tabulate(counts, header = header) + nl
        return table

    def print_counts(self, norm = False, nan = True, length = 10):
        print(self._tabulate_counts(norm, nan, length))
        
    #@mem(maxsize = None)
    def _basic_info(self):
        return {'name': self._name, 'index': self._index, 'type': self._type, 'rows': self._rows, 'nan': self.count_nan(), 'distinct': self.distinct(nan = False)}

    #@mem(maxsize=None)
    def numerical_info(self):
        info = {'min': self.min(), 'max': self.max(), 'span': self.span(), 'nan': self.count_nan(0), 'mean': self.mean(), 'median': self.median(), 'mode': self.mode(), 'mode_frequency': self.mode(), 'std': self.std(), 'sem': self.sem(), 'iqr': self.iqr(), 'density': self.density()}
        return {k : self._to_string(info[k]) for k in info.keys()}

    def datetime_info(self):
        return {'form': self._form, 'delta_form': self._delta_form}
    
    
    def info(self):
        info = self._basic_info()
        info.update(self.numerical_info()) if self.is_countable() else None
        info.update(self.datetime_info()) if self.is_datetime() else None
        return info

    def plot(self, bins = 100):
        #plt.figure(0, figsize = (15, 8)); plt.clf()
        plt.figure(0); plt.clf()
        bins = min(bins, len(self.unique())) if self.is_countable() else None
        plt.hist(self.get_section(nan = False), bins = bins) if self.is_countable() else None
        plt.bar(self.counts().keys(), self.counts().values()) if self.is_uncountable() else None
        plt.xlabel(self._name); plt.ylabel('count')
        plt.xticks(rotation = 90) if self.is_categorical() else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()
    
    
    def _tabulate_info(self):
        info = self.info()
        table = [list(info.keys()), list(info.values())]
        table = tabulate(np.transpose(table))
        return table + nl

    def print(self):
        print(self._tabulate_info())

    def _get_sample_data(self, length = 10):
        m = min(self._rows, length)
        start = [self._to_string(self._data[i]) for i in range(0, m)]
        end = [self._to_string(self._data[i]) for i in range(-m, 0)]
        out = ', '.join(start)
        out += ' ... ' if self._rows > length else ''
        out += ', '.join(end) if self._rows > 2 * length else ''
        return out

    def _to_string(self, el):
         #'nan' if self.is_nan(el) else
        return  el.item().strftime(self._form) if isinstance(el, np.datetime64) else self.timedelta64_to_string(el) if isinstance(el, np.timedelta64) else str(round(el, 2)) if is_number(el) else str(el)

    def _to_strings(self, data):
        return np.array([self._to_string(el) for el in data])

    def __str__(self):
        return self._tabulate_info() + nl + 0 * sp + self._get_sample_data()

    def __repr__(self):
        return str(self)

    def __getitem__(self, row):
        return self._data[row]
    
    def __setitem__(self, rows, value):
        rows = np.array(rows)
        rows = [r for r in self._Rows if rows[r]] if rows.dtype == np.bool_ else rows
        value = value._data if isinstance(value, data_class) else value if isinstance(value, list) or isinstance(value, np.ndarray) else [value] * self._rows
        for r in rows:
            self._data[r] = value[r]
        self._clear()

    def _clear(self):
        pass
        #self.numerical_info.cache_clear()

    def bins(self, bins = None, nan = True, string = False, width = 1, population = 0):
        bins = self.distinct(nan = nan) if bins is None else bins
        if self.is_countable():
            m, M = self.min(), self.max()
            center = linspace(m, M, bins)
            center = [self._to_string(el) for el in center] if string else center
            bins = get_bins(m, M, bins, width)
            nan = nan and self.count_nan() != 0
            counts = [self.between(b[0], b[1]) for b in bins]
            counts = {center[i]: counts[i] for i in range(len(counts))}
            counts.update({'nan': self.is_nan()}) if nan else None #???
            counts = {center: counts[center] for center in counts.keys() if np.count_nonzero(counts[center]) >= population}
        else:
            center = self.unique(nan = nan, string = string)
            counts = [self.equal(c) for c in center]
            counts = {center[i]: counts[i] for i in range(len(counts))}
            counts = {center: counts[center] for center in counts.keys() if np.count_nonzero(counts[center]) >= population}
            counts = dict(sorted(counts.items(), key = lambda el: np.count_nonzero(el[1]), reverse = True)[ : bins])

        return counts


    def equal(self, value):
        value = self._correct_value(value)
        rows = self._are_nan(self._data) if self.is_nan(value) else self._data == value
        #rows = [r for r in self._Rows if rows[r]]
        return rows

    def not_equal(self, value):
        value = self._correct_value(value)
        rows = self._are_not_nan(self._data) if self.is_nan(value) else self._data != value
        #rows = [r for r in self._Rows if rows[r]]
        return rows

    def _are_nan(self, data):
        return  np.array([self.is_nan(el) for el in data], dtype = np.bool_)
    
    def _are_not_nan(self, data):
        return  np.array([self.is_not_nan(el) for el in data], dtype = np.bool_)

    def is_nan(self, value = None):
        return self._are_nan(self._data) if value is None else is_nan(value, self._nans)
    
    def is_not_nan(self, value = None):
        return self._are_not_nan(self._data) if value is None else not is_nan(value, self._nans)
    


    def higher(self, value, equal = False):
        value = self._correct_value(value)
        return self._data >= value if equal else self._data > value

    def lower(self, value, equal = False):
        value = self._correct_value(value)
        return self._data <= value if equal else self._data < value

    def between(self, start, end, equal1 = True, equal2 = False):
        return self.higher(start, equal1) & self.lower(end, equal2)

    def _correct_value(self, value):
        return self.string_to_datetime64(value) if isinstance(value, str) and self.is_datetime() else value


    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new._set_name(self._name)
        new._set_type(self._type)
        new._set_index(self._index)
        new.set_nans(self._nans)
        return new

    def part(self, a = None, b = None):
        a = 0 if a is None else a
        b = self._rows if b is None else b
        return self.subset(np.arange(a, b))
    
    def select(self, rows):
        new = self.empty()
        new._set_data(self.get_section(rows))
        return new

    def argsort(self):
        return np.argsort(self._data)

    def sort(self, rows):
        self._data = self._data[rows]

    def tolist(self):
        return self._data.tolist()

    def div(self, k):
        new = self.copy()
        new._data /= k
        new._clear()
        new.rename(self._name + ' / ' + str(k))
        return new

    def divide(self, other):
        new = self.copy()
        new._data /= other._data
        new._clear()
        new.rename(self._name + ' / ' + other._name)
        return new

    def sub(self, k):
        data = self._data - k
        data = self.timedelta64_to_numbers(data) if self.is_datetime() else data
        new = data_class(data, self._name)
        new._set_type('numerical')
        new._set_name('(' + self._name + ' - ' + str(k) + ')')
        return new

    def add(self, k):
        data = self._data + k
        new = data_class(data, self._name)
        new._set_type('numerical')
        new._set_name('(' + self._name + ' + ' + str(k) + ')')
        return new

    def subtract(self, other):
        data = self._data  - other._data
        data = self.timedelta64_to_numbers(data) if self.is_datetime() else data
        new = data_class(data, self._name)
        new._set_type('numerical')
        new._set_name('(' + self._name + ' - ' + other._name + ')')
        return new

    def invert(self):
        new = self.copy()
        new._data *= -1
        return new

    def power(self, p):
        data = self._data ** p
        new = data_class(data, self._name)
        new._set_type('numerical')
        new._set_name('(' + self._name + '^' + str(p) + ')')
        return new

    def set(self, other):
        self.set_data(other._data)
        self._name = self._name
        self._type = self._type
        self._clear()

    def fill(self, value):
        self._data.fill(value)
        self._clear()

    def set_data(self, data):
        self._data = np.array(data)

    def order(self):
        return ordering(self._data)

    def rescale(self, method = 'std'):
        if method == 'std':
            self._data = (self._data - self.mean()) / self.std()
        elif method == 'span':
            self._data = (self._data - self.min()) / self.span()
        elif method == 'robust':
            if self.iqr() == 0:
                print(self._name, 'rescaled with span method')
                self.rescale('span')
            else:
                self._data = (self._data - self.median()) / self.iqr()


    def __eq__(self, data):
        if isinstance(data, data_class):
            return (self._data == data._data) | (self.is_not_nan() & data.is_nan())
        else:
            return self._data == data

    def __lt__(self, data):
        if isinstance(data, data_class):
            return (self._data < data._data) #| (self.is_nan() | data.is_nan())
        else:
            return self._data < data
        
        return self._data < col._data

    def __gt__(self, data):
        if isinstance(data, data_class):
            return (self._data > data._data) #| (self.is_nan() | data.is_nan())
        else:
            return self._data > data
        
        return self._data < col._data

    def __le__(self, col):
        return (self < col)  | (self == col)

    def __ge__(self, col):
        return (self > col)  | (self == col)

    def __hash__(self):
        return hash(tuple(self._data))

    def __len__(self):
        return self._rows

    def __setitem__(self, el, value):
        self._data[el] = value
        

    def find(self, pos):
        arg_pos = np.where(self._order == pos)[0][0]
        indexes = [self._order[i] for i in range(max(0, arg_pos - 5), min(arg_pos + 5, self._rows))]
        indexes.remove(pos)
        return indexes
        # a = self._data[i]
        # b = self._data[j]
        # return nan if is_nan(a) or is_nan(b) else int(a == b) if self.is_categorical() else timedelta64_to_number(abs(a - b), self._delta_form) if self.is_datetime() else abs(a - b)

    # def standard_scale(self):
    #     data = self._data.reshape(-1, 1)
    #     self._data = ss().fit(data).transform(data).reshape(-1)


def get_bins(a, b, bins = 10, width = 1):
    bin_width = width * (b - a) / (bins - 1) if b != a else 1
    #start, end = a - bin_width / 2, b + bin_width / 2
    edge = linspace(a, b, bins)
    return [[edge[i] - bin_width / 2 , edge[i] + bin_width / 2] for i in range(bins)]

def linspace(start, end, points):
    bins = points - 1
    delta = (end - start) / bins
    return np.arange(start, end + delta, delta) if delta != 0 else [start, end]

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

def random_datetime64(mean, std, form, delta_form):
    mean = string_to_datetime64(mean, form).item().timestamp()
    index = forms.index(delta_form)
    std = std * div[index]
    res = random.normalvariate(mean, std)
    return dt.datetime.fromtimestamp(res).strftime(form)
