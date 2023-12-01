from gioconda._methods import *
from matplotlib import pyplot as plt


class data_class():
    def __init__(self, data = [], name = '', index = 'none'):
        self._set_data(data)
        self._set_name(name)
        self._set_type()
        self._set_index(index)
        self._set_forms()

    def _set_data(self, data = []):
        self._data = np.array(data)
        self._update_length()

    def _set_name(self, name = 'none'):
        self._name = name


    def _set_type(self, type = 'categorical'):
        self._type = type

    def _set_index(self, index = 'none'):
        self._index = index

    def _set_forms(self, form = '%d/%m/%Y', delta_form = 'years'):
        self._form = form
        self._delta_form = delta_form
        
    def _update_length(self):
        self._rows = len(self._data)
        self._Rows = np.arange(self._rows)


    def get(self, row, string = False):
        el = self._data[row]
        el = self._to_string(el) if string else el
        return el

    def get_section(self, rows = None, nan = True, string = False):
        rows = self._correct_rows(rows)
        data = self._data[rows]
        data = data if nan else data[are_not_nan(data)]
        data = self._to_strings(data) if string else data
        return data
    
    def _correct_rows(self, rows):
        return np.array([row for row in rows if row in self._Rows]) if isinstance(rows, list) else self._Rows if rows is None else rows
    

    @mem(maxsize = None)
    def counts(self, norm = False, nan = True):
        u, v = np.unique(self._data, return_counts = 1)
        nans = are_nan(u)
        u = u if nan else u[~nans]
        v = v if nan else v[~nans]
        t = sum(v)
        c = {u[i] : v[i] for i in range(len(u))}
        c = {key: (100 * count / t) for key, count in c.items()} if norm else c
        return c

    def count(self, value, norm = False):
        counts = self.counts(norm)
        return counts[value] if value in counts else 0
        #return 100 * count / self.rows if norm else count
    
    def count_nan(self, norm = False):
        c = np.count_nonzero(are_nan(self._data))
        return nan if self._rows == 0 else 100 * c / self._rows if norm else c

    @mem(maxsize = None)
    def unique(self, nan = True):
        return np.array(list(self.counts(False, nan).keys()))

    def distinct(self, nan = True):
        return len(self.unique(nan))

    def mode(self):
        unique = self.unique()
        return unique[0] if len(unique) > 0 else nan


    def to_categorical(self):
        self._data = self._to_strings(self._data)
        self._update_length()
        self._set_type('categorical')
        return self

    def to_numerical(self, dictionary = None):
        self._data = self._to_numbers(dictionary)
        self._update_length()
        self._set_type('numerical')
        return self

    def _to_numbers(self, dictionary = None):
        if self.is_datetime():
            return  np.array([timedelta64_to_number(el, self._delta_form) if not is_nan(el) else nan for el in self._data - self.min()])
        elif dictionary is None:
            return np.array([np.float64(el) if not is_nan(el) else nan for el in self._data])
        else:
            return np.array([dictionary[el] if not is_nan(el) else nan for el in self._data])


    def to_datetime(self, form = '%d/%m/%Y', delta_form = 'years'):
        self._data = strings_to_datetime64(self._data, form)
        self._update_length()
        self._set_type('datetime')
        self._set_forms(form, delta_form)
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

    def replace(self, old, new):
        self._apply(lambda string: string.replace(old, new)) if self.is_categorical() else print('not categorical')

    def _apply(self, function):
        data = vectorize(function, self._data)
        self._set_data(data)

    def len(self):
        return self._rows

    def sum(self):
        data = self.get_section(nan = False); l = len(data)
        return nan if l == 0 or self.is_uncountable() else mean_datetime64(data) if self.is_datetime() else np.sum(data)
        
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

    def _get_numerical_data(self):
        data = self.get_section(nan = False)
        data = [el.item().timestamp() for el in data] if self.is_datetime() else data
        return data if self.is_countable() else []
    
    def multiply(self, k):
        self._set_data(k * self._data) if self.is_numerical() else print('not numerical')
        

    def _tabulate_counts(self, norm = False, length = 10):
        header = [self._name, 'count']
        counts = list(self.counts(norm = norm).items())[: length]
        table = tabulate(counts, header = header) + nl
        return table

    def _print_counts(self, norm = False, length = 10):
        print(self._tabulate_counts(norm, length))
        
    @mem(maxsize = None)
    def _basic_info(self):
        return {'name': self._name, 'index': self._index, 'type': self._type, 'rows': self._rows, 'nan': self.count_nan(), 'unique': self.distinct(nan=False)}

    @mem(maxsize=None)
    def numerical_info(self):
        info = {'min': self.min(), 'max': self.max(), 'span': self.span(), 'nan': self.count_nan(1), 'mean': self.mean(), 'median': self.median(), 'mode': self.mode(), 'std': self.std(), 'density': self.density()}
        return {k : self._to_string(info[k]) for k in info.keys()}
    
    def info(self):
        info = self._basic_info()
        info.update(self.numerical_info()) if self.is_countable() else None
        return info

    def plot(self, bins = 100):
        plt.figure(0, figsize = (15, 8)); plt.clf()
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
        return 'nan' if is_nan(el) else el.item().strftime(self._form) if isinstance(el, np.datetime64) else timedelta64_to_string(el, self._delta_form) if isinstance(el, np.timedelta64) else str(round(el, 2)) if is_number(el) else str(el)

    def _to_strings(self, data):
        return np.array([self._to_string(el) for el in data])

    def __str__(self):
        return self._tabulate_info() + nl + 0 * sp + self._get_sample_data()

    def __repr__(self):
        return str(self)

    def __getitem__(self, row):
        return self._data[row]


        

    def equal(self, value):
        data = self.get_section(string = 1) if isinstance(value, str) and self.is_datetime() else self._data
        return are_nan(self._data) if is_nan(value) else self._data == value

    def not_equal(self, value):
        return ~ self.equal(value)

    def higher(self, value, equal = False):
        return self._data >= value if equal else self._data > value

    def lower(self, value, equal = False):
        return ~self.higher(value, not equal)


    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new._set_name(self._name)
        new._set_type(self._type)
        new._set_index(self._index)
        return new

    def part(self, a = None, b = None):
        a = 0 if a is None else a
        b = self._rows if b is None else b
        return self.subset(np.arange(a, b))
    
    def subset(self, rows):
        new = self.empty()
        new._set_data(self.get_section(rows))
        return new

    def argsort(self):
        return np.argsort(self._data)

    def sort(self, rows):
        self._data = self._data[rows]

    def tolist(self):
        return self._data.tolist()
