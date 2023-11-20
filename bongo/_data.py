from bongo._methods import *
from matplotlib import pyplot as plt

class data_class():
    def __init__(self, data = [], name = '', index = 'none'):
        self.set_data(data)
        self.set_name(name)
        self.set_type()
        self.set_index(index)
        self.set_forms()

    def set_data(self, data = []):
        self.data = np.array(data)
        self.update_length()

    def set_name(self, name = 'none'):
        self.name = name

    def set_type(self, type = 'categorical'):
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
    def counts(self, norm = False, nan = True):
        u, v = np.unique(self.data, return_counts = 1)
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
        c = np.count_nonzero(are_nan(self.data))
        return nan if self.rows == 0 else 100 * c / self.rows if norm else c

    @mem(maxsize=None)
    def unique(self, nan = True):
        return np.array(list(self.counts(False, nan).keys()), dtype = self.data.dtype)

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

    def to_numerical(self, dictionary = None):
        self.data = self.to_numbers(self.data, dictionary)
        self.update_length()
        self.set_type('numerical')
        return self

    def to_numbers(self, data, dictionary = None):
        return vectorize(np.float64, data) if dictionary is None else [dictionary[el] for el in data]

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

    def apply(self, function):
        data = vectorize(function, self.data)
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
        counts = list(self.counts(norm = norm).items())[: length]
        table = tabulate(counts, header = header) + nl
        return table

    def print_counts(self, norm = False, length = 10):
        print(self.tabulate_counts(norm, length))
        
    @mem(maxsize = None)
    def basic_info(self):
        return {'name': self.name, 'index': self.index, 'type': self.type, 'rows': self.rows, 'nan': self.count_nan(), 'unique': self.distinct(nan=False)}

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
        plt.xticks(rotation = 90) if self.is_categorical() else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()
    
    
    def tabulate_info(self):
        info = self.info()
        table = [list(info.keys()), list(info.values())]
        table = tabulate(np.transpose(table))
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

    def __getitem__(self, row):
        return self.data[row]


        

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
