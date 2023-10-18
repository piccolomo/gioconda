from bongo._methods import *

class data_class():
    def __init__(self, data = None, type = 'none'):
        self.set_data(data, type = 'none')
        self.set_name()
        self.set_type(type)
        self.set_index()

    def set_data(self, data = None, type = 'none'):
        data = [] if data is None else data
        self.data = np.array(data, dtype = self.get_numpy_type(type))
        self.update_length()

    def get_numpy_type(self, type):
        return object if type == 'none' else '<U3' if type == 'categorical' else np.float64 if type == 'numerical' else np.datetime64 if type == 'datetime' else None
        
    def update_length(self):
        self.rows = len(self.data)
        self.Rows = np.arange(self.rows)

    def set_name(self, name = 'none'):
        self.name = name

    def set_type(self, type = 'none'):
        self.type = type

    def set_index(self, index = 'none'):
        self.index = index

    def get(self, row, string = False):
        el = self.data[row]
        el = self.to_string(el) if string else el
        return el

    def to_string(self, el):
        return str(el)

    def to_strings(self, data):
        return np.vectorize(self.to_string)(data)

    def get_section(self, rows = None, nan = True, string = False):
        rows = self.correct_rows(rows)
        data = self.data[rows]
        data = data if nan else data[~are_nan(data)]
        data = self.to_strings(data) if string else data
        return data

    def correct_rows(self, rows):
        return np.array([row for row in rows if row in self.Rows]) if isinstance(rows, list) else self.Rows if rows is None else rows

    def equal(self, value):
        return are_nan(self.data) if is_nan(value) else self.data == value

    def not_equal(self, value):
        return ~ self.equal(value)

    def counts(self, norm = False):
        c =  Counter(self.data)
        t = c.total()
        c = {key: (100 * count / t) for key, count in c.items()} if norm else c
        return c

    def unique(self, nan = True):
        return np.array(list(self.counts().keys()), dtype = self.data.dtype)

    def distinct(self, nan = True):
        return len(self.unique(nan))

    def count(self, value, norm = False):
        return self.counts(norm)[value]

    def count_nan(self, norm = False):
        return self.count('nan', norm) + self.count(np.nan, norm)

    def tabulate_counts(self, norm = False, length = 10):
        header = [self.name, 'count']
        counts = list(self.counts(norm = norm).items())
        table = tabulate(counts, header = header) + nl
        return table

    def print_counts(self, norm = False, length = 10):
        print(self.tabulate_counts(norm, length))

    def basic_info(self):
        return {'name': self.name, 'index': self.index, 'type': self.type, 'rows': self.rows, 'nan': self.count_nan(), 'unique': self.distinct()}

    def tabulate_basic_info(self):
        info = self.basic_info()
        table = [info.keys(), info.values()]
        table = tabulate(transpose(table))
        return table + nl

    def print_basic_info(self):
        print(self.tabulate_basic_info())

    def get_sample_data(self, length = 6):
        m = min(self.rows, length)
        start = [self.to_string(self.data[i]) for i in range(0, m)]
        end = [self.to_string(self.data[i]) for i in range(-m, 0)]
        out = ', '.join(start)
        out += ' ... ' if self.rows > length else ''
        out += ', '.join(end) if self.rows > 2 * length else ''
        return out

    def __str__(self):
        return self.tabulate_basic_info() + nl + sp + self.get_sample_data()

    def __repr__(self):
        return str(self)

    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new.set_name(self.name)
        new.set_type(self.type)
        new.set_index(self.index)
        return new

    def subset(self, rows):
        new = self.empty()
        new.set_data(self.get_section(rows))
        return new

    def part(self, a = None, b = None):
        a = 0 if a is None else a
        b = self.rows if b is None else b
        return self.subset(np.arange(a, b))


    def apply(self, function, *args):
        data = np.vectorize(function, *args)(self.data)
        self.set_data(data)

    def to_categorical(self):
        data = self.to_strings(self.data)
        self.set_data(data, 'categorical')
        self.set_type('categorical')

    def to_numerical(self):
        data = self.to_numbers(self.data)
        self.set_data(data, 'numerical')
        self.set_type('numerical')

    def to_numbers(self, data):
        return np.vectorize(np.float64)(data)

    def to_type_none(self):
        self.set_data(self.data, 'none')
        self.set_type('none')

    def to_datetime(self, form = '%d/%m/%Y', delta_form = 'years'):
        data = self.to_datetimes(self.data, form)
        self.set_data(data, 'datetime')
        self.set_type('datetime')
        self.set_forms(form, delta_form)

    def to_datetimes(self, data, form):
        data = np.vectorize(dt.strptime)(data, form)
        data = np.vectorize(np.datetime64)(data)
        return data
    
    def set_forms(self, form, delta_form):
        self.form = form
        self.delta_form = delta_form

        
    def is_categorical(self):
        return self.type == 'categorical'

    def is_numerical(self):
        return self.type == 'numerical'
    
    def is_datetime(self):
        return self.type == 'datetime'
    
    def is_not_categorical(self):
        return not self.is_categorical()

    
class categorical_data_class(data_class):
    def __init__(self, data):
        self.set_data(data)
        self.set_name()
        self.set_type('categorical')
        self.set_index()

    def set_data(self, data = None):
        data = [] if data is None else data
        self.data = np.array(data, dtype='<U3')
        self.update_length()

#     def strip(self):
#         self.apply(lambda string: string.strip())

#     def replace(self, old, new):
#         self.apply(lambda string: string.replace(old, new))

        
#     def to_float(self):
#         to_float = lambda el: float(el) if el != n else el
#         data = np.vectorize(to_float, *args)(self.data)
#         new = numerical_data_class(data)
#         new.set_name(self.name)
#         new.set_index(self.index)
#         return new
    
#     def to_integer(self):
#         to_int = lambda el: int(float(el)) if el != n else el
#         data = np.vectorize(to_int, *args)(self.data)
#         new = numerical_data_class(data)
#         new.set_name(self.name)
#         new.set_index(self.index)
#         return new
        
#     def to_datetime(self, form, form_delta):
#         data = np.array(self.data, dtype = 'datetime64')
#         new = datetime_data_class(data, form, form_delta)
#         new.set_name(self.name)
#         new.set_index(self.index)
#         return new



# class numerical_data_class(data_class):
#     def __init__(self, data = None):
#         super().__init__(data)
#         self.set_type('numerical')

#     def multiply(self, k):
#         self.set_data(k * self.data)

#     def round(self, decimals = 1):
#         fun = lambda el: np.round(el, decimals)
#         self.apply(fun)

#     def min(self):
#         return np.min(self.get(nan = False), default = n)
    
#     def max(self):
#         return np.max(self.get(nan = False), default = n)

#     def span(self):
#         m, M = self.min(), self.max()
#         return M - m if n not in [m, M] else n

#     def std(self):
#         data = self.get(nan = False); r = len(data)
#         return np.std(data) if r > 0 else n
    
#     def density(self):
#         span = self.span()
#         return 100 * self.std() / span

#     def mean(self):
#         data = self.get(nan = False); r = len(data)
#         return np.mean(data) if r > 0 else n

#     def median(self):
#         data = self.get(nan = False); r = len(data)
#         return np.median(data) if r > 0 else n
    
#     def mode(self):
#         data = self.get(nan = False); r = len(data)
#         return np.mode(data) if r > 0 else n

#     def numerical_info(self, string = False):
#         return {'name': self.name, 'min': self.min(), 'max': self.max(), 'span': self.span(), 'nan': self.nan(1), 'mean': self.mean(), 'median': self.median(), 'mode': self.mode(), 'std': self.std(), 'density': self.density()}

#     def info(self, string = False):
#         info = super().basic_info()
#         info.update(self.numerical_info(string = string))
#         return info

#     def to_string(self, el):
#         return el if isinstance(el, str) else str(round(el, 1))

    
#     def greater(self, value, equal = True):
#         data = data_to_string(self.data) if isinstance(value, str) else data
#         return data > value or (data == value and equal)
    
#     def lower(self, value, equal = True):
#         return [i for i in self.Rows if i not in self.greater(value, not equal)]
    


# class datetime_data_class(numerical_data_class):
#     def __init__(self, data, form, form_delta):
#         super().__init__(data)
#         self.set_type('datetime')

#     def std(self):
#         data = self.get(nan = False); r = len(data)
#         return np.std(data) if r > 0 else n
    
#     def density(self):
#         span = self.span()
#         return n if span == n else 100 * self.std() / span if span != 0 else 'inf'

#     def mean(self):
#         data = self.get(nan = False); r = len(data)
#         return np.mean(self.get(nan = False)) if r > 0 else n

#     def median(self):
#         data = self.get(nan = False); r = len(data)
#         return np.median(self.get(nan = False)) if r > 0 else n
    
#     def mode(self):
#         data = self.get(nan = False); r = len(data)
#         return np.mode(self.get(nan = False)) if r > 0 else n

#     def numerical_info(self, string = False):
#         info = super().numerical_info(False)
#         info = {el : self.to_string(info[el]) for el in info} if string else info
#         return info

#     def info(self, string = False):
#         info = super().info(False)
#         info = {el : self.to_string(info[el]) for el in info} if string else info
#         return info

#     def to_string(self, el):
#         return np.datetime64(el, form = self.form) if isinstance(el, np.datetime64) else timedelta_to_string(el, self.form_delta) if isinstance(el, np.timedelta64) else el if isinstance(el,str) else str(round(el, 1))

#     def empty(self):
#         new = self.__class__([], self.form, self.form_delta)
#         new.set_type(self.type)
#         new.set_name(self.name)
#         new.set_index(self.index)
#         return new

#     def apply(self, function, *args):
#         data = np.vectorize(function, *args)(data)
#         self.set_data(data)
        
#     def data_to_string(self, data):
#         return np.vectorize(self.to_string, *args)(data)

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
