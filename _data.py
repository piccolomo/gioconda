from bongo._methods import *

class data_class():
    def __init__(self, data = None):
        self.set_data(data)

    def set_data(self, data = None):
        data = [] if data is None else data
        self.data = data
        self.update_length()
        
    def set_type(self, type):
        self.type = type

    def set_index(self, index):
        self.index = index


    def update_length(self):
        self.rows = len(self.data)
        self.Rows = list(range(self.rows))

        
    def set_name(self, name):
        self.name = str(name)

    def set_row(self, row, el):
        self.data[row] = el

    def correct_rows(self, rows):
        return correct_range(rows, self.rows) if is_list(rows) or rows is None else index_to_range(rows, self.rows)


    
    def get_element(self, row, string = False):
        el = self.data[row]
        return n if el == n else self.to_string(el) if string else el
        
    def get(self, rows = None, nan = True, string = False):
        rows = self.correct_rows(rows)
        return [self.get_element(row, string) for row in rows if nan or (not nan and self.get_element(row) != n)]

    
    def apply(self, function, *args):
        data = [function(el, *args) for el in self.data]
        self.set_data(data)

        
    def is_numerical(self):
        return self.type == 'numerical'
    
    def is_datetime(self):
        return self.type == 'datetime'
    
    def is_categorical(self):
        return self.type == 'categorical'

    def is_not_categorical(self):
        return not self.is_categorical()
    

    def count(self, n, norm = False):
        c = len(self.equal(n))
        return 100 * c / self.rows if norm and self.rows != 0 else c

    def nan(self, norm = False):
        return self.count(n, norm)

    def not_nan(self, norm = False):
        c = self.count(n, norm = norm)
        t = self.rows if norm else 1
        return t - c

    def counts(self, norm = False, nan = True):
        u = unique(self.get(nan = nan))
        v = [self.count(el, norm) for el in u]
        c = transpose([u, v])
        return sorted(c, key = lambda el: self.min() if isinstance(el[1], str) else el[1], reverse = True)

    
    def tabulate_counts(self, norm = False, nan = True, length = 10):
        header = [self.name, 'count']
        table = self.counts(norm = norm, nan = nan)[ : length]
        table = tabulate(table, header = header) + nl
        return table

    def print_counts(self, norm = False, nan = True, length = 10):
        print(self.tabulate_counts(norm, nan, length))

    def unique(self, nan = True):
        #return [el[0] for el in self.counts(norm = 0, nan = nan)]
        return unique(self.get(nan = nan))


    def equal(self, value):
        transform = lambda el: self.to_string(el) if isinstance(value, str) else el
        return [i for i in self.Rows if transform(self.data[i]) == value]

    def not_equal(self, value):
        return [i for i in self.Rows if i not in self.equal(value)]
    

    def select(self, value, data):
        new = data.empty()
        rows = self.equal(value)
        new.set_data(data.get(rows))
        return new

    def cross_count(self, value, data, norm = False):
        s = self.select(value, data)
        c = s.not_nan() if s.is_categorical() else s.mean()
        return 100 * c / self.rows if norm else c

    def cross_counts(self, data, norm = False, nan = True):
        u = self.unique(nan)
        v = [self.cross_count(el, data, norm) for el in u]
        c = transpose([u, v])
        return sorted(c, key = lambda el: data.min() if isinstance(el[1], str) else el[1], reverse = True)

    def cross_unique(self, data, nan = True):
        return [el[0] for el in self.cross_counts(data, 0, nan)]

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
        new.set_data([self.data[i] for i in rows])
        return new

    def part(self, a = None, b = None):
        return self.subset(range(a, b))

    def basic_info(self):
        unique = len(self.unique(nan = True))
        return {'name': self.name, 'index': self.index, 'type': self.type, 'rows': self.rows, 'nan': self.nan(), 'unique': unique}

    def __str__(self):
        info = self.basic_info()
        table = [info.keys(), info.values()]
        table = tabulate(transpose(table))
        return table + nl

    def __repr__(self):
        return str(self)

    def to_string(self, el):
        return str(el)


class categorical_data_class(data_class):
    def __init__(self, data):
        super().__init__(data)
        self.set_type('categorical')

    def strip(self):
        self.apply(lambda string: string.strip())

    def replace(self, old, new):
        self.apply(lambda string: string.replace(old, new))

        
    def to_float(self):
        to_float = lambda el: float(el) if el != n else el
        data = list(map(to_float, self.data))
        new = numerical_data_class(data)
        new.set_name(self.name)
        new.set_index(self.index)
        return new
    
    def to_integer(self):
        to_int = lambda el: int(float(el)) if el != n else el
        data = list(map(to_int, self.data))
        new = numerical_data_class(data)
        new.set_name(self.name)
        new.set_index(self.index)
        return new
        
    def to_datetime(self, form, form_delta):
        to_date = lambda el: dt.datetime.strptime(el, form).replace(tzinfo = dt.timezone.utc) if el != n else el
        data = list(map(to_date, self.data))
        new = datetime_data_class(data, form, form_delta)
        new.set_name(self.name)
        new.set_index(self.index)
        return new



class numerical_data_class(data_class):
    def __init__(self, data = None):
        super().__init__(data)
        self.set_type('numerical')

    def multiply(self, k):
        data = [el * k if el != n else n for el in self.data]
        self.set_data(data)

    def round(self, decimals = 1):
        fun = lambda el: round(el, decimals) if el != n else el
        self.apply(fun)

    def min(self):
        return min(self.get(nan = False), default = n)
    
    def max(self):
        return max(self.get(nan = False), default = n)

    def span(self):
        m, M = self.min(), self.max()
        return M - m if n not in [m, M] else n

    def std(self):
        data = self.get(nan = False); r = len(data)
        return std(data) if r > 0 else n
    
    def density(self):
        span = self.span()
        return n if span == n else 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        data = self.get(nan = False); r = len(data)
        return mean(data) if r > 0 else n

    def median(self):
        data = self.get(nan = False); r = len(data)
        return median(data) if r > 0 else n
    
    def mode(self):
        data = self.get(nan = False); r = len(data)
        return mode(data) if r > 0 else n

    def numerical_info(self, string = False):
        return {'name': self.name, 'mean': self.mean(), 'std': self.std(), 'density': self.density(), 'median': self.median(), 'mode': self.mode(), 'min': self.min(), 'max': self.max(), 'span': self.span(), 'nan': self.nan(1)}

    def info(self, string = False):
        info = super().info()
        info.update(self.numerical_info(string = string))
        return info

    def to_string(self, el):
        return el if isinstance(el, str) else str(round(el, 1))

    
    def greater(self, value, equal = True):
        transform = lambda el: self.to_string(el) if isinstance(value, str) else el
        return [i for i in self.Rows if transform(self.data[i]) >= value]
    
    def lower(self, value, equal = True):
        return [i for i in self.Rows if i not in self.greater(value, not equal)]
    


class datetime_data_class(numerical_data_class):
    def __init__(self, data, form, form_delta):
        super().__init__(data)
        self.set_type('datetime')
        self.set_forms(form, form_delta)

    def set_forms(self, form, form_delta):
        self.form = form
        self.form_delta = form_delta

    def std(self):
        data = self.get(nan = False); r = len(data)
        return std_datetime(data) if r > 0 else n
    
    def density(self):
        span = self.span()
        return n if span == n else 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        data = self.get(nan = False); r = len(data)
        return mean_datetime(self.get(nan = False)) if r > 0 else n

    def median(self):
        data = self.get(nan = False); r = len(data)
        return median_datetime(self.get(nan = False)) if r > 0 else n
    
    def mode(self):
        data = self.get(nan = False); r = len(data)
        return mode_datetime(self.get(nan = False)) if r > 0 else n

    def numerical_info(self, string = False):
        info = super().numerical_info(False)
        info = {el : self.to_string(info[el]) for el in info} if string else info
        return info

    def info(self, string = False):
        info = super().info(False)
        info = {el : self.to_string(info[el]) for el in info} if string else info
        return info

    def to_string(self, el):
        return n if el == n else time_to_string(el, self.form) if isinstance(el, dt.datetime) else timedelta_to_string(el, self.form_delta) if isinstance(el, dt.timedelta) else el if isinstance(el,str) else str(round(el, 1))

    def empty(self):
        new = self.__class__([], self.form, self.form_delta)
        new.set_type(self.type)
        new.set_name(self.name)
        new.set_index(self.index)
        return new


