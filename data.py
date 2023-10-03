from bongo.methods import *

class data_class():
    def __init__(self, data):
        self.set_data(data)

    def set_data(self, data):
        self.data = data
        self.update_length()
        
    def set_type(self, type):
        self.type = type


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
        return [self.get_element(row, string) for row in rows if nan or (not nan and self.get(row) != n)]

    
    def apply(self, function, *args):
        data = [function(el, *args) for el in self.data]
        self.set_data(data)

        
    def is_numerical(self):
        return self.type == 'numerical'
    
    def is_datetime(self):
        return self.type == 'datetime'
    
    def is_categorical(self):
        return self.type == 'categorical'
    

    def where(self, value):
        rows = [i for i in self.Rows if self.data[i] == value]
        return rows

    def count(self, n):
        return len(self.where(n))

    def nan(self):
        return self.count(n)

    def not_nan(self):
        return self.rows - self.count(n)

    def counts(self, norm = False, nan = True):
        u = unique(self.get_rows(nan = nan))
        v = [self.count(el) for el in u]
        v = normalize(v) if norm else v
        c = transpose([u, v])
        return sorted(c, key = lambda el: el[1], reverse = True)

    def unique(self, nan = True):
        return [el[0] for el in self.counts(0, nan)]
        #return unique(self.get_rows(nan = nan, string = string))

    def select(self, el, data):
        new = data.empty()
        new_data = [data.data[row] for row in self.Rows if self.data[row] == el]
        new.set_data(new_data);
        return new

    def cross_counts(self, data, norm = False, nan = True):
        u = self.unique(nan)
        v = [self.select(el, data) for el in u]
        v = [el.not_nan() for el in v] if data.is_categorical() else [el.mean() for el in v] 
        v = normalize(v) if norm else v
        c = transpose([u, v])
        return sorted(c, key = lambda el: el[1], reverse = True)

    def cross_unique(self, data, nan = True):
        return [el[0] for el in self.cross_counts(data, 0, nan)]

    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new.set_name(self.name)
        new.set_type(self.type)
        return new

    def subset(self, rows):
        new = self.empty()
        new.set_data([self.data[i] for i in rows])
        return new

    def part(self, a = None, b = None):
        return self.subset(range(a, b))

    def __str__(self):
        out   =      'name    ' + self.name
        out  += nl + 'type    ' + self.type
        out  += nl + 'rows    ' + str(self.rows)
        out  += nl + 'nan     ' + str(self.nan()) 
        out  += nl + 'unique  ' + str(len(self.unique()))
        out  += nl + 'firsts  ' + ', '.join(self.get_rows(5, string = 1))
        out  += nl + 'lasts   ' + ', '.join(self.get_rows(-5, string = 1))
        return out

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
        return new
    
    def to_integer(self):
        to_int = lambda el: int(float(el)) if el != n else el
        data = list(map(to_int, self.data))
        new = numerical_data_class(data)
        new.set_name(self.name)
        return new
        
    def to_datetime(self, form, form_delta):
        to_date = lambda el: dt.datetime.strptime(el, form).replace(tzinfo = dt.timezone.utc) if el != n else el
        data = list(map(to_date, self.data))
        new = datetime_data_class(data, form, form_delta)
        new.set_name(self.name)
        return new
    

    def print_info(self, normalize = 1, rows = 10):
        cols = [self.name, '']
        table = self.counts(normalize, rows)[:rows]
        table = tabulate_data(table, headers = cols, decimals = 1)
        print(table)

    def get_index(self, data = None):
        u = self.unique() if data is None else self.cross_unique(data)
        return [u.index(el) if el != n else n for el in self.data]

    def get_ticks(self):
        u = self.unique(nan = 0)
        return (range(len(u)), u)


class numerical_data_class(data_class):
    def __init__(self, data):
        super().__init__(data)
        self.set_type('numerical')

    def mul(self, k):
        data = [el * k if el != n else n for el in self.data]
        self.set_data(data)

    def round(self, decimals = 1):
        fun = lambda el: round(el, decimals) if el != n else el
        self.apply(fun)

    def min(self):
        return min(self.get_rows(nan = False), default = n)
    
    def max(self):
        return max(self.get_rows(nan = False), default = n)

    def span(self):
        return self.max() - self.min() if self.rows > 0 else n

    def std(self):
        return std(self.get_rows(nan = False)) if self.rows > 0 else n
    
    def density(self):
        span = self.span()
        return n if span == n else 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        return mean(self.get_rows(nan = False)) if self.rows > 0 else n

    def median(self):
        return median(self.get_rows(nan = False)) if self.rows > 0 else n
    
    def mode(self):
        return mode(self.get_rows(nan = False)) if self.rows > 0 else n

    def get_info(self, string = False):
        info = [self.name, self.mean(), self.median(), self.mode(), self.std(), self.span(), self.density()]
        info = [self.to_string(el) for el in info] if string else info
        return info

    def get_index(self):
        return self.data

    def get_ticks(self):
        t = list(np.linspace(self.min(), self.max(), 5))
        return t, t

    def to_string(self, el):
        return str(round(el, 1))
    
    def __str__(self):
        out = super().__str__()
        out  += nl + 'min     ' + self.to_string(self.min())
        out  += nl + 'max     ' + self.to_string(self.max())
        out  += nl + 'mean    ' + self.to_string(self.mean())
        out  += nl + 'std     ' + self.to_string(self.std())
        return out



class datetime_data_class(numerical_data_class):
    def __init__(self, data, form, form_delta):
        super().__init__(data)
        self.set_type('datetime')
        self.set_forms(form, form_delta)

    def set_forms(self, form, form_delta):
        self.form = form
        self.form_delta = form_delta

    def std(self):
        return std_datetime(self.get_rows(nan = False)) if self.rows > 0 else n
    
    def density(self):
        span = self.span()
        return n if span == n else 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        return mean_datetime(self.get_rows(nan = False)) if self.rows > 0 else n

    def median(self):
        return median_datetime(self.get_rows(nan = False)) if self.rows > 0 else n
    
    def mode(self):
        return mode_datetime(self.get_rows(nan = False)) if self.rows > 0 else n

    def get_info(self, string = False):
        info = super().get_info(False)
        info = [self.to_string(el) for el in info] if string else info
        return info

    def to_string(self, el):
         return n if el == n else time_to_string(el, self.form) if isinstance(el, dt.datetime) else timedelta_to_string(el, self.form_delta) if isinstance(el, dt.timedelta) else str(el)

    def get_index(self):
        ref = self.min()
        return [timedelta_to_number(el - ref, self.form_delta) if el != n else n for el in self.data]

    def get_ticks(self):
        t = linspace(0, timedelta_to_number(self.span(), self.form_delta), 5)
        l = datetime_linspace(self.min(), self.max(), 5)
        l = [self.to_string(el)  for el in l]
        return t, l

    def empty(self):
        new = self.__class__([], self.form, self.form_delta)
        new.set_type(self.type)
        new.set_name(self.name)
        return new

    def __str__(self):
        return super().__str__() + sp + self.form_delta



