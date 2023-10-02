from utility.methods import *

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
        
    def get(self, row, string = False):
        el = self.data[row]
        return self.to_string(el) if string else el
        
    def get_rows(self, rows = None, nan = True, string = False):
        rows = self.correct_rows(rows)
        return [self.get(row, string) for row in rows if nan or (not nan and self.get(row) != n)]

    
    def apply(self, function, *args):
        data = [function(el, *args) for el in self.data]
        self.set_data(data)

        
    def is_numerical(self):
        return self.type == 'numerical'
    
    def is_datetime(self):
        return self.type == 'datetime'
    
    def is_categorical(self):
        return self.type == 'categorical'


    def count(self, el, fun = len, data = None):
        check_data = self.get_rows()
        count_data = check_data if data is None else data.get_rows()
        data = [count_data[i] for i in self.Rows if check_data[i] == el]
        data = [el for el in data if el != n] if fun != len else data
        return fun(data)

    def unique(self, nan = True, string = False, fun = len):
        c = self.counts(nan = nan, fun = fun, data = data)
        return unique(self.get_rows(nan = nan, string = string))

    def nan(self):
        return self.count(n)

    def counts(self, normalize = False, nan = True, fun = len, data = None):
        u = self.unique(nan)
        v = np.array([self.count(el, fun = fun, data = data) for el in u])
        v = 100 * v / sum(v) if normalize else v
        return sorted(zip(u, v), key = lambda el: el[1], reverse = True)

    
    def copy(self):
        return copy(self)

    def empty(self):
        new = self.__class__([])
        new.set_name(self.name)
        return new

    def subset(self, rows):
        new = self.empty()
        new.set_data([self.data[i] for i in rows])
        return new

    def part(self, a = None, b = None):
        return self.subset(range(a, b))

    def where(self, value):
        rows = [i for i in self.Rows if self.data[i] == value]
        return rows

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
        to_int = lambda el: int(el) if el != n else el
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
    

    def print_info(self, normalize = 1, rows = 10, fun = len):
        cols = [self.name, '']
        table = self.counts(normalize, rows, fun = len)[:rows]
        table = tabulate_data(table, headers = cols, decimals = 1)
        print(table)

    def get_index(self, data = None, fun = len):
        u = self.unique() if data is None else self.
        return [u.index(el) if el != n else n for el in self.data]

    def get_ticks(self):
        u = self.unique(nan = 0)
        return (range(len(u)), u)

    # def to_string(self, el):
    #     return el


class numerical_data_class(data_class):
    def __init__(self, data):
        super().__init__(data)
        self.set_type('numerical')

    def round(self, decimals = 1):
        fun = lambda el: round(el, decimals) if el != n else el
        self.apply(fun)

    def min(self):
        return min(self.get_rows(nan = False))
    
    def max(self):
        return max(self.get_rows(nan = False))

    def span(self):
        return self.max() - self.min()

    def std(self):
        return std(self.get_rows(nan = False))
    
    def density(self):
        span = self.span()
        return 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        return mean(self.get_rows(nan = False))

    def median(self):
        return median(self.get_rows(nan = False))
    
    def mode(self):
        return mode(self.get_rows(nan = False))

    def get_info(self, string = False):
        info = [self.name, self.mean(), self.median(), self.mode(), self.std(), self.span(), self.density(), self.nan()]
        info = [str(el) for el in info] if string else info
        return info

    # def to_string(self, el):
    #     return str(el)

    def get_index(self):
        return self.data

    def get_ticks(self):
        t = list(np.linspace(self.min(), self.max(), 5))
        return t, t



class datetime_data_class(numerical_data_class):
    def __init__(self, data, form, form_delta):
        super().__init__(data)
        self.set_type('datetime')
        self.set_forms(form, form_delta)

    def set_forms(self, form, form_delta):
        self.form = form
        self.form_delta = form_delta

    def std(self):
        return std_datetime(self.get_rows(nan = False))
    
    def density(self):
        span = self.span()
        return 100 * self.std() / span if span != 0 else 'inf'

    def mean(self):
        return mean_datetime(self.get_rows(nan = False))

    def median(self):
        return median_datetime(self.get_rows(nan = False))
    
    def mode(self):
        return mode_datetime(self.get_rows(nan = False))

    def get_info(self, string = False):
        info = super().get_info(False)
        info = [self.to_string(el) for el in info] if string else info
        return info

    def to_string(self, el):
         return n if el == n else el.strftime(self.form) if isinstance(el, dt.datetime) else timedelta_to_number(el, self.form_delta) if isinstance(el, dt.timedelta) else str(el)

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
        new.set_name(self.name)
        return new



