from bongo.data import *
from bongo.methods import *


class matrix_class():
    
    def __init__(self):
        pass

    def set_data(self, data):
        self.data = data
        self.update_dimensions()

    def set_matrix(self, matrix):
        self.m = matrix
        data = [categorical_data_class(data) for data in transpose(matrix)]
        self.set_data(data)
        self.set_names(self.Cols)

    def update_dimensions(self):
        self.cols = len(self.data)
        self.rows = self.get_col(0).rows if self.cols > 0 else 0
        self.Rows = list(range(self.rows))
        self.Cols = list(range(self.cols))

        
    def set(self, row, col, el):
        self.get_col(col).set(row, el)
        
    def set_names(self, names):
        [self.data[col].set_name(names[col]) for col in self.Cols]

    def set_col(self, col, data_obj):
        col = self.get_col_index(col)
        self.data[col] = data_obj
        
    def set_col_data(self, col, data):
        self.get_col(col).set_data(data)

    def set_type(self, col, type):
        self.get_col(col).set_type(type)

    def mul(self, col, k):
        c = self.get_col(col)
        c.mul(k) if c.is_numerical() else None

        
    def get_col_index(self, col):
        return self.get_names().index(col) if isinstance(col, str) else col

    def get_cols_indexes(self, cols = None):
        cols = self.Cols if cols is None else list(map(self.get_col_index, cols))
        return correct_range(cols, self.cols)

    
    def get_col(self, col):
        col = self.get_col_index(col)
        return self.data[col]
    col = get_col

    def get_name(self, col):
        return self.get_col(col).name
    
    def get_names(self, cols = None, index = False):
        names = ['i'] if index else []
        return names + [self.get_name(col) for col in self.get_cols_indexes(cols)]

    def print_cols(self):
        table = transpose([self.Cols, self.get_names(), self.get_types()])
        table = tabulate_data(table, grid = 0, headers = ['i', 'column', 'type'])
        print(table)

    def get_type(self, col):
        return self.get_col(col).type

    def get_types(self, cols = None):
        return [self.get_col(col).type for col in self.get_cols_indexes(cols)]


    def get_transpose(self):
        return transpose(self.get_section())

    def get(self, row, col):
        return self.get_col(col).get(row)
    
    def get_rows(self, col, rows = None, nan = True, string = False):
        return self.get_col(col).get_rows(rows, nan = nan, string = string)

    def get_section(self, rows = None, cols = None, index = False, string = False):
        data = transpose([self.get_rows(col, rows, string = string) for col in cols])
        data = [[rows[i]] + data[i] for i in range(len(rows))] if index else data
        return data


    def is_categorical(self, col):
        return self.get_col(col).is_categorical()
    
    def is_numerical(self, col):
        return self.get_col(col).is_numerical()
    
    def is_datetime(self, col):
        return self.get_col(col).is_datetime()
    

    def get_numerical_cols(self):
        return [self.get_name(col) for col in self.Cols if self.is_numerical(col)]
    
    def get_datetime_cols(self):
        return [self.get_name(col) for col in self.Cols if self.is_datetime(col)]
    
    def get_categorical_cols(self):
        return [self.get_name(col) for col in self.Cols if self.is_categorical(col)]


    def strip_col(self, col):
        self.get_col(col).strip()

    def replace_col(self, col, old, new):
        self.get_col(col).replace(old, new)

    def strip(self):
        [self.strip_col(col) for col in self.Cols]

    def replace(self, old, new):
        [self.replace_col(col, old, new) for col in self.Cols]
    

    def to_float(self, col):
        self.set_col(col, self.get_col(col).to_float())
        
    def round(self, col, decimals = 1):
        self.get_col(col).round(decimals)
        
    def to_integer(self, col):
        self.set_col(col, self.get_col(col).to_integer())
        
    def to_datetime(self, col, form, delta_form):
        self.set_col(col, self.get_col(col).to_datetime(form, delta_form))

    # def apply(self, col, function, *args):
    #     self.get_col(col).apply(function, *args)

    def unique(self, col, nan = True):
        return self.get_col(col).unique(nan = nan)
    
    def counts(self, col, normalize = True):
        return self.get_col(col).counts(normalize)

    def describe_numerical(self):
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density']
        info = [[self.get_col_index(col)] + self.get_col(col).get_info() for col in self.get_numerical_cols()]
        table = tabulate_data(info, headers = cols, decimals = 1)
        print(table)

    def describe_datetime(self, form = 'days'):
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density']
        info = [[self.get_col_index(col)] + self.get_col(col).get_info(string = True) for col in self.get_datetime_cols()]
        table = tabulate_data(info, headers = cols, decimals = 1)
        print(table)

    def describe_categorical(self, normalize = 0, cols = None, rows = 10):
        cols = self.correct_cols(cols)
        [self.get_col(col).print_info(normalize, rows = rows) for col in cols if self.is_categorical(col)]


    def plot(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        x = c1.get_index(data = c2) if c1.is_categorical() else c1.get_index()
        y = c2.get_index(data = c1) if c2.is_categorical() else c2.get_index()
        xy = [el for el in transpose([x, y]) if n not in el]
        x, y = transpose(xy)
        plt.clf()
        plt.scatter(x, y)
        plt.xticks(*c1.get_ticks())
        plt.yticks(*c2.get_ticks())
        plt.xlabel(c1.name)
        plt.ylabel(c2.name)
        plt.show()
        
    def categorical_tab(self, col1, col2, length = 5, norm = True, log = False):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        unique1 = c1.cross_unique(c2)
        unique2 = c2.cross_unique(c1)
        counts = [[self.where(u1, col1).where(u2, col2).rows for u2 in unique2] for u1 in unique1]
        counts = [normalize(el) for el in counts] if norm else counts
        corr = [correlate(el) for el in counts]
        table = [[unique1[i]] + counts[i][:length] + [corr[i]] for i in range(len(counts))]
        header = [c1.name + ' / ' + c2.name] + unique2[:length] + ['corr']
        table = tabulate_data(table, headers = header, grid = True, decimals = 1)
        print(table) if log else None
        corr = mean(corr)
        print('mean correlation', round(corr, 1), '%')  if log else None
        return counts
        
    def correlate_categorical(self, col1, col2):
        return cramers(self.categorical_tab(col1, col2))
    
    def correlate_numerical(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        return correlate_numerical(c1.get_rows(), c2.get_rows())

    def correlate_categorical_to_numerical(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        i1 = c1.get_index(c2)
        i2 = c2.get_index()
        return correlate_numerical(i1, i2)

    def correlate(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        cc = c1.is_categorical() and c2.is_categorical()
        cn = c1.is_categorical() and not c2.is_categorical()
        nc = not c1.is_categorical() and c2.is_categorical()
        nn = not c1.is_categorical() and not c2.is_categorical()
        return self.correlate_categorical(col1, col2) if cc else self.correlate_numerical(col1, col2) if nn else self.correlate_categorical_to_numerical(col1, col2) if cn else self.correlate_cat_to_num(col2, col1)


    def correct_cols(self, cols):
        return self.get_cols_indexes(cols) if is_list(cols) or cols is None else index_to_range(self.get_col_index(cols), self.cols)
        
    def correct_rows(self, rows):
        return correct_range(rows, self.rows) if is_list(rows) or rows is None else index_to_range(rows, self.rows)


    def copy(self):
        return copy(self)

    def subset(self, rows = None):
        rows = self.correct_rows(rows)
        rows = correct_range(rows, self.rows) if is_list(rows) else index_to_range(rows, self.rows)
        new = matrix_class()
        data = [data.subset(rows) for data in self.data]
        new.set_data(data)
        #new.set_names(self.get_names())
        return new
        
    def part(self, start = None, end = None):
        start = correct_left_index(start, self.rows)
        end = correct_right_index(end, self.rows)
        return self.subset(range(start, end))

    def where(self, value, col):
        return self.subset(self.get_col(col).where(value))


    def tabulate_data(self, header = True, index = False, rows = None, cols = None, grid = False, decimals = 1):
        return tabulate_data(self.get_section(rows, cols, index, 1), headers = self.get_names(cols, index) if header else [], decimals = decimals, grid = grid)
        print(table)

    def tabulate_types(self, rows = None, cols = None, grid = False):
        return tabulate_data([self.get_cols_indexes(cols), self.get_types(cols)], headers = self.get_names(cols), grid = False)
        print(table)

    def tabulate_dimensions(self):
        return tabulate_data([[self.rows, self.cols]], headers = ['rows', 'cols'], grid = 1)

    def tabulate(self, header = True, index = False, info = True, rows = None, cols = None, grid = False, decimals = 1):
        rows = self.correct_rows(rows)
        cols = self.correct_cols(cols)
        table = self.tabulate_data(header, index, rows, cols, grid, decimals)
        table = table + nl + self.tabulate_types(rows, cols, grid) if info else None
        table = table + nl + self.tabulate_dimensions() if info else None
        return table

    def print(self, header = True, index = True, info = True, rows = None, cols = None, grid = False, decimals = 1):
        print(self.tabulate(header, index, info, rows, cols, grid, decimals))

    def __repr__(self):
        rows, cols = 10, 3
        rows = list(range(rows)) + list(range(-rows, 0))
        cols = list(range(cols)) + list(range(-cols, 0))
        return self.tabulate(1, 1, 1, rows, cols)
