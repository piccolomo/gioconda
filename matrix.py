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
        self.rows = self.col(0).rows if self.cols > 0 else 0
        self.Rows = list(range(self.rows))
        self.Cols = list(range(self.cols))

        
    def set(self, row, col, el):
        self.col(col).set(row, el)
        
    def set_names(self, names):
        [self.data[col].set_name(names[col]) for col in self.Cols]

    def set_col(self, col, data_obj):
        col = self.get_col_index(col)
        self.data[col] = data_obj
        
    def set_col_data(self, col, data):
        self.col(col).set_data(data)

    def set_type(self, col, type):
        self.col(col).set_type(type)

    def multiply(self, col, k):
        c = self.col(col)
        c.multiply(k) if c.is_numerical() else None


        
    def get_col_index(self, col):
        return self.names().index(col) if isinstance(col, str) else col

    def get_cols_indexes(self, cols = None):
        cols = self.Cols if cols is None else list(map(self.get_col_index, cols))
        return correct_range(cols, self.cols)

    def correct_cols(self, cols):
        return self.get_cols_indexes(cols) if is_list(cols) or cols is None else index_to_range(self.get_col_index(cols), self.cols)
        
    def correct_rows(self, rows):
        return correct_range(rows, self.rows) if is_list(rows) or rows is None else index_to_range(rows, self.rows)


    
    def col(self, col):
        col = self.get_col_index(col)
        return self.data[col]

    def name(self, col):
        return self.col(col).name
    
    def names(self, cols = None, index = False):
        names = ['i'] if index else []
        return names + [self.name(col) for col in self.get_cols_indexes(cols)]

    def print_cols(self):
        table = transpose([self.Cols, self.names(), self.types()])
        table = tabulate(table, headers = ['i', 'column', 'type'])
        print(table)

    def type(self, col):
        return self.col(col).type

    def types(self, cols = None):
        return [self.col(col).type for col in self.get_cols_indexes(cols)]


    def transpose(self):
        return transpose(self.get_section())

    def get(self, col, rows = None, nan = True, string = False):
        return self.col(col).get(rows, nan = nan, string = string)

    def section(self, rows = None, cols = None, index = False, string = False):
        rows = self.correct_rows(rows)
        cols = self.correct_cols(cols)
        data = transpose([self.get(col, rows, string = string if self.is_datetime(col) else 0) for col in cols])
        data = [[rows[i]] + data[i] for i in range(len(rows))] if index else data
        return data


    def is_categorical(self, col):
        return self.col(col).is_categorical()
    
    def is_numerical(self, col):
        return self.col(col).is_numerical()
    
    def is_datetime(self, col):
        return self.col(col).is_datetime()
    

    def numerical_cols(self):
        return [self.name(col) for col in self.Cols if self.is_numerical(col)]
    
    def datetime_cols(self):
        return [self.name(col) for col in self.Cols if self.is_datetime(col)]
    
    def categorical_cols(self):
        return [self.name(col) for col in self.Cols if self.is_categorical(col)]


    def strip_col(self, col):
        self.col(col).strip()

    def replace_col(self, col, old, new):
        self.col(col).replace(old, new)

    def strip(self):
        [self.strip_col(col) for col in self.Cols]

    def replace(self, old, new):
        [self.replace_col(col, old, new) for col in self.Cols]
    

    def to_float(self, col):
        self.set_col(col, self.col(col).to_float())
        
    def round(self, col, decimals = 1):
        self.col(col).round(decimals)
        
    def to_integer(self, col):
        self.set_col(col, self.col(col).to_integer())
        
    def to_datetime(self, col, form, delta_form):
        self.set_col(col, self.col(col).to_datetime(form, delta_form))


        
    def copy(self):
        return copy(self)

    def subset(self, rows = None):
        rows = self.correct_rows(rows)
        rows = correct_range(rows, self.rows) if is_list(rows) else index_to_range(rows, self.rows)
        new = matrix_class()
        data = [data.subset(rows) for data in self.data]
        new.set_data(data)
        #new.set_names(self.names())
        return new
        
    def part(self, start = None, end = None):
        start = correct_left_index(start, self.rows)
        end = correct_right_index(end, self.rows)
        return self.subset(range(start, end))

    def where(self, value, col):
        return self.subset(self.col(col).where(value))

    

    def unique(self, col, nan = True):
        return self.col(col).unique(nan = nan)
    
    def counts(self, col, normalize = True):
        return self.col(col).counts(normalize)

    def numerical_info(self):
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density']
        info = [[self.get_col_index(col)] + self.col(col).info() for col in self.numerical_cols()]
        table = tabulate(info, headers = cols, decimals = 1)
        print(table)

    def datetime_info(self, form = 'days'):
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density']
        info = [[self.get_col_index(col)] + self.col(col).info(string = True) for col in self.datetime_cols()]
        table = tabulate(info, headers = cols, decimals = 1)
        print(table)

    def categorical_info(self, norm = 0, cols = None, length = 10):
        cols = self.correct_cols(cols)
        [self.col(col).print_counts(norm = norm, length = length) for col in cols if self.is_categorical(col)]



        
    def tabulate(self, col1, col2, length = 5, norm = True, log = False):
        c1, c2 = self.col(col1), self.col(col2)
        unique1 = c1.cross_unique(c2)
        unique2 = c2.cross_unique(c1)
        counts = [[self.where(u1, col1).where(u2, col2).rows for u2 in unique2] for u1 in unique1]
        counts = [normalize(el) for el in counts] if norm else counts
        corr = [correlate(el) for el in counts]
        table = [[unique1[i]] + counts[i][:length] + [corr[i]] for i in range(len(counts))]
        header = [c1.name + ' / ' + c2.name] + unique2[:length] + ['corr']
        table = tabulate(table, headers = header, decimals = 1)
        print(table) if log else None
        corr = mean(corr)
        print('mean correlation', round(corr, 1), '%')  if log else None
        return counts
        
    def categorical_correlation(self, col1, col2):
        return cramers(self.categorical_tab(col1, col2))
    
    def numerical_correlation_(self, col1, col2):
        c1, c2 = self.col(col1), self.col(col2)
        return correlate_numerical(c1.rows(), c2.rows())

    def mix_correlation(self, col1, col2):
        c1, c2 = self.col(col1), self.col(col2)
        i1 = c1.get_index(c2)
        i2 = c2.get_index()
        return correlate_numerical(i1, i2)

    def correlation(self, col1, col2):
        c1, c2 = self.col(col1), self.col(col2)
        cc = c1.is_categorical() and c2.is_categorical()
        cn = c1.is_categorical() and not c2.is_categorical()
        nc = not c1.is_categorical() and c2.is_categorical()
        nn = not c1.is_categorical() and not c2.is_categorical()
        return self.correlate_categorical(col1, col2) if cc else self.correlate_numerical(col1, col2) if nn else self.correlate_categorical_to_numerical(col1, col2) if cn else self.correlate_cat_to_num(col2, col1)








    def plot(self, col1, col2):
        c1, c2 = self.col(col1), self.col(col2)
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
        return correlate_numerical(c1.get_index(), c2.get_index())

    def correlate_categorical_to_numerical(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        i1 = c1.get_index(c2)
        i2 = c2.get_index()
        return correlate_numerical(i1, i2)

    def correlate(self, col1, col2):
        print(col1, col2)
        c1, c2 = self.get_col(col1), self.get_col(col2)
        cc = c1.is_categorical() and c2.is_categorical()
        cn = c1.is_categorical() and not c2.is_categorical()
        nc = not c1.is_categorical() and c2.is_categorical()
        nn = not c1.is_categorical() and not c2.is_categorical()
        return self.correlate_categorical(col1, col2) if cc else self.correlate_numerical(col1, col2) if nn else self.correlate_categorical_to_numerical(col1, col2) if cn else self.correlate_categorical_to_numerical(col2, col1)


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
        return self.subset(self.col(col).where(value))


    def tabulate_data(self, header = True, index = False, rows = None, cols = None, decimals = 1):
        headers = self.names(cols, index) if header else None
        footers = self.types(cols) if header else None
        return tabulate(self.section(rows, cols, index, 1), headers = headers, footers = footers, decimals = decimals)

    def tabulate_types(self, cols = None):
        return tabulate(transpose([self.names(cols), self.get_cols_indexes(cols), self.types(cols)]), headers = ['name', 'id', 'type'])

    def tabulate_dimensions(self):
        return tabulate([[self.rows, self.cols]], headers = ['rows', 'cols'])

    def tabulate_info(self, cols = None):
        return self.tabulate_types(cols) + 2 * nl + self.tabulate_dimensions()

    def print(self, header = True, index = False, info = True, rows = None, cols = None, decimals = 1):
        print(self.tabulate_data(header, index, rows, cols, decimals))
        print(nl + self.tabulate_dimensions()) if info else None

    def __repr__(self):
        rows, cols = 10, 3
        rows = list(range(rows)) + list(range(-rows, 0))
        #cols = list(range(cols)) + list(range(-cols, 0))
        #return self.tabulate_i(1, 1, 0, rows)
        return self.tabulate_info()
