from bongo._data import *
from bongo._methods import *


class matrix_class():
    
    def __init__(self, matrix = None):
        self.create_data()
        self.set_matrix(matrix)

    def create_data(self):
        self.data = []
        self.update_size()

    def set_matrix(self, matrix = None):
        self.data = []
        matrix = [[]] if matrix is None else matrix
        [self.add_data(data) for data in transpose(matrix)]
        
    def add_data(self, data):
        data = categorical_data_class(data)
        data.set_index(self.cols); data.set_name(self.cols)
        self.data.append(data)
        self.update_size()

    def update_size(self):
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
        cols = self.correct_cols(cols)
        names = ['i'] if index else []
        return names + [self.name(col) for col in cols]

    def type(self, col):
        return self.col(col).type

    def types(self, cols = None):
        cols = self.correct_cols(cols)
        return [self.col(col).type for col in cols]


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

    def is_not_categorical(self, col):
        return self.col(col).is_not_categorical()

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

    def non_categorical_cols(self):
        return [self.name(col) for col in self.Cols if not self.is_categorical(col)]


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
    

    def unique(self, col, nan = True):
        return self.col(col).unique(nan = nan)

    def cross_unique(self, col1, col2, nan = True):
        return self.col(col1).cross_unique(self.col(col2), nan = nan)

    
    def count(self, el, col, norm = False):
        return self.col(col).count(el, norm)
    
    def counts(self, col, norm = True):
        return self.col(col).counts(norm)

    def numerical_info(self):
        cols = self.non_categorical_cols()
        col0 = cols[0]
        header = list(self.col(col0).numerical_info(string = 1).keys())
        table = [list(self.col(col).numerical_info(string = 1).values()) for col in cols]
        table = [header] + table
        #table = transpose([header]) + transpose(table)
        table = tabulate(transpose(table))#, header = header)
        print(table + nl)

    def categorical_info(self, norm = 0, cols = None, length = 10):
        cols = self.correct_cols(cols)
        [self.col(col).print_counts(norm = norm, length = length) for col in cols if self.is_categorical(col)]

    def tab(self, col1, col2, log = True, length = 5):
        c1, c2 = self.col(col1), self.col(col2)
        if c1.is_not_categorical() and c2.is_not_categorical():
            print('Warning: At least one column should be categorical')
        elif c1.is_categorical() and c2.is_categorical():
            return self.categorical_tab(col1, col2, log = log, length = length)
        elif c1.is_categorical() and c2.is_not_categorical():
            return self.mixed_categorical_tab(col1, col2, log = log)
        else:# c1.is_not_categorical() and c2.is_categorical():
            return self.mixed_categorical_tab(col2, col1, log = log)

        
    def categorical_tab(self, col1, col2, log = True, length = 5):
        c1, c2 = self.col(col1), self.col(col2)
        unique1 = c1.cross_unique(c2)
        unique2 = c2.cross_unique(c1)
        counts = [[self.equal(u1, col1).count(u2, col2, norm = False) for u2 in unique2] for u1 in unique1]
        corr = cramers(counts)
        unique1 = [c1.to_string(el) for el in unique1]
        unique2 = [c2.to_string(el) for el in unique2]
        table = [[unique1[i]] + counts[i][:length] for i in range(len(counts))]
        header = [c1.name + ' / ' + c2.name] + unique2[:length]
        table = tabulate(table, header = header, decimals = 1)
        print(table) if log else None
        print(' Cramers:', round(100 * corr, 1) if corr != n else n, '%' + nl) if log else None
        return corr

    def mixed_categorical_tab(self, col1, col2, log = True):
        c1, c2 = self.col(col1), self.col(col2)
        unique1 = c1.cross_unique(c2)
        unique2 = c2.cross_unique(c1)
        sub_data = [self.equal(u1, col1).col(col2) for u1 in unique1]
        table = [[data.mean(), data.std(), data.rows] for data in sub_data]
        std = mean([data[1] for data in table if data[1] != n])
        std_old = c2.std()
        corr = (std - std_old) / std_old
        table = [[c2.to_string(el) for el in data] for data in table]
        unique1 = [c1.to_string(el) for el in unique1]
        unique2 = [c2.to_string(el) for el in unique2]
        table = [[unique1[i]] + table[i] for i in range(len(table))]
        header = [c1.name + ' / ' + c2.name, 'mean', 'std', 'len']
        table = tabulate(table, header = header, decimals = 1)
        print(table) if log else None
        print(' STD change from', c2.to_string(std), 'is', round(100 * corr, 1), '%' + nl) if log else None
        return corr


    def plot(self, col, bins = 100):
        c = self.col(col)
        is_num = c.is_numerical() or c.is_datetime()
        data = c.get(nan = False) if is_num else c.get(nan = False)
        plt.figure(0, figsize = (15, 8))
        plt.clf()
        plt.hist(data, bins = min(bins, len(c.unique()))) if is_num else None
        plt.bar(*transpose(self.col(col).counts())) if not is_num else None
        #plt.bar(data)
        plt.xlabel(c.name); plt.ylabel('count')
        plt.xticks(rotation = 90) if not is_num else None
        #plt.yticks(rotation = 90, ha = 'right') if not c2.is_numerical() else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()
        
    def crossplot(self, col1, col2):
        c1, c2 = self.col(col1), self.col(col2)
        x, y = c1.get(nan = True), c2.get(nan = True)
        x, y = transpose([el for el in transpose([x, y]) if n not in el]) 
        plt.figure(0, figsize = (15, 8))
        plt.clf()
        plt.scatter(x, y)
        plt.xlabel(c1.name); plt.ylabel(c2.name)
        plt.xticks(rotation = 90) if not c1.is_numerical() else None
        #plt.yticks(rotation = 90, ha = 'right') if not c2.is_numerical() else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()


        
    def tabulate_data(self, header = True, index = False, rows = None, cols = None, decimals = 1):
        header = self.names(cols, index) if header else None
        return tabulate(self.section(rows, cols, index, 1), header = header, decimals = decimals)

    def tabulate_types(self, cols = None):
        cols = self.Cols if cols is None else cols
        table = transpose([self.get_cols_indexes(cols), self.names(cols), self.types(cols)])
        return tabulate(table, header = ['i', 'column', 'type'])
        #return tabulate(transpose([self.names(cols), self.get_cols_indexes(cols), self.types(cols)]), header = ['name', 'id', 'type'])

    def print_types(self, cols = None):
        print(self.tabulate_types(cols))

    def tabulate_dimensions(self):
        return tabulate([[self.rows, self.cols]], header = ['rows', 'cols'])

    def tabulate_info(self, cols = None):
        return self.tabulate_dimensions() + 2 * nl + self.tabulate_types(cols)

    def print(self, header = True, index = False, rows = None, cols = None, decimals = 1):
        rows = plx.th() - 7 if rows is None else rows
        print(self.tabulate_data(header, index, rows, cols, decimals))
        print(nl + self.tabulate_dimensions())

    def __repr__(self):
        rows, cols = 10, 3
        rows = list(range(rows)) + list(range(-rows, 0))
        #cols = list(range(cols)) + list(range(-cols, 0))
        #return self.tabulate_i(1, 1, 0, rows)
        return self.tabulate_info()

    
    def copy(self):
        return copy(self)

    def subset(self, rows = None):
        rows = self.correct_rows(rows)
        rows = correct_range(rows, self.rows) if is_list(rows) else index_to_range(rows, self.rows)
        new = matrix_class()
        new.data = [data.subset(rows) for data in self.data]
        new.update_size()
        return new
        
    def part(self, start = None, end = None):
        start = correct_left_index(start, self.rows)
        end = correct_right_index(end, self.rows)
        return self.subset(range(start, end))

    def equal(self, value, col):
        return self.subset(self.col(col).equal(value))

    def not_equal(self, value, col):
        return self.subset(self.col(col).not_equal(value))
    
    def greater(self, value, col, equal = True):
        return self.subset(self.col(col).greater(value, equal))

    def lower(self, value, col, equal = True):
        return self.subset(self.col(col).lower(value, equal))
