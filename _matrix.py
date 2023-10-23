from bongo._data import *

class matrix_class():
    def __init__(self):
        self.create_data()

    def create_data(self):
        self.data = []
        self.update_size()

    def add_data(self, data, name = None):
        index = self.cols
        name = name if name is not None else index
        data = data_class(data, name, index)
        self.data = np.append(self.data, data)
        self.update_size()
        return data

    def update_size(self):
        self.cols = len(self.data)
        self.Cols = np.arange(self.cols)
        self.rows = self.col(0).rows if self.cols > 0 else 0
        self.Rows = np.arange(self.rows)

    def add_matrix(self, matrix, header = False):
        cols = len(matrix[0])
        names = matrix[0] if header else [None] * cols
        matrix = matrix[1:] if header else matrix
        data = transpose(matrix)
        [self.add_data(data[i], names[i]) for i in range(cols)]
        

    def col(self, col):
        col = self.index(col)
        return self.data[col]

    def index(self, col):
        return self.names().index(col) if isinstance(col, str) else col

    def indexes(self, cols):
        return np.vectorize(self.index)(cols)

    def name(self, col):
        return self.col(col).name

    def names(self, cols = None, index = False):
        cols = self.correct_cols(cols)
        return [self.name(col) for col in cols]

    def correct_cols(self, cols):
        return np.array([col for col in cols if col in self.Cols]) if isinstance(cols, list) else self.Cols if cols is None else cols

    def type(self, col):
        return self.col(col).type

    def types(self, cols = None):
        cols = self.correct_cols(cols)
        return [self.col(col).type for col in cols]

    
    def get_section(self, rows = None, cols = None, nan = True, string = False, index = False):
        rows = self.correct_rows(rows)
        cols = self.correct_cols(cols)
        data = np.transpose([self.col(col).get_section(rows, nan = nan, string = string) for col in cols])
        data = np.concatenate([np.transpose([self.Rows]), data], axis = 1) if index else data
        return data

    def correct_rows(self, rows):
        return self.col(0).correct_rows(rows)


    def count(self, col, el, norm = False):
        return self.col(col).count(el, norm)
    
    def counts(self, col, norm = False):
        return self.col(col).counts(norm)

    def unique(self, col, nan = True):
        return self.col(col).unique(nan)


    def to_numerical(self, col):
        return self.col(col).to_numerical()
    
    def to_categorical(self, col):
        return self.col(col).to_categorical()
    
    def to_datetime(self, col, form = '%d/%m/%Y', delta_form = 'years'):
        return self.col(col).to_datetime(form, delta_form)

    def is_categorical(self, col):
        return self.col(col).is_categorical()

    def is_non_categorical(self, col):
        return self.col(col).is_non_categorical()

    def is_numerical(self, col):
        return self.col(col).is_numerical()
    
    def is_datetime(self, col):
        return self.col(col).is_datetime()

    def is_countable(self, col):
        return self.col(col).is_countable()

    def is_uncountable(self, col):
        return self.col(col).is_uncountable()
    
    def categorical_cols(self):
        return [self.name(col) for col in self.Cols if self.is_categorical(col)]
    
    def countable_cols(self):
        return [self.name(col) for col in self.Cols if self.is_countable(col)]


    def strip(self, col):
        self.col(col).strip()

    def replace(self, col, old, new):
        self.col(col).replace(old, new)


    def numerical_info(self):
        cols = self.countable_cols()
        infos = [self.col(col).numerical_info() for col in cols]
        header = list(infos[0].keys())
        table = [list(el.values()) for el in infos]
        table = [header] + table
        header = [''] + cols 
        table = tabulate(transpose(table), header = header)
        print(table + nl)

    def categorical_info(self, norm = 0, cols = None, length = 10):
        cols = self.categorical_cols()
        [self.col(col).print_counts(norm = norm, length = length) for col in cols]

    def categorical_cross_counts(self, col1, col2, norm = False):
        unique1 = list(self.unique(col1)); unique2 = list(self.unique(col2))
        counts = [[self.equal(col1, u1).count(col2, u2, norm = norm) for u2 in unique2] for u1 in unique1]
        table = [[unique1[i]] + counts[i] for i in range(len(counts))]
        header = [self.name(col1) + ' / ' + self.name(col2)] + unique2
        table = tabulate(table, header = header, decimals = 1)
        print(table)

    def mixed_cross_counts(self, col1, col2):
        unique1 = list(self.unique(col1)); unique2 = list(self.unique(col2))
        data1 = [self.equal(col1, u1).col(col2) for u1 in unique1]
        table = [[data.mean(), data.std(), data.rows] for data in data1]
        table = [[unique1[i]] + table[i] for i in range(len(table))]
        header = [self.name(col1) + ' / ' + self.name(col2), 'mean', 'std', 'len']
        table = tabulate(table, header = header, decimals = 1)
        print(table)

    def tab(self, col1, col2, norm = False):
        if self.is_non_categorical(col1) and self.is_non_categorical(col2):
            print('Warning: At least one column should be categorical')
        elif self.is_categorical(col1) and self.is_categorical(col2):
            return self.categorical_cross_counts(col1, col2, norm = False)
        elif self.is_categorical(col1) and self.is_non_categorical(col2):
            return self.mixed_cross_counts(col1, col2)
        else:
            return self.mixed_cross_counts(col2, col1)
        
    def plot(self, col, bins = 100):
        self.col(col).plot(bins)

    def cross_plot(self, col1, col2):
        plt.figure(0, figsize = (15, 8)); plt.clf()
        plt.scatter(self.col(col1).get_section(nan = True), self.get(col2).get_section(nan = True))
        plt.xlabel(self.name(col1)); plt.ylabel(self.name(col2))
        plt.xticks(rotation = 90) if self.is_numerical(col1) else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()


    def tabulate_data(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        header = self.names(cols, index) if header else None
        return tabulate(self.get_section(rows, cols, index = index, string = 1), header = header, decimals = decimals)

    def tabulate_dimensions(self):
        return tabulate([[self.rows, self.cols]], header = ['rows', 'cols'])

    def tabulate_types(self, cols = None):
        cols = self.Cols if cols is None else cols
        table = transpose([self.indexes(cols), self.names(cols), self.types(cols)])
        return tabulate(table, header = ['i', 'column', 'type'])

    def tabulate_info(self, cols = None):
        return self.tabulate_dimensions() + 2 * nl + self.tabulate_types(cols)

    def print(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        rows = min(self.rows, plx.th() - 8) if rows is None else rows
        print(self.tabulate_data(np.arange(rows), cols, header, index, decimals))
        print(nl + self.tabulate_dimensions())

    def __repr__(self):
        return self.tabulate_info()


    def equal(self, col, value):
        return self.subset(self.col(col).equal(value))

    def not_equal(self, col, value):
        return self.subset(self.col(col).not_equal(value))
    
    def greater(self, col, value, equal = True):
        return self.subset(self.col(col).greater(value, equal))

    def lower(self, col, value, equal = True):
        return self.subset(self.col(col).lower(value, equal))

    def subset(self, rows = None):
        rows = self.correct_rows(rows)
        new = matrix_class()
        new.data = [data.subset(rows) for data in self.data]
        new.update_size()
        return new
    
    def part(self, start = None, end = None):
        start = 0 if start is None else max(0, start)
        end = self.rows if end is None else min(end, self.rows)
        return self.subset(np.arange(start, end))



    
    # def numerical_cols(self):
    #     return [self.name(col) for col in self.Cols if self.is_numerical(col)]
    
    # def datetime_cols(self):
    #     return [self.name(col) for col in self.Cols if self.is_datetime(col)]

    # def non_categorical_cols(self):
    #     return [self.name(col) for col in self.Cols if not self.is_categorical(col)]
    








    


        


    
    def copy(self):
        return copy(self)
    


        
        
    # def set_col_names(self, names):
    #     [self.data[col].set_name(names[col]) for col in self.Cols]

#     def set(self, row, col, el):
#         self.col(col).set(row, el)
        
#     def set_col(self, col, data_obj):
#         col = self.get_col_index(col)
#         self.data[col] = data_obj
        
#     def set_col_data(self, col, data):
#         self.col(col).set_data(data)

#     def set_type(self, col, type):
#         self.col(col).set_type(type)

#     def multiply(self, col, k):
#         c = self.col(col)
#         c.multiply(k) if c.is_numerical() else None

#     def get_cols_indexes(self, cols = None):
#         cols = self.Cols if cols is None else np.vectorize(self.get_col_index)(cols)
#         return correct_range(cols, self.cols)

#     def transpose(self):
#         return np.transpose(self.get_section())

#     def to_float(self, col):
#         self.set_col(col, self.col(col).to_float())
        
#     def round(self, col, decimals = 1):
#         self.col(col).round(decimals)
        
#     def to_integer(self, col):
#         self.set_col(col, self.col(col).to_integer())
        
#     def to_datetime(self, col, form, delta_form):
#         self.set_col(col, self.col(col).to_datetime(form, delta_form))
    

#     def unique(self, col, nan = True):
#         return self.col(col).unique(nan = nan)

#     def cross_unique(self, col1, col2, nan = True):
#         return self.col(col1).cross_unique(self.col(col2), nan = nan)



#     def print_types(self, cols = None):
#         print(self.tabulate_types(cols))

