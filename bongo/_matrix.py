from bongo._data import *
from bongo._methods import *


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
        self.set_cols(len(self.data))
        self.set_rows(self.col(0).rows if self.cols > 0 else 0)

    def set_rows(self, rows):
        self.rows = rows
        self.Rows = np.arange(self.rows)

    def set_cols(self, cols):
        self.cols = cols
        self.Cols = np.arange(self.cols)

    def update_indexes(self):
        [self.data[col].set_index(col) for col in self.Cols]

    def add_matrix(self, matrix, header = False):
        cols = len(matrix[0])
        names = matrix[0] if header else [None] * cols
        matrix = matrix[1:] if header else matrix
        data = np.transpose(matrix)
        [self.add_data(data[i], names[i]) for i in range(cols)]

    def set_names(self, names):
        [self.col(col).set_name(names[col]) for col in self.Cols]
        

    def col(self, col):
        col = self.index(col)
        return self.data[col]

    def index(self, col):
        return self.names().index(col) if isinstance(col, str) else col

    def indexes(self, cols):
        return vectorize(self.index, cols)

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
        return self.col(col).count(el, norm = norm)
    
    def counts(self, col, norm = False, nan = True):
        return self.col(col).counts(norm, nan)

    def unique(self, col, nan = True):
        return self.col(col).unique(nan)

    def distinct(self, col, nan = True):
        return self.col(col).distinct(nan)

    def cross_count(self, col1, col2, val1, val2, norm = False):
        # return self.equal(col1, val1).count(col2, val2, norm = norm)
        rows1 = self.col(col1).equal(val1)
        count = np.count_nonzero(rows1 & self.col(col2).equal(val2))
        return 100 * count / np.count_nonzero(rows1) if norm else count


    def to_numerical(self, col, dictionary = None):
        return self.col(col).to_numerical(dictionary)
    
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


    def strip(self, cols = None):
        cols = self.correct_cols(cols)
        [self.col(col).strip() for col in cols]

    def replace(self, old, new, cols = None):
        cols = self.correct_cols(cols)
        [self.col(col).replace(old, new) for col in cols]


    def numerical_info(self):
        cols = self.countable_cols()
        infos = [self.col(col).numerical_info() for col in cols]
        header = list(infos[0].keys())
        table = [list(el.values()) for el in infos]
        table = [header] + table
        header = [''] + cols 
        table = tabulate(np.transpose(table), header = header)
        print(table + nl)

    def categorical_info(self, norm = False, cols = None, length = 10):
        cols = self.correct_cols(cols)
        cols = [col for col in cols if self.is_categorical(col)]
        [self.col(col).print_counts(norm = norm, length = length) for col in cols]

    def categorical_cross_counts(self, col1, col2, norm = False, length = 10):
        unique1 = list(self.unique(col1))[:length]; unique2 = list(self.unique(col2))[:length]
        counts = [[self.cross_count(col1, col2, u1, u2, norm = norm) for u2 in unique2] for u1 in unique1]
        table = [[unique1[i]] + counts[i] for i in range(len(counts))]
        header = [self.name(col1) + ' / ' + self.name(col2)] + unique2
        table = tabulate(table, header = header, decimals = 1)
        print(table)

    def mixed_cross_counts(self, col1, col2, length = 10):
        unique1 = list(self.unique(col1))[:length]; unique2 = list(self.unique(col2))[:length]
        data1 = [self.equal(col1, u1).col(col2) for u1 in unique1]
        table = [[data.to_string(data.mean()), data.to_string(data.std()), data.rows] for data in data1]
        table = [[unique1[i]] + table[i] for i in range(len(table))]
        header = [self.name(col1) + ' / ' + self.name(col2), 'mean', 'std', 'len']
        table = tabulate(table, header = header, decimals = 1)
        print(table)

    def tab(self, col1, col2, norm = False, length = 10):
        if self.is_non_categorical(col1) and self.is_non_categorical(col2):
            print('Warning: At least one column should be categorical')
        elif self.is_categorical(col1) and self.is_categorical(col2):
            return self.categorical_cross_counts(col1, col2, norm = norm, length = length)
        elif self.is_categorical(col1) and self.is_non_categorical(col2):
            return self.mixed_cross_counts(col1, col2, length = length)
        else:
            return self.mixed_cross_counts(col2, col1, length = length)
        
    def plot(self, col, bins = 100):
        self.col(col).plot(bins)

    def cross_plot(self, col1, col2):
        plt.figure(0, figsize = (15, 8)); plt.clf()
        plt.scatter(self.col(col1).get_section(nan = True), self.col(col2).get_section(nan = True))
        plt.xlabel(self.name(col1)); plt.ylabel(self.name(col2))
        plt.xticks(rotation = 90) if self.is_categorical(col1) else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()


    def tabulate_data(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        header = self.names(cols, index) if header else None
        return tabulate(self.get_section(rows, cols, index = index, string = 1), header = header, decimals = decimals)

    def tabulate_dimensions(self):
        return tabulate([[self.rows, self.cols]], header = ['rows', 'cols'])

    def tabulate_types(self, cols = None):
        cols = self.Cols if cols is None else cols
        table = np.transpose([self.indexes(cols), self.names(cols), self.types(cols)])
        return tabulate(table, header = ['i', 'column', 'type'])

    def tabulate_info(self, cols = None):
        return self.tabulate_dimensions() + 2 * nl + self.tabulate_types(cols)

    def print(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        rows = min(self.rows, plx.th() - 8) if rows is None else rows
        print(self.tabulate_data(np.arange(rows), cols, header, index, decimals))
        print(nl + self.tabulate_dimensions())

    def __repr__(self):
        return self.tabulate_info()

    def __getitem__(self, col):
        return self.col(col)



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

    def copy(self):
        return copy(self)

    
    def simulate_categorical(self, name = None, length = 5, nan_ratio = 0.1):
        categories = [random_word(5) for i in range(length)]
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random.choice(categories) for el in self.Rows]
        self.add_data(data, name)

    def simulate_numerical(self, name = None, mean = 0, std = 1, nan_ratio = 0.1):
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random.normalvariate(mean, std) for el in self.Rows]
        self.add_data(data, name)
        self.to_numerical(name)

    def simulate_datetime(self, name = None, mean = "12/10/2000", std = 1, form = '%d/%m/%Y', delta_form = 'years', nan_ratio = 0.1):
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random_datetime64(mean, std, form, delta_form) for el in self.Rows]
        self.add_data(data, name)
        self.to_datetime(name, form)

    def duplicate(self, col, name):
        data = self.col(col).data
        self.add_data(data, name)
        self.col(name).set_type(self.col(col).type)

    def delete(self, col):
        index = self.index(col)
        self.data = np.delete(self.data, index)
        self.update_size()
        self.update_indexes()
        
