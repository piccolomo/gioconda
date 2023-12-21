from gioconda._data import *
from gioconda._methods import *


class matrix_class():
    def __init__(self):
        self._create_data()

    def _create_data(self):
        self._data = []
        self._update_size()

    def add_data(self, data, name = None):
        index = self._columns
        name = name if name is not None else str(index)
        data = data_class(data, name, index)
        self._data.append(data)
        self._update_size()
        return data

    def add_index(self):
        self.add_data(self._Rows, 'index')
        self.to_numerical('index')
        self._update_attributes()
        return self
    
    def add(self, data_obj):
        self._data = np.append(self._data, data_obj)
        self._update_size()
        self._update_attributes()

    def _update_size(self):
        self._set_columns(len(self._data))
        self.set_rows(self.column(0)._rows if self._columns > 0 else 0)

    def set_rows(self, rows):
        self._rows = rows
        self._Rows = np.arange(self._rows)

    def _set_columns(self, cols):
        self._columns = cols
        self._Columns = np.arange(self._columns)

    def shape(self):
        return self._rows, self._columns

    def _update_attributes(self):
        [setattr(self, name, self.column(name)) for name in self.columns()]

    def _update_indexes(self):
        [self._data[col]._set_index(col) for col in self._Columns]

    def _add_matrix(self, matrix, header = False):
        cols = len(matrix[0])
        columns = matrix[0] if header else [None] * cols
        matrix = matrix[1:] if header else matrix
        data = np.transpose(matrix)
        [self.add_data(data[i], columns[i]) for i in range(cols)]
        self._update_attributes()

    def set_columns(self, columns):
        [self.column(col)._set_name(columns[col]) for col in self._Columns]
        

    def column(self, col):
        col = self.column_index(col)
        return self._data[col]

    def column_index(self, col):
        return self.columns().index(col) if isinstance(col, str) else col

    def columns_indexes(self, cols):
        return vectorize(self.column_index, cols)

    def column_name(self, col):
        return self.column(col)._name

    def rename_column(self, col, name):
        self.column(col)._set_name(name)

    def columns(self, cols = None, index = False):
        cols = self._correct_columns(cols)
        index_col = ['index'] if index else []
        return index_col + [self.column_name(col) for col in cols]

    def _correct_columns(self, cols):
        return [col for col in cols if col in self._Columns or col in self.columns()] if isinstance(cols, list) or isinstance(cols, np.ndarray) else self._Columns if cols is None else [cols]

    def _type(self, col):
        return self.column(col)._type

    def _types(self, cols = None):
        cols = self._correct_columns(cols)
        return [self.column(col)._type for col in cols]

    
    def get_section(self, rows = None, cols = None, nan = True, string = False, index = False):
        rows = self._correct_rows(rows)
        cols = self._correct_columns(cols)
        data = np.transpose([self.column(col).get_section(rows, nan = nan, string = string) for col in cols])
        data = np.concatenate([np.transpose([rows]), data], axis = 1) if index else data
        return data

    def _correct_rows(self, rows):
        return self.column(0)._correct_rows(rows)

    def count(self, col, el, norm = False):
        return self.column(col).count(el, norm = norm)
    
    def counts(self, col, norm = False, nan = True):
        return self.column(col).counts(norm, nan)

    def unique(self, col, nan = True):
        return self.column(col).unique(nan)

    def distinct(self, col, nan = True):
        return self.column(col).distinct(nan)

    def _cross_count(self, col1, col2, val1, val2, norm = False):
        # return self.equal(col1, val1).count(col2, val2, norm = norm)
        rows1 = self.column(col1).equal(val1)
        count = np.count_nonzero(rows1 & self.column(col2).equal(val2))
        return 100 * count / np.count_nonzero(rows1) if norm else count


    def to_numerical(self, col, dictionary = None):
        return self.column(col).to_numerical(dictionary)
    
    def to_categorical(self, col):
        return self.column(col).to_categorical()
    
    def to_datetime(self, col, form = '%d/%m/%Y', delta_form = 'years'):
        return self.column(col).to_datetime(form, delta_form)
    

    def is_categorical(self, col):
        return self.column(col).is_categorical()

    def is_non_categorical(self, col):
        return self.column(col).is_non_categorical()

    def is_numerical(self, col):
        return self.column(col).is_numerical()
    
    def is_datetime(self, col):
        return self.column(col).is_datetime()

    def is_countable(self, col):
        return self.column(col).is_countable()

    def is_uncountable(self, col):
        return self.column(col).is_uncountable()
    
    def categorical_columns(self):
        return [self.column_name(col) for col in self._Columns if self.is_categorical(col)]
    
    def numerical_columns(self):
        return [self.column_name(col) for col in self._Columns if self.is_numerical(col)]

    def datetime_columns(self):
        return [self.column_name(col) for col in self._Columns if self.is_datetime(col)]
    
    def countable_columns(self):
        return [self.column_name(col) for col in self._Columns if self.is_countable(col)]


    def strip(self, cols = None):
        cols = self._correct_columns(cols)
        [self.column(col).strip() for col in cols]

    def replace(self, old, new, cols = None):
        cols = self._correct_columns(cols)
        [self.column(col).replace(old, new) for col in cols]

    def replace_nan(self, cols = None, metric = 'mean'):
        cols = self._correct_columns(cols)
        [self.column(col).replace_nan(metric) for col in cols]


    def numerical_info(self):
        cols = self.countable_columns()
        infos = [self.column(col).numerical_info() for col in cols]
        header = list(infos[0].keys())
        table = [list(el.values()) for el in infos]
        table = [header] + table
        header = [''] + cols 
        table = tabulate(np.transpose(table), header = header)
        print(table + nl)

    def categorical_info(self, norm = False, cols = None, length = 10):
        cols = self._correct_columns(cols)
        cols = [col for col in cols if self.is_categorical(col)]
        [self.column(col)._print_counts(norm = norm, length = length) for col in cols]

    def get_metric(self, col, metric = 'mean'):
        return self.column(col).get_metric(metric)

    def _numerical_cross_counts(self, col1, col2, length = None):
        bins = [self.select(b) for b in self.column(col1).bins(length)]
        metrics = ['mean', 'std', 'length']
        counts = [[b.get_metric(col1, 'mean')] + [b.get_metric(col2, metric) for metric in metrics] for b in bins]
        table = tabulate(counts, [''] + metrics, decimals = 1)
        print(self.column_name(col1), 'vs', self.column_name(col2))
        print(table)
        

    def _categorical_cross_counts(self, col1, col2, norm = False, length = None):
        length = 10 if length is None else None
        unique1 = list(self.unique(col1))[:length]; unique2 = list(self.unique(col2))[:length]
        counts = [[self._cross_count(col1, col2, u1, u2, norm = norm) for u2 in unique2] for u1 in unique1]
        table = [[unique1[i]] + counts[i] for i in range(len(counts))]
        header = [self.column_name(col1) + ' / ' + self.column_name(col2)] + unique2
        table = tabulate(table, header = header, decimals = 1)
        print(table)

    def _mixed_cross_counts(self, col1, col2, length = None):
        unique = list(self.unique(col1))[ : length];
        metrics = ['mean', 'std', 'length']
        dics = [self.get_encoding(col1, col2, metric, 1) for metric in metrics]
        values = [[u] + [dic[u] for dic in dics] for u in unique]
        table = tabulate(values, [''] + metrics, decimals = 1)
        print(self.column_name(col1), 'vs', self.column_name(col2))
        print(table)

    def tab(self, col1, col2, norm = False, length = None):
        c1 = self.column(col1)
        length = c1._get_max_bins() if length is None and c1.is_countable() else length
        if self.is_non_categorical(col1) and self.is_non_categorical(col2):
            return self._numerical_cross_counts(col1, col2, length)
        elif self.is_categorical(col1) and self.is_categorical(col2):
            self._categorical_cross_counts(col1, col2, norm = norm, length = length)
        elif self.is_categorical(col1) and self.is_non_categorical(col2):
            self._mixed_cross_counts(col1, col2, length)
        else:
            print('Warning: Columns switched')
            self._mixed_cross_counts(col2, col1, metric, length)
        
    def plot(self, col, bins = 100):
        self.column(col).plot(bins)

    def cross_plot(self, col1, col2, length = 100):
        x, y = self.column(col1).get_section(nan = True), self.column(col2).get_section(nan = True)
        plt.figure(0, figsize = (15, 8)); plt.clf()
        plt.scatter(x, y, alpha=0.5)
        #x, y = remove_nan(x, y)
        if self.is_countable(col1):
            d = self.distinct(col1, 0)
            if d < 100:
                bins = self.column(col1).bins(d)
                xf = [self.column(col1).select(b).mean() for b in bins]
                yf = [self.column(col2).select(b).mean() for b in bins]
            else:
                xf, yf = smooth(x, y, length, 4)
            plt.plot(xf, yf, color = 'red')

        plt.xlabel(self.column_name(col1)); plt.ylabel(self.column_name(col2))
        plt.xticks(rotation = 90) if self.is_categorical(col1) else None
        plt.tight_layout(); plt.pause(0.1); plt.show(block = 1); plt.clf(); plt.close()


    def _tabulate_data(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        header = self.columns(cols, index) if header else None
        return tabulate(self.get_section(rows, cols, index = index, string = 1), header = header, decimals = decimals)

    def _tabulate_dimensions(self):
        return tabulate([[self._rows, self._columns]], header = ['rows', 'cols'])

    def _tabulate_types(self, cols = None):
        cols = self._Columns if cols is None else cols
        table = np.transpose([self.columns_indexes(cols), self.columns(cols), self._types(cols)])
        return tabulate(table, header = ['i', 'column', 'type'])

    def _tabulate_info(self, cols = None):
        return self._tabulate_dimensions() + 2 * nl + self._tabulate_types(cols)

    def print(self, rows = None, cols = None, header = True, index = False, decimals = 1):
        rows = min(40, th()) if rows is None else rows
        rows = (np.arange(rows) if rows >= 0 else np.arange(self._rows + rows, self._rows)) if isinstance(rows, int) else rows 
        rows = np.arange(rows) if isinstance(rows, int) else self._correct_rows(rows)
        print(self._tabulate_data(rows, cols, header, index, decimals))
        print(nl + self._tabulate_dimensions())

    def __repr__(self):
        return self._tabulate_info()

    
    def __getitem__(self, cols):
        cols = [cols] if isinstance(cols, str) or isinstance(cols, int) else cols 
        new = matrix_class()
        for col in cols:
            new._data.append(self.column(col))
        new._update_size()
        new._update_attributes()
        return new

    def equal(self, col, value):
        return self.column(col).equal(value)

    def not_equal(self, col, value):
        return self.column(col).not_equal(value)
    
    def higher(self, col, value, equal = True):
        return self.column(col).higher(value, equal)

    def lower(self, col, value, equal = True):
        return self.column(col).lower(value, equal)

    def between(self, col, start, end, equal1 = True, equal2 = False):
        return self.column(col).between(start, end, equal1, equal2)

    def select(self, rows = None, cols = None):
        cols = self._correct_columns(cols)
        new = self[cols]
        new._data = [data.select(rows) for data in new._data]
        new._update_size()
        new._update_attributes()
        return new

    def select_numerical_columns(self):
        return self[self.numerical_columns()]

    def select_countable_columns(self):
        return self[self.countable_columns()]
    
    def part(self, start = None, end = None):
        start = 0 if start is None else max(0, start)
        end = self._rows if end is None else min(end, self._rows)
        return self.select(np.arange(start, end))

    def copy(self):
        return copy(self)

    
    def simulate_categorical(self, name = None, categories = 5, nan_ratio = 0.1):
        categories = [random_word(5) for i in range(categories)] if isinstance(categories, int) else categories
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random.choice(categories) for el in self._Rows]
        self.add_data(data, name)

    def simulate_numerical(self, name = None, mean = 0, std = 1, nan_ratio = 0.1):
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random.normalvariate(mean, std) for el in self._Rows]
        self.add_data(data, name)
        self.to_numerical(name)

    def simulate_datetime(self, name = None, mean = "12/10/2000", std = 1, form = '%d/%m/%Y', delta_form = 'years', nan_ratio = 0.1):
        data = ['nan' if random.uniform(0, 1) < nan_ratio else random_datetime64(mean, std, form, delta_form) for el in self._Rows]
        self.add_data(data, name)
        self.to_datetime(name, form)

    def duplicate(self, col, name):
        data = self.column(col)._data
        self.add_data(data, name)
        self.column(name)._set_type(self.column(col)._type)
        return self

    def delete(self, col):
        index = self.column_index(col)
        self._data.pop(index)
        self._update_size()
        self._update_indexes()
        return self

    def sort_values(self, by):
        cols = [by] if isinstance(by, str) or isinstance(by, int) else by
        for col in cols:
            rows = self.column(col).argsort()
            self.column(col).sort(rows)
        return self

    def to_pandas(self):
        import pandas as pd
        df = pd.DataFrame(self.get_section(), columns = self.columns())
        return df

    def get_encoding(self, col1, col2, metric = 'mean', string = False):
        unique = list(self.unique(col1)); l = len(unique)
        group = [self.select(self.equal(col1, u)).column(col2) for u in unique]
        get_metric = lambda el: el.len() if metric == 'length' else el.sum() if metric == 'sum' else el.std() if metric == 'std' else el.mean() if metric in ['mean', 'order'] else el
        values = [get_metric(el) for el in group]
        values = ordering(values) if metric == 'order' else values
        to_string = self.column(col1)._to_string
        values = list(map(to_string, values)) if string else values
        dictionary =  {unique[i]: values[i] for i in range(l)}
        return dictionary

    def encode(self, col1, col2, metric = 'mean', string = False):
        d = self.get_encoding(col1, col2, metric, string)
        self.to_numerical(col1, d)
        return self

    def to_pickle(self, path, log = True):
        import pickle
        print("writing pickle") if log else None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print("pickle written!\n") if log else None

    def set(self, col, new):
        self.column(col).set(new)

    def split(self, ratio = 0.2):
        test = round(ratio * self._rows)
        test = np.array(sample(list(self._Rows), k = test))
        train = np.array([el for el in self._Rows if el not in test])
        return self.select(train), self.select(test)

    def order_by(self, col):
        rows = self.column(col).order()
        return self.select(rows = rows)

    def __len__(self):
        return self._rows
