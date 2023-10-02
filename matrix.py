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
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density', n]
        info = [self.get_col(col).get_info() for col in self.get_numerical_cols()]
        table = tabulate_data(info, headers = cols, decimals = 1)
        print(table)

    def describe_datetime(self, form = 'days'):
        cols = ['i', 'col', 'mean', 'median', 'mode', 'std', 'span', 'density', n]
        info = [self.get_col(col).get_info(string = True) for col in self.get_datetime_cols()]
        table = tabulate_data(info, headers = cols, decimals = 1)
        print(table)

    def describe_categorical(self, normalize = 0, cols = None, rows = 10):
        cols = self.correct_cols(cols)
        [self.get_col(col).print_info(normalize, rows = rows) for col in cols if self.is_categorical(col)]


    def plot(self, col1, col2):
        c1, c2 = self.get_col(col1), self.get_col(col2)
        x = c1.get_index()
        data = c1 if c1.is_categorical() else None
        y = c2.get_index(data)
        xy = [el for el in transpose([x, y]) if n not in el]
        x, y = transpose(xy)
        plt.clf()
        plt.scatter(x, y)
        plt.xticks(*c1.get_ticks())
        plt.yticks(*c2.get_ticks())
        plt.xlabel(c1.name)
        plt.ylabel(c2.name)
        plt.show()
        
    def crosstab(self, col1, col2, cols = 10):
        unique1 = self.get_col(col1).unique()
        unique2 = self.get_col(col2).unique()
        counts = [[self.where(col1, u1).where(col2, u2).rows for u2 in unique2] for u1 in unique1]
        corr_rows = [corr(data) for data in counts]
        
        to_string1 = self.get_col(col1).to_string
        to_string2 = self.get_col(col2).to_string
        unique1 = [to_string1(el) for el in unique1]
        unique2 = [to_string2(el) for el in unique2]
        header = [''] + unique2[:cols] + ['']
        table = [[unique1[i]] + counts[i][:cols] + [corr_rows[i]] for i in range(len(counts))]
        # table  = [[''] + unique2]
        # table += [[to_string1(u1)] + 
        table = tabulate_data(table, headers = header, grid = True, decimals = 1)
        print(table)
        print('correlation', col1, 'implies', col2, round(mean(corr_rows), 1), '%')
        return 
        

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

    def where(self, value, cols = None):
        cols = self.get_cols_indexes(cols)
        rows = join([self.get_col(col).where(value) for col in cols])
        return self.subset(rows)


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
    



# to_int = lambda el: np.int64(el) if el != 'nan' else nan
# to_bool = lambda el: bool(el) if el != 'nan' else nan

# def get_numerical_frame(frame):
#     return frame.select_dtypes(include = ['number'])

# def get_datetime_frame(frame):
#     return frame.select_dtypes(include = ['datetime'])

# def get_categorical_frame(frame):
#     frame = frame.select_dtypes(exclude = ['number', 'datetime'])
#     frame.replace(np.nan, 'nan', inplace = True)
#     return frame

# def datetime_to_numerical(frame):
#     frame = frame.copy()
#     frame -= frame.min()
#     for col in frame.cols:
#         frame[col] = frame[col].apply(timedelta_to_years)
#     return frame

# def hstack_frames(frame1, frame2):
#     return pd.concat([frame1, frame2], axis = 1)



# def print_multiindex_frame(frame, decimals = 1):
#     first_index = frame.index.names[0]
#     groups = frame.groupby(first_index)
#     first_index_values = list(set([el[0] for el in frame.index]))
#     for value in first_index_values:
#         print(value)
#         print_frame(frame.xs(value).reset_index(), style = 'simple_outline')

# def describe_categorical_frame(frame, length = 5):
#     cols = list(frame.cols)
#     data = []
#     for col in cols:
#         values = frame[col].value_counts(normalize = True) * 100
#         keys = values.keys().to_list()
#         counts = list(values.values)
        
#         lv = min(length, len(values)); r = range(lv)
#         for i in r:
#             data.append([col, keys[i], counts[i]])

#     data = pd.DataFrame(data, cols =  ['col', 'value', 'count'])
#     data.set_index(['column', 'value'], inplace = True)
#     return data

# def describe_numerical_frame(frame):
#     info = pd.DataFrame()
#     cols = list(frame.columns)
#     info.index = cols
#     mean = [frame[col].mean() for col in cols]
#     mode = [frame[col].mode()[0] for col in cols]
#     std = np.array([frame[col].std() for col in cols])
#     m = np.array( [frame[col].min() for col in cols])
#     M =  np.array([frame[col].max() for col in cols])
#     spread =  M - m
#     info['mean'] = mean
#     info['mode'] = mode
#     info['std'] = std
#     info['spread'] = spread
#     info['density'] = std / spread
#     return info

# def describe_datetime_frame(frame):
#     info = describe_numerical_frame(frame)
#     info['mean'] = info['mean'].dt.strftime('%d/%m/%Y')
#     info['mode'] = info['mode'].dt.strftime('%d/%m/%Y')
#     info['std'] = info['std'].apply(timedelta_to_years)
#     info['spread'] = info['spread'].apply(timedelta_to_years)
#     return info
    

# def get_numerical_correlations(frame):
#     return 100 * frame.corr(method = 'spearman')

# def get_categorical_correlations(frame):
#     cols = frame.columns
#     new = pd.DataFrame(columns = cols, index = cols)
#     l = len(cols); rg = range(l)
#     for row in cols:
#         for col in cols:
#             print(row, col)
#             cross = pd.crosstab(frame[row], frame[col], normalize = 0) * 100
#             new[row][col] = int(k2(cross).pvalue < 0.05)
#             # row, col = cols[r], cols[c]
#             # if col != row:
#             #     print(row, 'vs', col)
#             #     i = cross.idxmax(axis = 1)
#             #     m = cross.max(axis = 1)
#             #     p = m > -50
#             #     m = pd.concat([i[p], m[p]], axis = 1)
#             #     print_frame(m)
                
#             #corr = 100 if row == col else 100 * cramers_v(frame[row], frame[col])
#             #new[row][col] = round(corr, 1)
#     return new

# def get_mix_correlations(categorical, numerical):
#     cols_cat  = categorical.columns
#     cols_num = numerical.columns
#     for row in cols_cat:
#         for col in cols_num:
#             cat = categorical[row]
#             num = numerical[col]
#             print(row, col)
#             plt.clf()
#             sns.violinplot(x = cat, y = num, bw = 0.15)
#             plt.xticks(rotation='vertical')
#             plt.tight_layout()
#             #plt.xlabel(''); plt.ylabel('')
#             plt.show(block = 0)
#             input()
# # def get_mix_correlations(categorical, numerical):
# #     cols_cat  = categorical.columns
# #     cols_num = numerical.columns
# #     new = pd.DataFrame(columns = cols_num, index = cols_cat)
# #     for row in cols_cat:
# #         for col in cols_num:
# #             t = [num[cat == label].to_numpy() for label in unique]
# #             res = kruskal(*t, nan_policy = 'omit').pvalue if len(unique) > 1 else np.nan
# #             new[col][row] = round(res, 1)
# #     return new
            
    
# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2/n
#     r, k = confusion_matrix.shape
#     phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
#     rcorr = r-((r-1)**2)/(n-1)
#     kcorr = k-((k-1)**2)/(n-1)
#     corr = min((kcorr - 1), (rcorr - 1))
#     return np.sqrt(phi2corr / corr) if corr != 0 else np.nan
