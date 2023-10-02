from utility.file import *
from utility.data import *
from utility.date import *
from matplotlib import pyplot as plt

# Read Picke
data = read_pickle(data_pickle)
l = len(data)
data = data[data.status == 'terminated']
data.drop(['tenancy', 'property', 'status'], axis = 1, inplace = True)
data.reset_index(inplace = True, drop=True)

numerical = get_numerical_frame(data)
categorical = get_categorical_frame(data)
datetime = get_datetime_frame(data)
numerical1 = hstack_frames(numerical, datetime)
numerical2 = hstack_frames(numerical, datetime_to_numerical(datetime))

#print_frame(get_numerical_correlations(numerical))

#corr = get_categorical_correlations(categorical)
# for col in corr.columns:
#     print_frame(pd.DataFrame(corr[col]))

#numerical = hstack_frames(numerical, datetime)
#get_mix_correlations(categorical, numerical2)

num_cols = [el for el in  numerical2.columns if el != 'duration']
cat_cols = [el for el in  categorical.columns]

from scipy.stats import binned_statistic as bs
bins = 10
for col in num_cols:
    #col = 'start'
    print(col)
    data = numerical1[col]
    unique = data.unique()
    unique = unique[~np.isnan(unique)]
    lu = len(unique)
    bins = min(lu, 30)
    m, M = data.min(), data.max()
    span = M - m
    dx = span / (bins - 1)
    center = [m + dx * i for i in range(bins)]
    data_binned = [[numerical1.duration.loc[i] for i in data.index if c - dx / 2 <= data[i] < c + dx / 2] for c in center]
    plt.clf()
    plt.boxplot(data_binned, showfliers = False, positions = center)
    plt.xlabel(col)
    plt.ylabel('duration')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.show(block = 0)
    input('press')
    

for col in cat_cols:
    print(col)
    plt.clf()
    g = data.groupby(col)
    df = pd.DataFrame({col: vals['duration'] for col, vals in g})
    df.columns = [str(el) for el in df.columns]
    df = df.iloc[:, :15]
    meds = df.median()
    meds.sort_values(ascending=False, inplace=True)
    df = df[meds.index]
    df.boxplot(rot = 90, grid = 0, showfliers=False)
    plt.xlabel(col)
    plt.ylabel('duration')
    #plt.xticks(rotation = 'vertical')
    plt.tight_layout()
    plt.show(block = 0)
    input()
