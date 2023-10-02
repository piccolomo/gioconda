from utility.file import *
from utility.data import *
from matplotlib import pyplot as plt

# Read Picke
data = read_pickle(data_pickle)
# l = len(data)

numerical = get_numerical_frame(data)
datetime = get_datetime_frame(data)
categorical = get_categorical_frame(data)

numerical_descr = describe_numerical_frame(numerical)
datetime_descr = describe_datetime_frame(datetime)
categorical_descr = describe_categorical_frame(categorical)

print_frame(numerical_descr)
print_frame(datetime_descr)
print_multiindex_frame(categorical_descr)


