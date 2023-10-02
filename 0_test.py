from utility.file import *
from utility.data import *
# from utility.data import *

# Read CSV
data = read_data('sample', header = 1)

#data = data.part(0, 100)

data.to_integer('n1')
data.to_float('n2')

data.to_datetime('d1', '%Y-%m-%d', 'years')
data.to_datetime('d2', '%Y-%m-%d', 'years')


# # data.print(rows=range(10), cols=range(5))
# # # data = [list(map(clean_string, row)) for row in data]
# # # data = transpose(data)
# # # columns =  

# # data[0] = list(map(to_int, data[0])) # tenancy
# # data[1] = list(map(to_int, data[1])) # people
# # data[5] = list(map(to_int, data[5])) # duration
# # data[16] = list(map(to_int, data[16])) # built
# # data[17] = list(map(to_int, data[17])) # surface
# # data[18] = list(map(to_int, data[18])) # value
# # data[19] = list(map(to_int, data[19])) # beds
# # data[25] = list(map(to_int, data[25])) # caution
# # data[26] = list(map(to_int, data[26])) # hoarding
# # data[27] = list(map(to_int, data[25])) # financial_risk
# # data[28] = list(map(to_int, data[25])) # safeg
# # data[29] = list(map(to_int, data[25])) # vulner
# # data[30] = list(map(to_int, data[25])) # anti

# # data = transpose(data)

# # data.replace('nan', np.nan, inplace = True)
# # #data = data.loc[ : 10000]
# # l = len(data)


# # # Add Columns
# # data.columns =


# # # Convert Dates 
# # data.start = data.start.apply(to_datetime)
# # data.end = data.end.apply(to_datetime)
# # data.birthday = data.birthday.apply(to_datetime)
# # data.duration = data.duration / 365.25
# # #data.beds = data.beds.apply(to_int) # nan is a float64

# # # Strip Spaces
# # data.property = data.property.apply(lambda string: string.strip())
# # data.frequency = data.frequency.apply(lambda string: string.strip())

# # data.caution = data.caution.apply(to_bool)
# # data.hoarding = data.hoarding.apply(to_bool)
# # data.financial_risk = data.financial_risk.apply(to_bool)
# # data.safeguarding = data.safeguarding.apply(to_bool)
# # data.vulnerable = data.vulnerable.apply(to_bool)
# # data.antisocial = data.antisocial.apply(to_bool)

# # # Write Pickle
# # write_pickle(data_pickle, data)

