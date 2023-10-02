from utility.file import *
from utility.data import *
# from utility.data import *

# Read CSV
data = read_data('data')
data.set_names(['tenancy', 'people', 'frequency', 'start', 'end', 'duration', 'status', 'type', 'sub_type', 'property', 'category', 'region', 'area', 'class', 'construction', 'style', 'built', 'surface', 'value', 'beds', 'gender', 'birthday', 'ethnicity', 'nationality', 'religion', 'caution', 'hoarding', 'financial', 'safeguarding', 'vulnerable', 'antisocial', 'impairment'])

data = data.part(0, 100)

print('cleaning')
data.replace('"','')
data.strip()
data.replace('\ufeff','')
data.replace('NULL', 'nan')
print('data cleaned!\n')

print('converting')
data.to_integer('tenancy')
data.to_integer('people')
data.to_integer('duration')
data.to_integer('built')
data.to_integer('surface')
data.to_float('value')
data.to_integer('beds')
data.to_integer('caution')
data.to_integer('hoarding')
data.to_integer('financial')
data.to_integer('safeguarding')
data.to_integer('vulnerable')
data.to_integer('antisocial')
 
data.to_datetime('start', '%Y-%m-%d', 'years')
data.to_datetime('end', '%Y-%m-%d', 'years')
data.to_datetime('birthday', '%Y-%m-%d', 'years')
print('data converted!\n')


# data.print(rows=range(10), cols=range(5))
# # data = [list(map(clean_string, row)) for row in data]
# # data = transpose(data)
# # columns =  

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

