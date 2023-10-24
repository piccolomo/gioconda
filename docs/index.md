# Bongo

Bongo is a package to simplify **tabular data manadgment**. 

The tool is a wrapper around numpy at the moment and it is at a very early stage of developement. Its future will depend on the comunity inputs.

For simplicity, there are only three data types that bongo recognize: **categorical, numerical and datetime**. 

At the start all data is categorical and can be imported from text file using the `data = read_data(file_name, delimiter, header)` function. Setting `header = True` tells bongo to use the first row for the columns names.

If the column names are not included in the text file, they can be set with the `set_names(names)` method. 

Once imported, each column can be addressed by name with `data.col('name')` or by index with `data.col(index)`.

These are some relavent methods for each column:

- `to_numerical()` turns data in numerical form
- `to_datetime(form, delta_form)` turns data in datetime form; `delta_form` is used to represent time differences and could be `'seconds', 'minutes', 'hours', 'days', 'months', 'years'`.
- `strip()` is used to remove initial and final spaces in categorical data
- `replace(old, new)` to replace and old string with a new one in categorical data
- `numerical_info()` will return basic numerical information (like mean, std, median etc) as a dictionary. Alternattivelly the `mean()`, `std()` ... functions could be used directly.
- `plot(bins)` is used to plot the data using matplotlib. If the data is categorical the plot will be an histogram.
- `count(norm)` to count how many times an element appears; norm is used to normalize the result.
- `count_nan(norm)` to count nan values; note that Bongo makes no distinction between  None, 'nan', numpy.nan and numpy.nat. 
- `counts(norm)` to show all counts
- `print_counts(norm)` to print the counts in a nicelly formatted form
- `unique()` to show all unique values
- `distinct` to show the length of unique values


These are some relavent methods for the overall data:

- `get_section(rows, cols, nan, string)` will return a portion of the data set; nan is used to include or not nan values and string to return the result in string form (usefull mostly for datetime objects).
- `strip(cols)` and `replace(old, new, cols)` similarly to the column based counterpart but applied to all columns or to the columns specified
- `numerical_info()` to print the numerical info for all numerical columns
- `categorical_info()` to print all caunts for all categorical columns
- `tab(column1, column2, norm, length)` to print the cross counts for the columns specified. At least one has to be categorical and if the other is numerical or datetime, the mean and std for each unique value will be shown.
- `cross_plot(column1, column2)` to plot one columns against another
- `equal(col, value), not_equal, greater, lower`, to select a partion of the data set based on the condition provided
- `part(start, end)` to select a portion of the data set from given start and end rows
- `print(rows, cols, header, index, decimals)` to print the data; rows and cols can be specified as well as to include or not the column names with header, add the row indexes with index and the decimal points. 