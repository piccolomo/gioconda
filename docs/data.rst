Column Management
=================


Accessing a Column
------------------

Once the data is imported, each column can be addressed by with the ``data.column(col)`` method, where ``col`` could be the column index integer or its string name.

Its representation would look something like this (for categorical columns):

.. code-block:: console

    name    │c1         
    index   │0          
    type    │categorical
    rows    │67         
    nan     │0          
    unique  │3          

     CategoryA, CategoryB, CategoryA, CategoryC, CategoryB, CategoryA, CategoryC, CategoryA, CategoryB, CategoryC ... CategoryA, CategoryB, CategoryA, CategoryC, CategoryB, CategoryA, CategoryC, CategoryB, CategoryA, CategoryC

or like this, for numerical columns:

.. code-block:: console

    name     │n1       
    index    │5        
    type     │numerical
    rows     │67       
    nan      │0.0      
    unique   │65       
    min      │1.62     
    max      │4.78     
    span     │3.16     
    mean     │3.27     
    median   │3.14     
    mode     │1.62     
    std      │0.83     
    density  │26.32    
    
     3.14, 2.71, 1.62, 4.01, 2.18, 3.33, 2.92, 2.05, 1.99, 4.78 ... 3.62, 3.08, 2.53, 4.02, 3.28, 2.34, 4.46, 3.9, 2.55, 3.18


To access the previous information, as a dictionary, use the column ``info()`` or ``numerical_info()`` methods. For specific information access its ``min(), man(), mean(), count(), count_nan()`` etc.. methods. 


Column Operations
-----------------
To modify one particular column use one of the following methods:

For categorical columns:

- ``strip()`` to remove initial and final spaces in each entry.
- ``replace(old, new)`` to replace and ``old`` string with a ``new`` one.

For numerical columns:

- ``multiply(k)`` to multiply the data by a constant


Column Visualization
--------------------
Use the ``plot(bins)`` method to plot a column using ``matplotlib``. The ``bins`` parameter is used to create a histogram for numerical columns.


Cross Columns 
-------------
To get information relative to two columns, use:

- ``tab(column1, column2, norm, length)`` to print the cross counts for the columns specified. At least one has to be categorical and if the other is numerical or datetime, the mean and std for each unique value will be printed instead.
- ``cross_plot(column1, column2)`` to plot one columns against another.