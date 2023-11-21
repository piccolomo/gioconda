Data Management
===============


Column Types
------------

When data is initially loaded, all columns are interpreted as categorical. To specify the columns types and interpret the data accordingly, follow this example:


.. code:: python

   import bongo 

   data = bongo.read(bongo.test_data_path, header = True)

   data.to_numerical('n1')
   data.to_numerical('n2')

   data.to_datetime('d1', form = '%Y-%m-%d')
   data.to_datetime('d2', form = '%Y-%m-%d')

In the previous example ``form`` is a parameter used to interpret datetime objects. Now the data is represented as:

.. code-block:: console

   rows  │cols
   67    │7   

   i  │column  │type       
   0  │c1      │categorical
   1  │c2      │categorical
   2  │c3      │categorical
   3  │d1      │datetime   
   4  │d2      │datetime   
   5  │n1      │numerical  
   6  │n2      │numerical 



Numerical Info
--------------
To get at a glance all relevant information on the numerical (and datetime) columns use the ``data.numerical_info()`` method, which for the test data would print the following table:

.. code-block:: console

            │d1          │d2          │n1     │n2   
   min      │2023-01-15  │2023-02-20  │1.62   │0.4  
   max      │2034-01-30  │2034-02-25  │4.78   │2.5  
   span     │11.0        │11.0        │3.16   │2.1  
   nan      │0.0         │0.0         │0.0    │0.0  
   mean     │2028-07-16  │2028-08-18  │3.27   │1.36 
   median   │2028-07-04  │2028-08-06  │3.14   │1.3  
   mode     │2023-01-15  │2023-02-20  │1.62   │0.4  
   std      │3.2         │3.2         │0.83   │0.62 
   density  │29.2        │29.27       │26.32  │29.66



Categorical Info
----------------
To get at a glance all relevant information on the numerical columns use the ``data.categorical_info()`` method, which will psrint the counts for each categorical column. It accept the following parameters:

- ``cols`` the list of categorical columns (integers or column string names or both).
- ``norm = False`` whatever to normalize or not the counts.
- ``length = 10`` the maximum number of categories to show (useful for data with a really large number of categories).

For the test data this returns:

.. code-block:: console

    c1         │count
    CategoryA  │25   
    CategoryB  │22   
    CategoryC  │20   
    
    c2      │count
    Blue    │17   
    Green   │17   
    Red     │17   
    Yellow  │16   
    
    c3   │count
    No   │25   
    Yes  │42  
 

Selection
---------
One could easily select a subset of the original dataset using the following methods:

- ``data.equal(col, value)``: to select all the rows where the given column is equal to the given value.
- ``data.not_equal(col, value)``: to select all the rows where the given column is not equal to the given value.
- ``data.greater(col, value, equal = True)``: to select all the rows where the given column is greater (or optionally equal) to the given value.
- ``data.lower(col, value, equal = True)``: to select all the rows where the given column is lower (or optionally equal) to the given value.
- ``data.subset(rows)``: to select the specified rows.
- ``data.part(start, end)``: to select the rows from ``start`` to ``end``.
- ``data.copy()``: to copy the entire dataset. 

