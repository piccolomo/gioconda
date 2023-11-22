Loading Data
============


Text Data
---------

Data can be imported from text file (like :file:`.txt` or :file:`.csv`) using the :meth:`read()` method.

.. code:: python

    import gioconda

    path = gioconda.join(b.script_folder(), 'data.csv')
    data = gioconda.read(path, delimiter = ',', header = False, log = True)

As one would except, the `delimiter` parameter specifies how columns are separated in the text file, while `header` specifies if column headers are included. If column headers are included, gioconda recognizes them as column names. 

To manually set the column names, use the method ``data.set_names(names)``, where names is a list of strings. 


Test Data
---------

To load a test data set use:

.. code:: python

    import gioconda

    data = gioconda.read(gioconda.test_data_path, header = 1)

which will be represented as:

.. code-block:: console

   rows  │cols
   67    │7   

   i  │column  │type       
   0  │c1      │categorical
   1  │c2      │categorical
   2  │c3      │categorical
   3  │d1      │categorical
   4  │d2      │categorical
   5  │n1      │categorical
   6  │n2      │categorical


Simulate Data
-------------
You could also simulate data this way:

.. code-block:: python
   
   import gioconda

   data = gioconda.matrix_class()
   data.set_rows(40)

   data.simulate_numerical('duration', mean = 2560, std = 3, nan_ratio = 0.1)
   data.simulate_categorical('frequency', length = 12, nan_ratio = 0.1)
   data.simulate_datetime('start', mean = "01/01/2004", std = 6, form = "%d/%m/%Y", delta_form = "days", nan_ratio = 0.1)

.. code-block:: console

   rows  │cols
   40    │3   

   i  │column     │type       
   0  │duration   │numerical  
   1  │frequency  │categorical
   2  │start      │datetime 

The meaning of the parameters should be intuitive: ``nan_ratio`` is the proportion of ``nan`` values, ``length``  the number of random string categories, ``std`` the standard deviation; ``delta_form`` is used to interpret the standard deviation for datetime columns and could be either ``years, months, days, hours, minutes, seconds``.


Getting Data
------------
To get the desired portion of data as a numpy array, use the method ``data.get_section()`` which accepts the following parameters:

- ``rows`` the list of rows.
- ``cols`` the list of columns (integers or column string names or both).
- ``nan = True`` whatever to include ``nan`` values.
- ``string = False`` whatever to turn the data to strings. 
- ``index = False`` whatever to include the row index number for each entry.