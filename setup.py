# /usr/bin/env python3
import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    author = "Savino Piccolomo",
    author_email = "piccolomo@gmail.com",
    name = 'gioconda',
    version='1.6.0',
    description = 'handles tabular data',
    long_description = README,
    long_description_content_type = "text/markdown",  
    license = "MIT",
    url = 'https://github.com/piccolomo/gioconda',
    packages = find_packages(),
    python_requires = ">=3.10.0, <=3.11.4",
    include_package_data = True,
    install_requires = ["numpy >= 1.24.0", "matplotlib >= 3.7.0"],
    classifiers = []
    )
