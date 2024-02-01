from gioconda._data import nl
import os, sys
import inspect
import openpyxl
from datetime import datetime

# File Utilities

def read_text(path, log = True):
    print("reading text lines in", path) if log else None
    with open(path, 'r', encoding = "utf-8") as file:
        text = file.readlines()
    text = [line.replace('\ufeff', '') for line in text if line != nl]
    print("text lines read!\n") if log else None
    return text

def split_lines(lines, delimiter = ','):
    return [line.replace("\n", "").split(delimiter) for line in lines]

def read_csv_(path, delimiter = ',', log = True):
    lines = read_text(path, log = log)
    matrix = split_lines(lines, delimiter)
    return matrix

def read_xlsx_(path, log = True):
    print("reading excel file in", path) if log else None
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    matrix = []
    for row in sheet.iter_rows(min_row = 1, max_row = sheet.max_row, min_col = 1, max_col = sheet.max_column):
        line = []
        for cell in row:
            el = cell.value
            el = el.strftime('%d/%m/%Y') if isinstance(el, datetime) else str(el)
            line.append(el)
        matrix.append(line)
    workbook.close()
    return matrix

def write_text(path, text, log = True):
    print("writing text in", path) if log else None
    file = open(path, "w")
    file.write(text)
    file.close()
    print("text written.\n") if log else None

