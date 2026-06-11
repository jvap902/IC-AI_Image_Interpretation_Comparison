import csv
import ast
import numpy as np
import pandas as pd
from src.codifications import modelCod

def writeCsvLine(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(data)

def findInCsv(file_path, params, values):
    if(len(params) != len(values)):
        raise ValueError("The number of parameters should be the as the number of values serched")
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = list(csv.DictReader(file))
        
        ans = []
        
        for row in reader:
            all_equal = True
            
            for idx, param in enumerate(params):
                if (str(values[idx]) != row[param]):
                    all_equal = False
                    break
            
            if all_equal:
                ans.append(row)
                
    return ans

def getStringIntArray(string):
    
    string = string[1:-1]
    ans = [int(num) for num in string.split(', ')]
    
    return ans

def getStringStrArray(string):    
    
    return ast.literal_eval(string)

if __name__ == '__main__':
    pass