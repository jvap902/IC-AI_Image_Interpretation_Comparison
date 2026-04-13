import json
from .defaultPaths import jsonInfoPath
import numpy as np
import pandas as pd
from ..codifications import *

info_json_path = jsonInfoPath()

def updateJson(fields, values, json_path=info_json_path):
    with open(json_path, "r+") as f:
        json_data = json.load(f)
        
        for idx, field in enumerate(fields):
            json_data[field] = values[idx]
        
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()

def getJsonInfo(fields=[], json_path=info_json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    if len(fields) == 0:
        return json_data
    
    data = []

    for field in fields:
        data.append(json_data[field])
        
    return data
    
if __name__ == "__main__":
    pass