# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:42 2026

@author: alex283h
"""
from joblib import load
import pandas as pd

# predict by point
def predict(x, y, source):
    if source not in ['top', 'bottom']:
        print("You should set top/bottom source!")
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):    
        print("You should set int/float for x and y!")
        return None
    # load models
    if source == "top":
        model = load("top_model.joblib")
    if source == "bottom":
        model = load("bottom_model.joblib")
    x = x / 3200
    y = y / 1800
    result = model.predict(pd.DataFrame([[x, y]], columns=["input_x", "input_y"]))[0].astype(int)
    return result

# demonstration of work
if __name__ == "__main__":
    x, y = 100, 150
    top_predictions = predict(x = x, y = y, source = "top")
    print(f"Predictions for top x = {x}, y = {y}: pr_x = {top_predictions[0]}, pr_y = {top_predictions[1]}")
    x, y = 1000, 1500
    bottom_predictions = predict(x = x, y = y, source = "bottom")
    print(f"Predictions for bottom x = {x}, y = {y}: pr_x = {bottom_predictions[0]}, pr_y = {bottom_predictions[1]}")
    
    
    
    
    
    
    

