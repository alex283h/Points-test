# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 12:15:17 2026

@author: alex283h
"""

from main_functions import build_points_dataframe, load_split_list
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from joblib import dump

# get main metric
def get_med(true_xy, pred_xy):
    """
    Compute mean Euclidean distance between pairs of 2D points.

    Args:
        true_xy: Array-like of shape (N, 2)
        pred_xy: Array-like of shape (N, 2)

    Returns:
        float: Mean Euclidean distance.
    """
    true_xy = np.asarray(true_xy)
    pred_xy = np.asarray(pred_xy)

    distances = np.sqrt(np.sum(np.square(true_xy - pred_xy), axis=1))
    med = np.mean(distances)
    return float(med)

if __name__ == "__main__":
    # get train and val fodlers
    train_folders = load_split_list(json_path = "../split.json", split = "train")
    val_folders = load_split_list(json_path = "../split.json", split = "val")
    # create train and val data sets for top and bottom data
    train_dataset_top = build_points_dataframe(train_folders, "top", verbose = False)
    print(f"Train dataset top shape: {train_dataset_top.shape}")
    train_dataset_bottom = build_points_dataframe(train_folders, "bottom", verbose = False)
    print(f"Train dataset bottom shape: {train_dataset_bottom.shape}")
    val_dataset_top = build_points_dataframe(val_folders, "top", verbose = False)
    print(f"Validation dataset top shape: {val_dataset_top.shape}")
    val_dataset_bottom = build_points_dataframe(val_folders, "bottom", verbose = False)
    print(f"Validation dataset bottom shape: {val_dataset_bottom.shape}")
   
    # load train and val data set for "top" model
    train_input_xy = train_dataset_top[["input_x", "input_y"]] / (3200, 1800)
    train_output_xy = train_dataset_top[["output_x", "output_y"]] 
    val_input_xy = val_dataset_top[["input_x", "input_y"]] / (3200, 1800)
    val_output_xy = val_dataset_top[["output_x", "output_y"]]
    top_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=15,
                                                         weights="uniform",
                                                         metric="euclidean"))
    # train model
    top_model.fit(train_input_xy, train_output_xy)
    # get predictions and metric
    pred_xy = top_model.predict(val_input_xy)
    # clipping values
    pred_xy[:, 0] = np.clip(pred_xy[:, 0], 0, 3200)
    pred_xy[:, 1] = np.clip(pred_xy[:, 1], 0, 1800)
    med = get_med(val_output_xy, pred_xy)
    print(f"Train MED = {med:.4f}") 
    
    # load train and val data set for "bottom" model
    train_input_xy = train_dataset_bottom[["input_x", "input_y"]] / (3200, 1800)
    train_output_xy = train_dataset_bottom[["output_x", "output_y"]] 
    val_input_xy = val_dataset_bottom[["input_x", "input_y"]] / (3200, 1800)
    val_output_xy = val_dataset_bottom[["output_x", "output_y"]]
    bottom_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=15,
                                                         weights="uniform",
                                                         metric="euclidean"))
    # train model
    bottom_model.fit(train_input_xy, train_output_xy)
    # get predictions and metric
    pred_xy = bottom_model.predict(val_input_xy)
    # clipping values
    pred_xy[:, 0] = np.clip(pred_xy[:, 0], 0, 3200)
    pred_xy[:, 1] = np.clip(pred_xy[:, 1], 0, 1800)
    med = get_med(val_output_xy, pred_xy)
    print(f"Bottom MED = {med:.4f}") 
    
    # save models for inference
    dump(top_model, "top_model.joblib")
    dump(bottom_model, "bottom_model.joblib")