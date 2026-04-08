# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:58:22 2026

@author: alex283h
"""

import json
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np

# Load a list of paths train or val folders (sessions)
def load_split_list(json_path, split = "train"):
    """
    Load a list of paths for the given split from a JSON file.

    Args:
        json_path (str): Path to the JSON file.
        split (str): Split name, for example "train" or "val".

    Returns:
        list[str]: List of paths for the selected split.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(f"Key '{split}' was not found in the JSON file.")

    split_list = data[split]

    if not isinstance(split_list, list):
        raise ValueError(f"Value for '{split}' must be a list.")
    
    if '../' in json_path:
        split_list = ['../' + path for path in split_list]

    return split_list

# Load coordinates JSON for the given sample (session) directory
def load_coords_json(sample_dir, view = "top", verbose = True):
    """
    Load coordinates JSON for the given sample directory.

    Args:
        sample_dir (str | Path): Path to the sample directory.
        view (str): Either "bottom" or "top".

    Returns:
        list[dict]: Parsed JSON data.
    """
    sample_dir = Path(sample_dir)

    if view not in {"bottom", "top"}:
        raise ValueError("Parameter 'view' must be either 'bottom' or 'top'.")

    json_path = sample_dir / f"coords_{view}.json"
    
    if not json_path.exists():
        if verbose:
            print(f"File was not found: {json_path}")
            print("sample_dir:", sample_dir)
            print("resolved:", sample_dir.resolve())
            print("dir exists:", sample_dir.exists())
            print("is dir:", sample_dir.is_dir())
            json_path = sample_dir / "coords_top.json"
            print("json_path:", json_path)
            print("json exists:", json_path.exists())
        return []
    
    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # fix paths to images
    for item in json_data:
        file1 = Path(item["file1_path"])
        file2 = Path(item["file2_path"])
    
        item["file1_path"] = str(Path(sample_dir) / file1.parent.name / file1.name)
        item["file2_path"] = str(Path(sample_dir) / file2.parent.name / file2.name)
        
        item["file1_path"] = item["file1_path"].replace("\\", "/")
        item["file2_path"] = item["file2_path"].replace("\\", "/")
        
    return json_data

# extract inputs and outputs of a model
def build_points_dataframe(train_folders, view = "bottom", verbose = True):
    """
    Build a dataframe with matched input/output point pairs
    from annotation JSON files.

    Args:
        train_folders (list[str]): List of sample directory paths.
        view (str): View name used to load JSON annotations,
            for example "bottom" or "top".

    Returns:
        pd.DataFrame: Dataframe with matched points and file paths.

        Columns:
            - input_file
            - output_file
            - point_number
            - input_x
            - input_y
            - output_x
            - output_y
    """
    data_table = []
    hs = []
    ws = []

    for train_folder in train_folders:
        
        train_examples = load_coords_json(sample_dir = train_folder, view=view, verbose = verbose)
        
        for train_example in train_examples:
            # Get input and output image paths
            output_file = train_example["file1_path"]
            input_file = train_example["file2_path"]
            with Image.open(input_file) as img1, Image.open(output_file) as img2:
                if img1.size != img2.size:
                    if verbose:
                        print(f"Error in image shapes: {input_file}, {output_file}!")
                else:
                    hs.append(img1.size[0])
                    ws.append(img1.size[1])
            
            # Get point lists
            output_points = train_example["image1_coordinates"]
            input_points = train_example["image2_coordinates"]

            # Create a lookup table for input points by point number
            input_points_by_number = {point["number"]: point for point in input_points}

            # Collect matched point pairs (filter skipped)
            for op in output_points:
                ip = input_points_by_number.get(op["number"])
                if ip is None:
                    p_idx = op["number"]
                    if verbose:
                        print(f"Point {p_idx} has not been found in {view} image!")
                    continue
                raw = {"input_file": input_file,
                       "output_file": output_file,
                       "point_number": op["number"],
                       "input_x": ip["x"],
                       "input_y": ip["y"],
                       "output_x": op["x"],
                       "output_y": op["y"]}
                data_table.append(raw)
    
    # test hs and ws
    if verbose:
        print(f"Image height: {np.unique(hs)}")
        print(f"Image width: {np.unique(ws)}")
        
    return pd.DataFrame(data_table)

# for debugging
if __name__ == "__main__":
    train_folders = load_split_list(json_path = "../split.json", split = "train")
    r = build_points_dataframe(train_folders, "top")
    # h = 1800, w = 3200
    
    