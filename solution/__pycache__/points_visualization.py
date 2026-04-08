# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:58:22 2026

@author: alex283h
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# add one point on an image
def add_point_on_image(image, x, y, color = (0, 0, 255), radius = 5, thickness = -1):
    """
    Add one point on an image

    Args:
        image: numpy-array of an image
        x, y: coordinates
        color: color in rgb
        radius: radius of the point
        thickness: thickness of the point.

    Returns:
        new image with one point
    """
    result = image.copy()
    cv2.circle(result, (int(x), int(y)), radius, color, thickness)
    return result

# for visualization any train/val example
if __name__ == "__main__":
    from main_functions import load_split_list, load_coords_json
    train_folders = load_split_list(json_path = "../split.json", split = "train")
    train_examples = load_coords_json(sample_dir = train_folders[0], view = 'top')
    print(f"The number of train examples: {len(train_examples)}")
    # take one case
    train_example = train_examples[2]
    file_img1 = train_example['file1_path']
    file_img2 = train_example['file2_path']
    # load images
    img1 = cv2.cvtColor(cv2.imread(filename = file_img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(filename = file_img2), cv2.COLOR_BGR2RGB)
    # add points on images
    points1 = train_example['image1_coordinates']
    points2 = train_example['image2_coordinates']
    print(f"The number points on image 1: {len(points1)}")
    print(f"The number points on image 2: {len(points2)}")
    # matching points
    for p1 in points1:
        for p2 in points2:
            if p1['number'] == p2['number']:
                break
        if p1['number'] != p2['number']:
            continue
        # set color of the point
        cmap = plt.get_cmap("tab20")
        r, g, b, _ = cmap((p1['number'] - 1) % cmap.N)
        b, g, r = int(b * 255), int(g * 255), int(r * 255)
        # add points
        img1 = add_point_on_image(image = img1, 
                                  x = p1['x'], 
                                  y= p1['y'], 
                                  radius = 30, 
                                  color = (b, g, r))
        img2 = add_point_on_image(image = img2, 
                                  x = p2['x'], 
                                  y= p2['y'], 
                                  radius = 30,
                                  color = (b, g, r))
    
    plt.imshow(img1)
    plt.show()
    
    plt.imshow(img2)
    plt.show()
    
    
    
    
    
