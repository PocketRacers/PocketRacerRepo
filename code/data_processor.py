from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import time
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
import joblib
import matplotlib.pyplot as plt

def load_3d_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):

    datas = []
    files = os.listdir(img_path)
    os.chdir(img_path)
    steering = []
    throttle = []
    image = []
    for i in range(len(files)):
        f = files[i]
        if f[:5] == "video":
           ending = f[5:]
           body = ending[:-4]
           datas.append(f)
    for j in range(len(datas)):
          cap = cv2.VideoCapture(datas[j])
          fourcc = cv2.VideoWriter_fourcc(*'RGBA')
          cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          f = 0  
          frame_prev = None 
          while cap.isOpened():
            try:     
              if f == 0:
                 ret_prev, frame_prev = cap.read()      
                 frame_prev[31,31,0] = 0
              ret, frame = cap.read()
              if frame is None:
                break
              steering_val = frame[31,31,0]/100-1
              frame[31,31,0] = 0
              f += 1                       
              if (frame is not None) and (frame_prev is not None): # Clean up data  
                      frame_stacked = np.concatenate((frame_prev, frame), axis=2)  
                      frame_prev = frame
                      image.append(frame_stacked) 
                      steering.append(steering_val)   
                      throttle.append(0.0)              
            except Exception as e:
              print(e)
              pass
              
    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
        
    val_images, val_heading, val_steering, val_throttle = test_images, test_heading, test_steering, test_throttle = empty_images, empty_heading, empty_steering, empty_throttle = train_images, train_heading, train_steering, train_throttle = None, None, None, None,    
    if test_size is not None:
       train_bag, test_bag, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                image, steering, throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    if val_size is not None:
       train_bag, val_bag, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                train_bag, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    image = []
    steering = []
    throttle = []	      
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle  

