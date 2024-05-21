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
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from itertools import chain
import random
import bisect
import scipy
import statistics as st
from sklearn import preprocessing
from skimage.transform import warp_polar

def load_maml_dataset(n_stacked, support_img_path, support_out_path, query_img_path, query_out_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
                         
    df = pd.read_csv(support_out_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    yaw_1 = []
    yaw_2 = []
    yaw_3 = []        
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']           
        img_final = []
        img = fname        
        try:
           x.append(img[:])
        except: 
           print(img)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle)))         
    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std       
    steering = MinMaxScaler(steering)
    throttle = MinMaxScaler(throttle)
    x = np.stack(x[:])
    for r in range(len(steering)):
        steering[r] = np.stack([round(steering[r],1)]) 
        throttle[r] = np.stack([round(throttle[r],1)])           
    test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None,    	      
    if test_size is not None:
       train_images, test_images, train_steering, test_steering, train_throttle, test_throttle= train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    shuffle(train_images, train_steering, train_throttle)
    
    train_images_a = train_images
    train_steering_a = train_steering
    train_throttle_a = train_throttle
    test_images_a = test_images
    test_steering_a = test_steering
    test_throttle_a = test_throttle
    
    # Rinse and repeat 
    train_images = []
    train_steering = []
    train_throttle = []
    test_images = []
    test_steering = []
    test_throttle = []
    
    df = pd.read_csv(query_out_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    yaw_1 = []
    yaw_2 = []
    yaw_3 = []        
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']           
        img_final = []
        img = fname        
        try:
           x.append(img[:])
        except: 
           print(img)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle)))         
    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std       
    steering = MinMaxScaler(steering)
    throttle = MinMaxScaler(throttle)
    x = np.stack(x[:])
    for r in range(len(steering)):
        steering[r] = np.stack([round(steering[r],1)]) 
        throttle[r] = np.stack([round(throttle[r],1)])           
    test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None,    	      
    if test_size is not None:
       train_images, test_images, train_steering, test_steering, train_throttle, test_throttle= train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    train_images, train_steering, train_throttle = shuffle(train_images, train_steering, train_throttle)
    
    train_images_b = train_images
    train_steering_b = train_steering
    train_throttle_b = train_throttle
    test_images_b = test_images
    test_steering_b = test_steering
    test_throttle_b = test_throttle
    
    return train_images_a, test_images_a, train_steering_a, test_steering_a, train_throttle_a, test_throttle_a, train_images_b, test_images_b, train_steering_b, test_steering_b, train_throttle_b, test_throttle_b  
def load_2d_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
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
          print(j)
          cap = cv2.VideoCapture(datas[j])
          fourcc = cv2.VideoWriter_fourcc(*'RGBA')
          cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          f = 0  
          frame_prev = None 
          while cap.isOpened():
            try:     
              #if f == 0:
              #   ret_prev, frame_prev = cap.read()      
              #   frame_prev[31,31,0] = 0
              ret, frame = cap.read()
              if frame is None:
                break
              steering_val = frame[31,31,0]/100-1
              #print(steering_val)
              frame[31,31,0] = 0
              f += 1                       
              if (frame is not None): #and (frame_prev is not None): # Clean up data  
                      #frame_stacked = np.concatenate((frame_prev, frame), axis=2)  
                      #frame_stacked = cv2.resize(frame_stacked[:,:], (32,32), interpolation=cv2.INTER_LINEAR)     
                      #frame_prev = frame
                      image.append(frame) 
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
    #train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle  
      
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
          print(j)
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
              #print(steering_val)
              frame[31,31,0] = 0
              f += 1                       
              if (frame is not None) and (frame_prev is not None): # Clean up data  
                      frame_stacked = np.concatenate((frame_prev, frame), axis=2)  
                      #frame_stacked = np.stack((frame_prev, frame))
                      #frame_stacked = cv2.resize(frame_stacked[:,:], (32,32), interpolation=cv2.INTER_LINEAR)     
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
    #train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle  
  
def load_3d_polar_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
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
          print(j)
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
              #print(steering_val)
              frame[31,31,0] = 0
              f += 1                       
              if (frame is not None) and (frame_prev is not None): # Clean up data  
                      frame_stacked = np.concatenate((frame_prev, frame), axis=2)  
                      frame_prev = frame
                      image.append(cv2.linearPolar(frame_stacked, (16,16), 16, cv2.INTER_NEAREST))
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
    #train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle   
    
def load_rc_3d_polar_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
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
          print(j)
          cap = cv2.VideoCapture(datas[j])
          fourcc = cv2.VideoWriter_fourcc(*'RGBA')
          cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          f = 0  
          frame_prev = None 
          while cap.isOpened():
            try:     
              crash = False
              if f == 0:
                 ret_prev, frame_prev = cap.read()      
              ret, frame = cap.read()
              if frame is None:
                break
              steering_val = (frame[63,63,0]-123)/13
              throttle_val = (frame[63,63,1]-123)/13
              frame_prev[63,63,0] = 0
              frame_prev[63,63,1] = 0              
              frame[63,63,0] = 0
              frame[63,63,1] = 0
              f += 1                       
              if (steering_val == 0.5860000000000001 or steering_val == 0.5840000000000001) and (throttle_val == 0.5700000000000001 or throttle_val == 0.5680000000000001):
                 crash = True
              if crash == False and (frame is not None) and (frame_prev is not None): # Clean up data  
                      frame_stacked = np.concatenate((frame_prev,frame), axis=2)                        
                      frame_prev = frame
                      polar_image = warp_polar(frame_stacked, (32,32,6), radius=32, output_shape=(60,32,6), channel_axis=-1,  scaling='linear', order=0)     
                      #cv2.imshow('adf', polar_image[28:,:,:3])        
                      #cv2.waitKey(0)    
                      image.append(polar_image[28:,:,:])
                      steering.append(steering_val) 
                      throttle.append(throttle_val) 
                      
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
    #train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle 

    
def load_rc_3d_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
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
          print(j)
          cap = cv2.VideoCapture(datas[j])
          fourcc = cv2.VideoWriter_fourcc(*'RGBA')
          cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          f = 0  
          frame_prev = None 
          while cap.isOpened():
            try:     
              crash = False
              if f == 0:
                 ret_prev, frame_prev = cap.read()      
              ret, frame = cap.read()
              if frame is None:
                break
              steering_val = (frame[63,63,0]-123)/13
              throttle_val = (frame[63,63,1]-123)/13
              frame_prev[63,63,0] = 0
              frame_prev[63,63,1] = 0              
              frame[63,63,0] = 0
              frame[63,63,1] = 0
              f += 1                       
              if (steering_val == 0.5860000000000001 or steering_val == 0.5840000000000001) and (throttle_val == 0.5700000000000001 or throttle_val == 0.5680000000000001):
                 crash = True
              if crash == False and (frame is not None) and (frame_prev is not None): # Clean up data  
                      frame_stacked = np.concatenate((frame_prev,frame), axis=2)                        
                      frame_prev = frame
                      imagef = cv2.resize(frame_stacked[:,:], (32,32), interpolation=cv2.INTER_LINEAR)              
                      image.append(imagef)
                      steering.append(steering_val)   
                      throttle.append(throttle_val) 
                      
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
    #train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle 
     
def load_rc_2d_polar_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):

    files = os.listdir(img_path)
    os.chdir(img_path)
    steering = []
    throttle = []
    image = []
    datas =[]
    for i in range(len(files)):
        f = files[i]
        if f[:5] == "video":
           ending = f[5:]
           body = ending[:-4]
           datas.append(f)
    for j in range(len(datas)):
          print(j)
          cap = cv2.VideoCapture(datas[j])
          fourcc = cv2.VideoWriter_fourcc(*'RGBA')
          cap.set(cv2.CAP_PROP_FOURCC, fourcc)
          f = 0  
          frame_prev = None 
          while cap.isOpened():
            try:     
              crash = False
              ret, frame = cap.read()
              if frame is None:
                break
              steering_val = (frame[63,63,0]-123)/13
              throttle_val = (frame[63,63,1]-123)/13
              frame[63,63,0] = 0
              frame[63,63,1] = 0
              f += 1                       
              if (steering_val == 0.5860000000000001 or steering_val == 0.5840000000000001) and (throttle_val == 0.5700000000000001 or throttle_val == 0.5680000000000001):
                 crash = True
              if crash == False and (frame is not None): # Clean up data  
                      polar_image = warp_polar(frame, (32,32,3), radius=32, output_shape=(60,32,3), channel_axis=-1,  scaling='linear', order=0)                 
                      image.append(polar_image[28:,:])
                      steering.append(steering_val)   
                      throttle.append(throttle_val)            
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
    train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle  
    
def load_numpy_dataset(n_stacked, img_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):

    files = os.listdir(img_path)
    os.chdir(img_path)
    steering = []
    throttle = []
    image = []
    datas =[]
    for i in range(len(files)):
        f = files[i]
        if f[:5] == "video":
           ending = f[5:]
           body = ending[:-4]
           datas.append(f)
    for j in range(len(datas)):
       try:
          cap = cv2.VideoCapture(datas[j])
          f = 0
          while cap.isOpened():               
              ret, frame = cap.read()
              if frame is None:
                break
              image.append(cv2.resize(frame, (w,h), interpolation=cv2.INTER_LINEAR)) 
              steering.append(frame[63,63,0]/100 -1)   
              throttle.append(0.0)   
              f += 1      
          if f != control_data.shape[0]:
              print(f"Frame count mismatch: {f} (video) vs {control_data.shape[0]} (controls)!")
              print("The dataset is compromised; delete video with this mismatch")

       except: 
          pass                  
    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
        
    val_images, val_heading, val_steering, val_throttle = test_images, test_heading, test_steering, test_throttle = empty_images, empty_heading, empty_steering, empty_throttle = train_images, train_heading, train_steering, train_throttle = None, None, None, None,    
    if test_size is not None:
       train_bag, test_bag, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                image, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
    if val_size is not None:
       train_bag, val_bag, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                train_bag, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
    image = []
    steering = []
    throttle = []	      
    train_bag, train_steering, train_throttle = shuffle(train_bag, train_steering, train_throttle)
    return train_bag, val_bag, test_bag, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle   
 
def load_time_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    heading = []
    yaw_cos = []
    yaw_sin = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']        
        img_final = []
        img = fname        
        try:
           x.append(img[:])
        except: 
           print(img)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv))
        heading.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle))) 

    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std
        
    #steering = MinMaxScaler(steering)
    #throttle = MinMaxScaler(throttle)
    #heading = MinMaxScaler(heading)
    x = np.stack(x[:])

    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
        heading[r] = np.stack([heading[r]])  
    val_images, val_heading, val_steering, val_throttle = test_images, test_heading, test_steering, test_throttle = empty_images, empty_heading, empty_steering, empty_throttle = train_images, train_heading, train_steering, train_throttle = None, None, None, None,    
    if val_size is not None:
       train_images, val_images, train_steering, val_steering, train_throttle, val_throttle, train_heading, val_heading = train_test_split(
                x, steering, throttle, heading, test_size=test_size,
                random_state=123, shuffle=False
                )
    x = []
    steering = []
    throttle = []	      
    heading = []
    if test_size is not None:
       train_images, test_images, train_steering, test_steering, train_throttle, test_throttle, train_heading, test_heading = train_test_split(
                train_images, train_steering, train_throttle, train_heading, test_size=test_size,
                random_state=123, shuffle=False
                )
    return train_images, val_images, test_images, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle, train_heading, val_heading, test_heading
        
def load_3d_motor_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    heading = []
    yaw_cos = []
    yaw_sin = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']        
        img_final = []
        img = fname        
        try:
           x.append(img[:])
        except: 
           print(img)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv))
        heading.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle))) 

    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std
        
    steering = MinMaxScaler(steering)
    throttle = MinMaxScaler(throttle)
    #heading = MinMaxScaler(heading)
    x = np.stack(x[:])

    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
        heading[r] = np.stack([heading[r]])  
    val_images, val_heading, val_steering, val_throttle = test_images, test_heading, test_steering, test_throttle = empty_images, empty_heading, empty_steering, empty_throttle = train_images, train_heading, train_steering, train_throttle = None, None, None, None,    
    if val_size is not None:
       train_images, val_images, train_steering, val_steering, train_throttle, val_throttle, train_heading, val_heading = train_test_split(
                x, steering, throttle, heading, test_size=test_size,
                random_state=123, shuffle=True
                )
    x = []
    steering = []
    throttle = []	      
    heading = []
    if test_size is not None:
       train_images, test_images, train_steering, test_steering, train_throttle, test_throttle, train_heading, test_heading = train_test_split(
                train_images, train_steering, train_throttle, train_heading, test_size=test_size,
                random_state=123, shuffle=False
                )
    train_images, train_steering, train_throttle, train_heading = shuffle(train_images, train_steering, train_throttle, train_heading)
    return train_images, val_images, test_images, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle, train_heading, val_heading, test_heading
    
def load_ff_motor_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    x2 = []
    z = []
    steering = []
    throttle = []       
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        fname_2 = row['image_2']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']           
        img_final = []
        try:
           x.append(fname[:])
           x2.append(fname_2[:])
        except: 
           print(fname)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle)))         
    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std
        
    steering = MinMaxScaler(steering)
    throttle = MinMaxScaler(throttle)
    x = np.stack(x[:])
    x2 = np.stack(x2[:])
    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
           
    val_images, val_images2, val_sensors, val_steering, val_throttle = test_images, test_images2, test_sensors, test_steering, test_throttle = empty_images,empty_images2, empty_sensors, empty_steering, empty_throttle = train_images, train_images2, train_sensors, train_steering, train_throttle = None,  None, None, None, None,    
    if val_size is not None:
       train_images, val_images, train_images2, val_images2, train_steering, val_steering, train_throttle, val_throttle= train_test_split(
                x, x2, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
    x = []
    x2 = []
    steering = []
    throttle = []	      
    if test_size is not None:
       train_images, test_images, train_images2, test_images2, train_steering, test_steering, train_throttle, test_throttle= train_test_split(
                train_images, train_images2, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    train_images, train_images2, train_steering, train_throttle = shuffle(train_images, train_images2, train_steering, train_throttle)
    return train_images, val_images, test_images, train_images2, val_images2, test_images2,  train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

def load_3d_motor_dataset_inv(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    yaw_1 = []
    yaw_2 = []
    yaw_3 = []        
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        shaft_speed = row['shaft_speed']  
        thv = row['throttle']           
        img_final = []
        img = fname        
        try:
           x.append(img[:])
        except: 
           print(img)
        steering.append(np.float32(stv))
        throttle.append(np.float32(thv)) 
    print("steering Max" + str(max(steering)))
    print("steering Min" + str(min(steering)))
    print("throttle Max" + str(max(throttle)))
    print("throttle Min" + str(min(throttle)))         
    def MinMaxScaler(X):    
        X_std = (X - min(X)) / (max(X) - min(X))
        return X_std
        
    steering = MinMaxScaler(steering)
    throttle = MinMaxScaler(throttle)
    x = np.stack(x[:])

    for r in range(len(steering)):
        steering[r] = np.stack([steering[r]]) 
        throttle[r] = np.stack([throttle[r]])
    
    x = x[1:]
    steering = steering[:-1]
    throttle = throttle[:-1]       
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None,    
    if val_size is not None:
       train_images, val_images, train_steering, val_steering, train_throttle, val_throttle= train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
    x = []
    steering = []
    throttle = []	      
    if test_size is not None:
       train_images, test_images, train_steering, test_steering, train_throttle, test_throttle= train_test_split(
                train_images, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    train_images, train_steering, train_throttle = shuffle(train_images, train_steering, train_throttle)
    return train_images, val_images, test_images, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

   
def load_3d_itdm_motor_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        shaft_time = row['shafttime']
        batt = row['battery']
        '''                  
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        '''
        #if (mini_counter) < 4:
                    #img = cv2.imread(os.path.join(img_path, fname[:-1]))
                    #img = cv2.resize(img,(80,80))
                    #img = img[35:, :]
                    #cv2.imshow(str(img.shape), img)
                    #cv2.waitKey(10000)
                    #cv2.destroyAllWindows() 
        img = os.path.join(img_path, fname[:-1])
        img_stack.append(img)
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = shaft_speed
        if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0 and mini_counter != 1:
                    x.append(img_stack)
                    img_stack = img_stack[n_jump:]
                    ################IMPORTANT!!!! We add 0.5 to make the CUMSUM difference unique!!!!#######################
                    steering.append(np.stack([stv+0.6]))
                    try: 
                       shaft_speed = np.float32(shaft_speed)
                    except:
                       shaft_speed = np.float32(shaft_speed[:-3])  
                    throttle.append(np.stack([shaft_speed/((30.497-1.0534)/2)]))   
                    sen_stack = [batt]
                    z.append(np.stack(sen_stack))
                    thv_prev = shaft_speed
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    shuffle(train_images, train_sensors, train_steering, train_throttle)
    return train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle
def load_3d_sensor_undersampled_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        batt = row['battery']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        if (mini_counter) < 4:
                    img = cv2.imread(os.path.join(img_path, fname[:-1]))
                    #img = cv2.resize(img,(80,80))
                    #img = img[35:, :]
                    cv2.imshow(str(img.shape), img)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows() 
        img = os.path.join(img_path, fname[:-1])
        img_stack.append(img)
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = thv
        if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0 and mini_counter != 1:
                    x.append(img_stack)
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([thv]))
                    sen_stack = [np.float32(stv_prev), np.float32(thv_prev), np.float32(shaft_speed),  np.float32(acc_x), np.float32(acc_y), np.float32(batt)]
                    z.append(np.stack(sen_stack))
                    thv_prev = thv
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    # Start Cropping the Training Set
    bin_lower = []
    bin_upper = []
    train_steering_value = np.squeeze(train_steering) 
    train_sensors_value = np.squeeze(train_sensors)
    train_images_value = np.squeeze(train_images)
    start = train_steering_value[1]
    end = train_steering_value[1]
    for a in range(len(train_steering_value)):
               if train_steering_value[a] > end:  
                     end = train_steering_value[a]
               if train_steering[a] < start:
                     start = train_steering_value[a]
    bin_count = 100
    width = (end - start) / bin_count
    bin_cutoff = 1000
    for a in range(bin_count):
               bin_lower.append(start + width*a) 
               bin_upper.append(start + width*(a+1))            
    add_rows = []   
    for a in range(bin_count):
               counter = 1
               r = list(range(len(train_steering_value)))
               random.shuffle(r)
               for b in r:
                   if (bin_lower[a] < train_steering_value[b]) and (train_steering_value[b] < bin_upper[a]) and (counter <= bin_cutoff):
                       counter = counter + 1
                       add_rows.append(b)
                   if counter >= bin_cutoff:
                       break
    cropped_train_steering = []
    cropped_train_images = []
    cropped_train_throttle = []
    cropped_train_sensors = []
    cropped_train_steering_visual = []
    for a in range(len(add_rows)):
               cropped_train_steering.append(train_steering[add_rows[a]])
               cropped_train_throttle.append(train_throttle[add_rows[a]])
               cropped_train_sensors.append(train_sensors_value[add_rows[a]])
               cropped_train_images.append(train_images_value[add_rows[a]])
               # We need an extra array to show cropped train steering visually.
               cropped_train_steering_visual.append(train_steering_value[add_rows[a]]) 
    print(np.shape(train_steering))
    print(np.shape(np.stack(cropped_train_steering))) 
    print(np.shape(train_images))
    print(np.shape(np.stack(cropped_train_images))) 
    print(np.shape(train_throttle))
    print(np.shape(np.stack(cropped_train_throttle))) 
    print(np.shape(train_sensors))
    print(np.shape(np.stack(cropped_train_sensors))) 
    plt.hist(train_steering_value, bins=100)
    plt.hist(cropped_train_steering_visual, bins=100)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    shuffle(cropped_train_images, cropped_train_sensors, cropped_train_steering, cropped_train_throttle)
    return np.stack(cropped_train_images), np.stack(cropped_train_sensors), val_images, val_sensors, test_images, test_sensors, np.stack(cropped_train_steering), val_steering, test_steering, np.stack(cropped_train_throttle), val_throttle, test_throttle

def load_3d_sensor_oversampled_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        batt = row['battery']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        if (mini_counter) < 4:
                    img = cv2.imread(os.path.join(img_path, fname[:-1]))
                    #img = cv2.resize(img,(80,80))
                    #img = img[35:, :]
                    cv2.imshow(str(img.shape), img)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows() 
        img = os.path.join(img_path, fname[:-1])
        img_stack.append(img)
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = thv
        if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0 and mini_counter != 1:
                    x.append(img_stack)
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([thv]))
                    sen_stack = [np.float16(stv_prev), np.float16(thv_prev), np.float16(shaft_speed),  np.float16(acc_x), np.float(acc_y), np.float16(batt)]
                    z.append(np.stack(sen_stack))
                    thv_prev = thv
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    # Start Cropping the Training Set
    bin_lower = []
    bin_upper = []
    train_steering_value = np.squeeze(train_steering) 
    train_sensors_value = np.squeeze(train_sensors)
    train_images_value = np.squeeze(train_images)
    start = train_steering_value[1]
    end = train_steering_value[1]
    for a in range(len(train_steering_value)):
               if train_steering_value[a] > end:  
                     end = train_steering_value[a]
               if train_steering[a] < start:
                     start = train_steering_value[a]
    bin_count = 100
    width = (end - start) / bin_count
    bin_cutoff = 4000
    for a in range(bin_count):
               bin_lower.append(start + width*a) 
               bin_upper.append(start + width*(a+1))            
    add_rows = []   
    for a in range(bin_count):
               pool = []
               counter = 1
               r = list(range(len(train_steering_value)))
               for b in r:
                   if (bin_lower[a] < train_steering_value[b]) and (train_steering_value[b] < bin_upper[a]) and (counter <= bin_cutoff):
                       counter = counter + 1
                       add_rows.append(b)
                       pool.append(b)
                   if counter >= bin_cutoff:
                       break
               if counter < bin_cutoff:
                   pool_counter = 0
                   while counter <= bin_cutoff: 
                       add_rows.append(pool[pool_counter])
                       counter = counter + 1 
                       pool_counter = pool_counter + 1
                       if pool_counter == len(pool):       
                             pool_counter = 0 
    cropped_train_steering = []
    cropped_train_images = []
    cropped_train_throttle = []
    cropped_train_sensors = []
    cropped_train_steering_visual = []
    for a in range(len(add_rows)):
               cropped_train_steering.append(train_steering[add_rows[a]])
               cropped_train_throttle.append(train_throttle[add_rows[a]])
               cropped_train_sensors.append(train_sensors_value[add_rows[a]])
               cropped_train_images.append(train_images_value[add_rows[a]])
               # We need an extra array to show cropped train steering visually.
               cropped_train_steering_visual.append(train_steering_value[add_rows[a]]) 
    plt.hist(train_steering_value, bins=100)
    plt.hist(cropped_train_steering_visual, bins=100)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    shuffle(cropped_train_images, cropped_train_sensors, cropped_train_steering, cropped_train_throttle)
    return np.stack(cropped_train_images), np.stack(cropped_train_sensors), val_images, val_sensors, test_images, test_sensors, np.stack(cropped_train_steering), val_steering, test_steering, np.stack(cropped_train_throttle), val_throttle, test_throttle


# Data Processor for past action & sensor input alongside 2d images; only the training/val set is shuffled. 
def load_2d_sensor_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        batt = row['battery']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        #if (mini_counter) < 4:
        #            img = cv2.imread(os.path.join(img_path, fname[:-1]))
        #            cv2.imshow(str(img.shape), img)
        #            cv2.waitKey(1)
        #            cv2.destroyAllWindows() 
        img = os.path.join(img_path, fname[:-1])
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = thv
        if mini_counter != 1:	
                    x.append(img)
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([thv]))
                    sen_stack = [np.float16(batt)]
                    z.append(np.stack(sen_stack))
                    thv_prev = thv
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    #x = np.squeeze(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    shuffle(train_images, train_sensors, train_steering, train_throttle)
    return train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

# Data Processor for past action & sensor input alongside 2d images; only the training/val set is shuffled. 
def load_2d_motorspeed_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')    
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        batt = row['battery']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        img = os.path.join(img_path, fname[:-1])
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = shaft_speed
        if mini_counter != 1:	
                    x.append(img)
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([shaft_speed]))
                    sen_stack = [np.float16(batt)]
                    z.append(np.stack(sen_stack))
                    thv_prev = thv
                    stv_prev = shaft_speed
        mini_counter += 1
    x = np.stack(x)
    #x = np.squeeze(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    shuffle(train_images, train_sensors, train_steering, train_throttle)
    return train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle



# Old Data_Processor. Note: This shuffles the test set!
def load_dataset(n_stacked,  img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, n_jump=1):
    attrs = ['steering', 'throttle']
    df = pd.read_csv(csv_path, encoding='utf-8')
    x = []
    y = []
    steering = []
    throttle = []
    img = []
    stacked_img = []
    img_stack = []
    if ((concatenate >= 2) and (prediction_mode == 'categorical')):
        mini_counter = 1
        for i, row in tqdm(df.iterrows()):
                fname = row['image']
                stv = row['steering']
                thv = row['throttle']
                img = cv2.imread(os.path.join(img_path, fname[:-1]))
                img = (img/127.5) - 1
                if (img.shape[0] > 160) and (img.shape[1] > 160):
                        img = img[250:, :]
                        img = cv2.resize(img, (160, 120))
                if counter == 1:
                        cv2.imshow(str(img.shape), img)
                        cv2.waitKey(1)
                        cv2.destroyAllWindows()
                img_stack.append(img.astype(np.float32))
                if mini_counter == concatenate:
                        x.append(np.stack(img_stack))
                        steering.append(np.stack([stv]))
                        throttle.append(np.stack([thv]))
                        img_stack = []
                        mini_counter = 1
                else: 
                        mini_counter += 1
        x = np.stack(x)
        steering = np.stack(steering)
        throttle = np.stack(throttle)
        train_x, train_steering, train_throttle = x, steering, throttle
        val_x, val_steering, val_throttle = test_x, test_steering, test_throttle = None, None, None 
        if test_size is not None:
                train_x, test_x, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
        if val_size is not None:
                train_x, val_x, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                train_x, train_steering, train_throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
        return train_x, val_x, test_x, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

    if ((concatenate >= 2) and (prediction_mode == 'linear')):
            print("Two Stack 3D CNN")
            for i, row in tqdm(df.iterrows()):
                fname = row['image']
                stv = row['steering'] 
                thv = row['throttle']
                img = cv2.imread(os.path.join(img_path, fname[:-1]))
                img = img[60:, :]
                img_stack.append(img.astype(np.float16))
                if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0:
                    x.append(np.stack(img_stack))
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([thv]))
            x = np.stack(x)
            steering = np.stack(steering)
            throttle = np.stack(throttle)
            train_x, train_steering, train_throttle = x, steering, throttle
            val_x, val_steering, val_throttle = test_x, test_steering, test_throttle = None, None, None 
            if test_size is not None:
                train_x, test_x, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
            if val_size is not None:
                train_x, val_x, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                train_x, train_steering, train_throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
            return train_x, val_x, test_x, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

    if ((concatenate == 1) and (prediction_mode == 'linear')):
            print("One Stack")
            counter = 1
            for i, row in tqdm(df.iterrows()):
                fname = row['image']
                stv = row['steering']
                thv = row['throttle'] 
                img = cv2.imread(os.path.join(img_path, fname[:-1]))
                if (img.shape[0] == 0) and (img.shape[1] == 0):
                        print(fname)
                img = img[35:, :]
                img = (img/127.5) - 1
                counter = counter + 1             
                img_stack.append(img.astype(np.float16))
                if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0:
                    x.append(np.stack(img_stack))
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([thv]))
            steering = np.stack(steering)
            throttle = np.stack(throttle)
            train_x, train_steering, train_throttle = x, steering, throttle
            val_x, val_steering, val_throttle = test_x, test_steering, test_throttle = None, None, None 
            if test_size is not None:
                train_x, test_x, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                x, steering, throttle, test_size=test_size,
                random_state=123, shuffle=True
                )
            if val_size is not None:
                train_x, val_x, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                train_x, train_steering, train_throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
            return train_x, val_x, test_x, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

# Note: Dataset has the trajectory shuffled as specified by the batch size. 
def load_bc_itdm_dataset(n_stacked, img_path, csv_path, w, h, d, concatenate, prediction_mode,
                 val_size=None, test_size=None, batch_size=None, n_jump=1):
    df = pd.read_csv(csv_path, encoding='utf-8')
    x = []
    z = []
    steering = []
    throttle = []
    img = []
    img_stack = []
    sen_stack = []
    mini_counter = 1
    for i, row in tqdm(df.iterrows()):
        fname = row['image']
        stv = row['steering']
        thv = row['throttle']
        shaft_speed = row['shaftspeed']
        fr_rpm = row['fr_rpm']
        fl_rpm = row['fl_rpm']
        rr_rpm = row['rr_rpm']
        rl_rpm = row['rl_rpm']
        batt = row['battery']
        roll = row['roll']
        pitch = row['pitch']
        yaw = row['yaw']
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        ang_vel_x = row['ang_vel_x']
        ang_vel_y = row['ang_vel_y']
        ang_vel_z = row['ang_vel_z']
        if (mini_counter) < 4:
                    img = cv2.imread(os.path.join(img_path, fname[:-1]))
                    #img = cv2.resize(img,(80,80))
                    #img = img[35:, :]
                    cv2.imshow(str(img.shape), img)
                    cv2.waitKey(1)
                    cv2.destroyAllWindows() 
        img = os.path.join(img_path, fname[:-1])
        img_stack.append(img)
        if mini_counter == 1:
                    stv_prev = stv
                    thv_prev = thv
        if i+1 >= n_stacked and (i+1 - n_stacked) % n_jump == 0 and mini_counter != 1:
                    x.append(img_stack)
                    img_stack = img_stack[n_jump:]
                    steering.append(np.stack([stv]))
                    throttle.append(np.stack([shaft_speed/((30.497-1.0534)/2)])) 
                    sen_stack = [np.float32(batt)]
                    z.append(np.stack(sen_stack))
                    thv_prev = thv
                    stv_prev = stv
        mini_counter += 1
    x = np.stack(x)
    z = np.stack(z)
    steering = np.stack(steering) 
    throttle = np.stack(throttle)
    val_images, val_sensors, val_steering, val_throttle = test_images, test_sensors, test_steering, test_throttle = empty_images, empty_sensors, empty_steering, empty_throttle = train_images, train_sensors, train_steering, train_throttle = None, None, None, None      
    if val_size is not None:
       train_images, val_images, train_sensors, val_sensors, train_steering, val_steering, train_throttle, val_throttle = train_test_split(
                x, z, steering, throttle, test_size=val_size,
                random_state=123, shuffle=True
                )
    x = []
    z = []
    steering = []
    throttle = []	
    if test_size is not None:
       train_images, test_images, train_sensors, test_sensors, train_steering, test_steering, train_throttle, test_throttle = train_test_split(
                train_images, train_sensors, train_steering, train_throttle, test_size=test_size,
                random_state=123, shuffle=False
                )
    '''
    # Create a shuffled set of trajectories of random lengths that fit the batch size. 
    limit = batch_size-1
    # Get the breakpoints for the trajectories.
    breakpoints = []
    for y in range(int(len(train_images)/batch_size)): 
        breakpoints.append(y*batch_size)
    # Collect and store trajectories based on the breakpoints.
    count = []
    for a in range(len(breakpoints)-1):
        counter = []
        for b in range(breakpoints[a], breakpoints[a+1]):
             counter.append(b) 
        count.append(counter)
    # Sort the trajectories into bins of batch size.
    # Shuffle the batches themselves.
    bin_count = [] 
    tr_bin_count = []
    for b in range(len(count)):
        bin_count.append(count[b])
        if ((count[b][-1] % batch_size) == limit):  
            tr_bin_count.append(bin_count) 
            bin_count = []
    random.shuffle(tr_bin_count)
    # Store the values of the trajectories
    tr_bin_sensors = []
    tr_bin_steering = []
    tr_bin_throttle = []
    tr_bin_images = [] 
    train_sensors = list(chain.from_iterable(train_sensors))
    train_steering = list(chain.from_iterable(train_steering))
    train_throttle = list(chain.from_iterable(train_throttle))
    checker = list(chain.from_iterable(tr_bin_count))
    tr_bin_count = list(chain.from_iterable(checker))
    for c in range(len(tr_bin_count)):
        value = int(tr_bin_count[c])
        tr_bin_sensors.append(np.stack([train_sensors[value]]))
        tr_bin_images.append(train_images[value])
        tr_bin_steering.append(np.stack([train_steering[value]]))
        tr_bin_throttle.append(np.stack([train_throttle[value]]))
    # Stack the values.
    train_images = []
    train_sensors = []
    train_steering = []
    train_throttle = []
    train_images = np.stack(tr_bin_images)
    train_sensors = np.stack(tr_bin_sensors)
    train_steering = np.stack(tr_bin_steering)
    train_throttle = np.stack(tr_bin_throttle)

    # Create a shuffled set of trajectories of random lengths that fit the batch size. 
    limit = batch_size-1
    # Get the breakpoints for the trajectories.
    breakpoints = []
    for y in range(int(len(val_images)/batch_size)): 
        breakpoints.append(y*batch_size)
    # Collect and store trajectories based on the breakpoints.
    count = []
    for a in range(len(breakpoints)-1):
        counter = []
        for b in range(breakpoints[a], breakpoints[a+1]):
             counter.append(b) 
        count.append(counter)
    # Sort the trajectories into bins of batch size.
    # Shuffle the batches themselves.
    bin_count = [] 
    tr_bin_count = []
    for b in range(len(count)):
        bin_count.append(count[b])
        if ((count[b][-1] % batch_size) == limit):  
            tr_bin_count.append(bin_count) 
            bin_count = []
    random.shuffle(tr_bin_count)
    # Store the values of the trajectories
    tr_bin_sensors = []
    tr_bin_steering = []
    tr_bin_throttle = []
    tr_bin_images = [] 
    val_sensors = list(chain.from_iterable(val_sensors))
    val_steering = list(chain.from_iterable(val_steering))
    val_throttle = list(chain.from_iterable(val_throttle))
    checker = list(chain.from_iterable(tr_bin_count))
    tr_bin_count = list(chain.from_iterable(checker))
    for c in range(len(tr_bin_count)):
        value = int(tr_bin_count[c])
        tr_bin_sensors.append(np.stack([val_sensors[value]]))
        tr_bin_images.append(train_images[value])
        tr_bin_steering.append(np.stack([val_steering[value]]))
        tr_bin_throttle.append(np.stack([val_throttle[value]]))
    # Stack the values.
    val_images = []
    val_sensors = []
    val_steering = []
    val_throttle = []
    val_images = np.stack(tr_bin_images)
    val_sensors = np.stack(tr_bin_sensors)
    val_steering = np.stack(tr_bin_steering)
    val_throttle = np.stack(tr_bin_throttle)
    '''
    return train_images, train_sensors, val_images, val_sensors, test_images, test_sensors, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle

