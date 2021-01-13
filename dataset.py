import numpy as np
import matplotlib.pyplot as plt

import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.linear_model import LinearRegression

import sklearn
from sklearn import datasets

# The part for generating dataset

def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

# FRIEDMAN #1
def generate_F1():
  x, desire = sklearn.datasets.make_friedman1(20000)
  return x, desire

# FRIEDMAN #2
def generate_F2():
  x, desire = sklearn.datasets.make_friedman2(20000)
  return x, desire

# FRIEDMAN #3
def generate_F3():
  x, desire = sklearn.datasets.make_friedman3(20000)
  return x, desire

# CALIFORNIA HOUSING
def generate_CH():
  x = sklearn.datasets.fetch_california_housing()['data']
  desire = sklearn.datasets.fetch_california_housing()['target']
  return x, desire

# BOSTON HOUSING
def generate_BH():
  x = sklearn.datasets.load_boston()['data']
  desire = sklearn.datasets.load_boston()['target']
  return x, desire

# DIABETES
def generate_DB():
  x = sklearn.datasets.load_diabetes()['data']
  desire = sklearn.datasets.load_diabetes()['target']
  return x, desire

# FREQUENCY DOUBLER
def generate_FD(order = 3):
  i = np.linspace(0, 24*np.pi, 1200)
  x = np.sin(i)
  x = x/np.std(x)
  desire = np.sin(2*i)
  desire = desire/np.std(desire)
  x, desire = construct_time_windows(x, desire, order)
  return x, desire

# LORENZ SYSTEM
def generate_LORENZ(order = 10):
  dt = 0.01
  num_steps = 20000
  xs = np.empty(num_steps + 1)
  ys = np.empty(num_steps + 1)
  zs = np.empty(num_steps + 1)
  xs[0], ys[0], zs[0] = (0., 1., 1.05)
  for i in range(num_steps):
      x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
      xs[i + 1] = xs[i] + (x_dot * dt)
      ys[i + 1] = ys[i] + (y_dot * dt)
      zs[i + 1] = zs[i] + (z_dot * dt)

  x = ys
  desire = zs
  x, desire = construct_time_windows(x, desire, order)
  return x, desire

# LASER DATASET
def generate_LASER(order = 10):
  data = np.genfromtxt('./drive/MyDrive/ts_dataset/santafelaser.csv')
  x, desire = construct_time_windows(data[:-1], data[1:], order)
  return x, desire

# SUNSPOT 
def generate_SUN(order = 10):
  SN_list_ = []
  with open('./drive/MyDrive/SN_m_tot_V2.0.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
      SN_list = []
      list_ = row[0].split(' ')
      for number in list_:
        try: SN_list.append(float(number))
        except: continue
      SN_list_.append(np.array(SN_list, dtype='float32'))

  data = np.stack(SN_list_)[:, 3]
  x, desire = construct_time_windows(data[:-1], data[1:], order)
  return x, desire

# CO2
def generate_CO2(order = 10):
  co2_list_ = []
  with open('./drive/MyDrive/co2_mm_gl.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
      co2_list = []
      list_ = row[0].split(' ')
      for number in list_:
        try: co2_list.append(float(number))
        except: continue
      if len(co2_list) == 5:
        co2_list_.append(np.array(co2_list, dtype='float32'))
  data = np.stack(co2_list_)[:, 3]
  x, desire = construct_time_windows(data[:-1], data[1:], order)
  return x, desire

