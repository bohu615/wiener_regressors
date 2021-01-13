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

def construct_time_windows(x, desire, order):

  N = order # ORDER OF THE FILTERS

  x_list = []
  label_list = []
  DESIRE = []

  desire_windows = [] 

  for i in range(N-1, len(x)):
      x_list.append(x[i-N+1:i+1])
      label_list.append(desire[i])
          
  x_list = np.array(x_list)
  label_list = np.array(label_list)

  return x_list, label_list

class WF_assignment(nn.Module):
    def __init__(self, N, HIDDEN):
      super(WF_assignment, self).__init__()
      self.fc1 = nn.Linear(N, HIDDEN, bias=True)

    def forward(self, x):
      output = self.fc1(x)
      return output

class wf_layer_WIENERSOLUTION():
    def __init__(self, N):
        self.weights = None
        
    def forward(self, x):
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        
        return self.weights@x_.T
    
    def train(self, x, desire_list):
        
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        self.weights = desire_list.T@x_@np.linalg.pinv(x_.T@x_)
        return (x_.T@x_), desire_list.T@x_

def finding_the_constant(DESIRE, step, range):
  list_mean = []
  list_std = []

  batch = np.random.choice(DESIRE.shape[0], 3000000)
  y_train = DESIRE[batch]
  y_train = torch.from_numpy(y_train).float().cuda()

  batch = np.random.choice(DESIRE.shape[0], 3000000)
  y_train_2 = DESIRE[batch]
  y_train_2 = torch.from_numpy(y_train_2).float().cuda()

  for i in np.linspace(0.1, range, 2000):

    y_L1_distance = (y_train - y_train_2)/i

    distance_y_1_2 = torch.sum(y_L1_distance**2, 1)
    G_matrix_y_1_2 = compute_gram_matrix(distance_y_1_2, step)

    y_mean = torch.mean(G_matrix_y_1_2)
    y_std = torch.std(G_matrix_y_1_2)

    list_mean.append(y_mean.item())
    list_std.append(y_std.item())

  constant = np.linspace(0.1, range, 2000)[np.argmin(((np.array(list_mean)/np.array(list_std))-1)**2)]
  print('the constant is:', constant)
  return constant

def compute_gram_matrix(distance_matrix, sigma):
    G_matrix = torch.exp(-distance_matrix/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    return G_matrix

class WF(nn.Module):
    def __init__(self, N, HIDDEN, M):
        super(WF, self).__init__()
        self.fc1 = nn.Linear(N, HIDDEN, bias=True)

    def forward(self, x):
        x = (torch.sigmoid((self.fc1(x))))
        return x

class wf_layer_WIENERSOLUTION():
    def __init__(self, N):
        self.weights = None
        
    def forward(self, x):
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        
        return self.weights@x_.T
    
    def train(self, x, desire_list):
        
        x_ = np.concatenate((np.ones((x.shape[0], 1)), x), 1)
        self.weights = desire_list.T@x_@np.linalg.pinv(x_.T@x_)
        return (x_.T@x_), desire_list.T@x_

def compute_gram_matrix_np(distance_matrix, sigma):
    G_matrix = np.exp(-distance_matrix/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    return G_matrix

def normalize_tensor_np(tensor):
    mean = np.mean(tensor, 0)
    tensor = tensor-mean
    std = np.std(tensor, 0)
    tensor = tensor/std
    tensor = tensor
    return tensor

def compute_E_QMI_type1(x, z, step = 1, batch_size = 300):
    
    x = x.reshape(-1, 1)
    z = z.reshape(-1, 1)

    z = ((z-np.mean(z))/np.std(z))/np.sqrt(2)
    
    #np.random.seed(4)
    
    with temporary_seed(4):

      batch = np.random.choice(x.shape[0], batch_size)
      x_train = x[batch, :]
      y_train = z[batch, :]

      batch = np.random.choice(x.shape[0], batch_size)
      x_train_2 = x[batch, :]
      y_train_2 = z[batch, :]
      
    L1_distance = x_train - x_train_2
    L1_distance = normalize_tensor_np(L1_distance)
    y_L1_distance = y_train - y_train_2

    distance_x_1_2 = np.sum(L1_distance**2, 1)
    distance_y_1_2 = np.sum(y_L1_distance**2, 1)

    step = 1

    G_matrix_x_1_2 = compute_gram_matrix_np(distance_x_1_2, step) 
    G_matrix_y_1_2 = compute_gram_matrix_np(distance_y_1_2, step) 

    #G_matrix_x_1_2 = G_matrix_x_1_2 - np.mean(G_matrix_x_1_2) + np.std(G_matrix_x_1_2)
    #G_matrix_y_1_2 = G_matrix_y_1_2 - np.mean(G_matrix_y_1_2) + np.std(G_matrix_y_1_2)

    V_f = np.mean(G_matrix_x_1_2*G_matrix_y_1_2) 
    V_x = np.mean(G_matrix_x_1_2)
    V_y = np.mean(G_matrix_y_1_2)

    QMI = (-np.log2(V_x)) + (-np.log2(V_y))  - (-np.log2(V_f)) 
    
    return QMI

def compute_E_QMI_type2(x, z, step = 1, batch_size = 300):
    
    x = x.reshape(-1, 1)
    z = z.reshape(-1, 1)

    #np.random.seed(4)

    with temporary_seed(4):
    
      batch = np.random.choice(x.shape[0], batch_size)
      x_train = x[batch, :]
      y_train = z[batch, :]

      batch = np.random.choice(x.shape[0], batch_size)
      x_train_2 = x[batch, :]
      y_train_2 = z[batch, :]
    
    L1_distance = x_train - x_train_2
    y_L1_distance = y_train - y_train_2

    distance_x_1_2 = np.sum(L1_distance**2, 1)
    distance_y_1_2 = np.sum(y_L1_distance**2, 1)

    step = 1

    G_matrix_x_1_2 = compute_gram_matrix_np(distance_x_1_2, step) 
    G_matrix_y_1_2 = compute_gram_matrix_np(distance_y_1_2, step) 

    G_matrix_x_1_2 = G_matrix_x_1_2 - np.mean(G_matrix_x_1_2) + np.std(G_matrix_x_1_2)
    G_matrix_y_1_2 = G_matrix_y_1_2 - np.mean(G_matrix_y_1_2) + np.std(G_matrix_y_1_2)

    V_f = np.mean(G_matrix_x_1_2*G_matrix_y_1_2) 
    V_x = np.mean(G_matrix_x_1_2)
    V_y = np.mean(G_matrix_y_1_2)

    QMI = (-np.log2(V_x)) + (-np.log2(V_y)) - (-np.log2(V_f)) 
    
    return QMI

def QMI_curve(x, desire, D = 6, seed = 0, iter = 80000, bs = 3000, lr=0.1, prop=0.5, to_shuffle = False, save_iter = 1, step = 1, if_unormalized = False):

  np.random.seed(seed)
  shuffle = np.arange(0, x.shape[0])
  if to_shuffle:
    np.random.shuffle(shuffle)

  if if_unormalized:
    x, desire, x_list, label_list, x_test, test_label, desire_std = create_train_test_QMI(x, desire, shuffle, prop)
  else:
    x_, desire_, x_list, label_list, x_test, test_label, desire_std = create_train_test(x, desire, shuffle, prop)

  DESIRE = label_list.reshape(-1, 1)

  torch.manual_seed(seed)
  np.random.seed(seed)

  #constant = finding_the_constant(DESIRE, step, range = (0.01, 0.1))
  #DESIRE = DESIRE/constant
  #print('constant:', constant)

  net = WF_nn_MCA(x.shape[1], D).cuda()

  optimizer = optim.Adam([
              {'params': net.parameters(), 'lr': 1e-3},
          ])

  MI_list = []
  prediction_error_list = []
  G = []
  X = []

  prediction_error_list = []
  second_module_loss = []
  model_list = []
  type_1_list = []
  type_2_list= []

  correlation_train_list = []
  correlation_test_list = []

  norm_list = []

  beta_1 = 0.9
  beta_2 = 0.999
  beta_3 = 0.999

  m_t = 0
  v_t = 0
  c_t = 0

  for i in range(1, iter):

    batch = np.random.choice(x_list.shape[0], bs)
    x_train = torch.from_numpy(x_list[batch, :]).float().cuda()
    y_train = DESIRE[batch]
    y_train = torch.from_numpy(y_train).float().cuda()

    batch = np.random.choice(x_list.shape[0], bs)
    x_train_2 = torch.from_numpy(x_list[batch, :]).float().cuda()
    y_train_2 = DESIRE[batch]
    y_train_2 = torch.from_numpy(y_train_2).float().cuda()

    output_1 = net(x_train)
    output_2 = net(x_train_2)

    L1_distance = output_1 - output_2
    y_L1_distance = y_train - y_train_2
    
    distance_x_1_2 = torch.sum(L1_distance**2, 1)
    distance_y_1_2 = torch.sum(y_L1_distance**2, 1)

    sigma_x = step
    sigma_y = step

    G_matrix_x_1_2 = compute_gram_matrix(distance_x_1_2, sigma_x) 
    G_matrix_y_1_2 = compute_gram_matrix(distance_y_1_2, sigma_y) 

    m_t = beta_1*m_t + (1-beta_1)*torch.mean(G_matrix_x_1_2.detach())
    ex = m_t/(1-beta_1**i)

    ey = torch.mean(G_matrix_y_1_2.detach())

    c_t = beta_2*c_t + (1-beta_2)*torch.mean(G_matrix_x_1_2.detach()*G_matrix_y_1_2.detach())
    exy = c_t/(1-beta_2**i)

    v_t = beta_3*v_t + (1-beta_3)*torch.std(G_matrix_x_1_2.detach())**2
    stdx = torch.sqrt(v_t/(1-beta_3**i))

    x = G_matrix_x_1_2.detach()
    y = G_matrix_y_1_2.detach()

    LOSS = torch.mean((y-ey+ey*(x/stdx - ex/stdx))*G_matrix_x_1_2)/(exy-ex*ey+stdx*ey) - torch.mean((x/stdx - ex/stdx)*G_matrix_x_1_2)/(stdx)
    (-LOSS).backward(retain_graph=True)

    GD(net, lr)

    if i%save_iter == 0:

      output = net(torch.from_numpy(x_list).float().cuda()).cpu().data.numpy()
      layer_f = wf_layer_WIENERSOLUTION(output.shape[0])
      layer_f.train((output), label_list.reshape(-1))
      predict = layer_f.forward((output))

      error = np.mean((label_list.reshape(-1) - predict.reshape(-1))**2)
      second_module_loss.append(error)

      correlation_train_list.append(np.corrcoef(label_list.reshape(-1), predict.reshape(-1))[0, 1])

      output = net(torch.from_numpy(x_test).float().cuda()).cpu().data.numpy()
      predict = layer_f.forward((output))
      error = np.mean((test_label.reshape(-1) - predict.reshape(-1))**2)
      prediction_error_list.append(error)

      correlation_test_list.append(np.corrcoef(test_label.reshape(-1), predict.reshape(-1))[0, 1])

      model_output = torch.tanh(net(torch.from_numpy(x_test).float().cuda())).cpu().data.numpy()

  return np.array(second_module_loss)*(desire_std**2), np.array(prediction_error_list)*(desire_std**2), \
        norm_list, correlation_train_list, correlation_test_list, model_output, test_label, predict

def create_train_test(x, desire, shuffle, prop):
  train_size = int(shuffle.shape[0]*prop)

  x = (x - np.min(x, 0))/np.max((x - np.min(x, 0)), 0)
  desire = (desire - np.min(desire, 0))/np.max((desire - np.min(desire, 0)), 0)

  x = (x-0.5)*2
  desire = (desire-0.5)*2

  x_list = x[shuffle][0:train_size]
  x_list = x_list.reshape(x_list.shape[0], -1)
  label_list = np.array(desire)[shuffle].reshape(-1, 1)[0:train_size]

  desire_mean = np.mean(label_list)
  desire_std = np.std(label_list)
  label_list = (label_list-desire_mean)/desire_std
  DESIRE = label_list.reshape(-1, 1)

  x_test = x[shuffle][train_size:]
  x_test = x_test.reshape(x_test.shape[0], -1)
  test_label = np.array(desire)[shuffle].reshape(-1, 1)[train_size:]
  test_label = (test_label - desire_mean)/desire_std

  return x, desire, x_list, label_list, x_test, test_label, desire_std

def create_train_test_QMI(x, desire, shuffle, prop):
  train_size = int(prop*shuffle.shape[0])

  x = (x - np.min(x, 0))/np.max((x - np.min(x, 0)), 0)
  desire = (desire - np.min(desire, 0))/np.max((desire - np.min(desire, 0)), 0)

  x = (x-0.5)*2
  desire = (desire-0.5)*2

  x_list = x[shuffle][:]
  x_list = x_list.reshape(x_list.shape[0], -1)
  label_list = np.array(desire)[shuffle].reshape(-1, 1)[:]

  desire_std = 1

  x_test = x[shuffle][train_size:]
  x_test = x_test.reshape(x_test.shape[0], -1)
  test_label = np.array(desire)[shuffle].reshape(-1, 1)[train_size:]

  return x, desire, x_list, label_list, x_test, test_label, desire_std

class WF(nn.Module):
    def __init__(self, N, HIDDEN):
        super(WF, self).__init__()
        self.fc1 = nn.Linear(N, HIDDEN, bias=True)

    def forward(self, x):
        x = (torch.sigmoid((self.fc1(x))))
        return x

class WF_nn(nn.Module):
    def __init__(self, N, HIDDEN, M):
        super(WF_nn, self).__init__()
        self.fc1 = nn.Linear(N, HIDDEN, bias=True)
        self.fc2 = nn.Linear(HIDDEN, M, bias=True)

    def forward(self, x):
        x = (self.fc2(torch.tanh((self.fc1(x)))))
        return x

def NN_curve(x, desire, D = 6, seed = 0, iter = 40000, bs = 300, lr=1e-3, prop=0.5, to_shuffle = False, save_iter = 1):
  
  np.random.seed(seed)
  shuffle = np.arange(0, x.shape[0])

  if to_shuffle:
    np.random.shuffle(shuffle)

  x_, desire_, x_list, label_list, x_test, test_label, desire_std = create_train_test(x, desire, shuffle, prop)

  DESIRE = label_list.reshape(-1, 1)

  torch.manual_seed(seed)
  np.random.seed(seed)

  net = WF_nn(x.shape[1], D, 1).cuda()

  optimizer = optim.Adam([
              {'params': net.parameters(), 'lr': lr},
          ])

  MI_list = []
  prediction_error_list = []
  G = []
  X = []

  norm_list = []

  prediction_error_list = []
  second_module_loss = []

  correlation_train_list = []
  correlation_test_list = []

  for i in range(1, iter):
            
      batch = np.random.choice(x_list.shape[0], bs)
      x_train = torch.from_numpy(x_list[batch, :]).float().cuda()
      y_train = DESIRE[batch]
      y_train = torch.from_numpy(y_train).float().cuda()

      output_1 = (net(x_train))

      LOSS = torch.mean((output_1.view(-1) - y_train.view(-1))**2)
      (LOSS).backward()
      MI_list.append(LOSS.item())
          
      optimizer.step()
      optimizer.zero_grad()
      
      if i%save_iter == 0:
        output = net(torch.from_numpy(x_list).float().cuda()).cpu().data.numpy()
        predict = output

        error = np.mean((label_list.reshape(-1) - predict.reshape(-1))**2)
        second_module_loss.append(error)

        correlation_train_list.append(np.corrcoef(label_list.reshape(-1), predict.reshape(-1))[0, 1])

        output = net(torch.from_numpy(x_test).float().cuda()).cpu().data.numpy()
        predict = output
        error = np.mean((test_label.reshape(-1) - predict.reshape(-1))**2)
        prediction_error_list.append(error)

        correlation_test_list.append(np.corrcoef(test_label.reshape(-1), predict.reshape(-1))[0, 1])

        model_output = torch.tanh(net.fc1(torch.from_numpy(x_test).float().cuda())).cpu().data.numpy()

  return np.array(second_module_loss)*(desire_std**2), np.array(prediction_error_list)*(desire_std**2), \
        norm_list, correlation_train_list, correlation_test_list, model_output, test_label, predict 

def GD(net, lr):
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr*param.grad
            
    net.zero_grad()
    return 0

def MCA_curve(x, desire, D = 6, seed = 0, iter = 80000, bs = 300, lr = 0.1, prop=0.5, to_shuffle = False, save_iter = 1):

  np.random.seed(seed)
  shuffle = np.arange(0, x.shape[0])

  if to_shuffle:
    np.random.shuffle(shuffle)

  x_, desire_, x_list, label_list, x_test, test_label, desire_std = create_train_test(x, desire, shuffle, prop)

  DESIRE = label_list.reshape(-1, 1)

  torch.manual_seed(seed)
  np.random.seed(seed)

  net = WF_assignment(x.shape[1], D).cuda()
  output = net(torch.from_numpy(x_list).float().cuda())

  optimizer = optim.Adam([
              {'params': net.parameters(), 'lr': 1e-3},
          ])

  MI_list = []
  prediction_error_list = []
  G = []
  X = []

  prediction_error_list = []
  second_module_loss = []

  mean_cross_ = []
  mean_entropy_ = []
  correlation_ = []
  error_1 = []
  error_2 = []

  norm_list = []

  correlation_train_list = []
  correlation_test_list = []

  beta_1 = 0.9
  beta_2 = 0.999
  beta_3 = 0.999

  m_t = 0
  v_t = 0
  c_t = 0

  for i in range(1, iter):
    batch = np.random.choice(x_list.shape[0], bs)
    x_train = torch.from_numpy(x_list[batch, :]).float().cuda()
    y_train = DESIRE[batch]
    y_train = torch.from_numpy(y_train).float().cuda()

    output = torch.tanh(net(x_train))
    m_t = beta_1*m_t + (1-beta_1)*torch.mean(torch.sum(output.detach(), 1))
    mean = m_t/(1-beta_1**i)
    output = output - mean/(D)

    sum_value = torch.sum(output.detach(), 1).view(-1, 1)
    v_t = beta_2*v_t + (1-beta_2)*torch.std(sum_value)**2
    variance = v_t/(1-beta_2**i)
    std = torch.sqrt(variance)

    c_t = beta_3*c_t + (1-beta_3)*torch.mean(sum_value*y_train)
    correlation = c_t/(1-beta_3**i)

    gradient_1 = torch.mean(output*y_train, 0)/std
    (gradient_1).backward(-torch.ones(gradient_1.shape).cuda(), retain_graph=True)

    gradient_2 = correlation*torch.mean(sum_value*output, 0)/(std**3)
    (gradient_2).backward(torch.ones(gradient_2.shape).cuda())

    GD(net, lr)

    if i%save_iter == 0:

      output = torch.tanh(net(torch.from_numpy(x_list).float().cuda())).cpu().data.numpy()
      layer_f = wf_layer_WIENERSOLUTION(output.shape[0])
      layer_f.train((output), label_list.reshape(-1))
      predict = layer_f.forward((output))

      error = np.mean((label_list.reshape(-1) - predict.reshape(-1))**2)
      second_module_loss.append(error)

      correlation_train_list.append(np.corrcoef(label_list.reshape(-1), predict.reshape(-1))[0, 1])

      output = torch.tanh(net(torch.from_numpy(x_test).float().cuda())).cpu().data.numpy()
      predict = layer_f.forward((output))
      error = np.mean((test_label.reshape(-1) - predict.reshape(-1))**2)
      prediction_error_list.append(error)

      correlation_test_list.append(np.corrcoef(test_label.reshape(-1), predict.reshape(-1))[0, 1])

      model_output = torch.tanh(net(torch.from_numpy(x_test).float().cuda())).cpu().data.numpy()

  return np.array(second_module_loss)*(desire_std**2), np.array(prediction_error_list)*(desire_std**2), \
        norm_list, correlation_train_list, correlation_test_list, model_output, test_label, predict

def run_dataset(x, desire, to_shuffle = False, D=6, seed=0, prop=0.7, bs=64, bs_QMI=64, dataset='F1', iter = 10000, lr_NN=1e-3, lr_MCA=0.1, lr_QMI=0.1, QMI_step = 1, QMI_if_unormalized = False):
  NN_output_F1 = NN_curve(x, desire, D = D, seed = seed, iter = iter, bs=bs, lr=lr_NN, prop=prop, to_shuffle = False, save_iter = 1)
  MCA_output_F1 = MCA_curve(x, desire, D = D, seed = seed, iter = iter, bs=bs_QMI, lr=lr_MCA, prop=prop, to_shuffle = False, save_iter = 1)
  QMI_output_F1 = QMI_curve(x, desire, D = D, seed = seed, iter = iter, bs=bs_QMI, lr=lr_QMI, prop=prop, to_shuffle = False, save_iter = 1, step = QMI_step, if_unormalized = QMI_if_unormalized)

  np.save('./outputs/NN_output_seed{0}_dataset{1}.npy'.format(seed, dataset), NN_output_F1)
  np.save('./outputs/MCA_output_seed{0}_dataset{1}.npy'.format(seed, dataset), MCA_output_F1)
  np.save('./outputs/QMI_output_seed{0}_dataset{1}.npy'.format(seed, dataset), QMI_output_F1)

  return NN_output_F1, MCA_output_F1, QMI_output_F1
