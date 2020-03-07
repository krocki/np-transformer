# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
from collections import OrderedDict
from gradcheck import *
from utils import *
import time
import argparse

multi_layer = False

# positional encoding
def pos_table(N, dim):
  def get_angle(x, h):
    return x / np.power(10000, 2 * (h // 2) / dim)
  def get_angle_vec(x):
    return [get_angle(x, j) for j in range(dim)]
  tab = np.array([get_angle_vec(i) for i in range(N)]).astype(float)
  tab[:, 0::2] = np.sin(tab[:, 0::2])
  tab[:, 1::2] = np.cos(tab[:, 1::2])
  return tab

def sigmoid(hs):
  return 1.0/(1.0 + np.exp(-hs))

# for debug
def ident(x): return x
def dident(d, dy): return dy

def relu(x):
  return np.maximum(x, 0)

def drelu(y, dy):
  dy[y<=0] = 0
  return dy

def softmax(ys):
  m0 = np.max(ys, axis=0)
  ps = np.exp(ys-m0)
  sums = np.sum(ps, axis=0)
  ps = ps / sums
  return ps

def dsigmoid(h, dh):
  return dh * h * (1.0 - h)

def dsoftmax(dy, y):
  dx = dy * y
  sm_sum = np.sum(dx, axis=0)
  dx -= y * sm_sum
  return dx

def forward(xs, model):

  states = {};

  # 0
  states['xs'] = xs
  vs0 = np.dot(model['Wxv0'].T, xs)
  ks0 = np.dot(model['Wxk0'].T, xs)
  qs0 = np.dot(model['Wxq0'].T, xs)

  states['vs0'] = vs0
  states['qs0'] = qs0
  states['ks0'] = ks0

  att0 = np.dot(qs0, ks0.T)
  states['att0_sm'] = softmax(att0)
  zs0_pre = np.dot(states['att0_sm'], vs0)
  zs0 = sigmoid(zs0_pre); #residual #zs = xs + zs0
  #zs0 = zs0_pre

  states['zs0_pre'] = zs0_pre
  states['zs0'] = zs0

  if multi_layer:
    # 1
    vs1 = np.dot(model['Wxv1'].T, zs0)
    ks1 = np.dot(model['Wxk1'].T, zs0)
    qs1 = np.dot(model['Wxq1'].T, zs0)

    states['vs1'] = vs1
    states['qs1'] = qs1
    states['ks1'] = ks1

    att1 = np.dot(qs1, ks1.T)
    states['att1_sm'] = softmax(att1)
    zs1_pre = np.dot(states['att1_sm'], vs1)
    zs1_local = sigmoid(zs1_pre);
    if residual:
      zs1 = zs0 + zs1_local
    else:
      zs1 = zs1_local

    states['zs1_pre'] = zs1_pre
    states['zs1'] = zs1
    #
    ys = np.dot(model['Wzy'].T, zs1)
  else:
    ys = np.dot(model['Wzy'].T, zs0)

  ps = sigmoid(ys);
  states['ps'] = ps
  states['ys'] = ys

  return states

def lossfun(states, ts):
  #ce = -np.log(states['ps'][ts>0])
  #return np.sum(ce)
  #mse
  mse = (ts - states['ps']) ** 2
  return np.sum(mse)

def backward(states, model, ts):

  grads = dict_zeros_like(model)
  #MSE
  dy = 2.0 * (states['ps'] - ts)
  dy = dsigmoid(states['ps'], dy)
  #CE
  #dy = states['ps'] - ts
  states['dy'] = dy

  if multi_layer:
    grads['Wzy'] = np.dot(states['zs1'], dy.T)
    grads['zs1'] = np.dot(model['Wzy'], dy)

    grads['zs1'] = dsigmoid(states['zs1'], grads['zs1'])

    datt1_sm = np.dot(grads['zs1'], states['vs1'].T)

    # backprop through softmax
    datt1 = dsoftmax(datt1_sm, states['att1_sm'])
    grads['datt1_sm'] = datt1_sm
    #grads['datt1'] = datt1

    grads['vs1'] = np.dot(states['att1_sm'].T, grads['zs1'])
    grads['qs1'] = np.dot(datt1, states['ks1'])
    grads['ks1'] = np.dot(datt1.T, states['qs1'])

    grads['Wxv1'] = np.dot(states['zs0'], grads['vs1'].T)
    grads['Wxq1'] = np.dot(states['zs0'], grads['qs1'].T)
    grads['Wxk1'] = np.dot(states['zs0'], grads['ks1'].T)

    grads['zs0']  = np.dot(model['Wxv1'], grads['vs1'])
    grads['zs0'] += np.dot(model['Wxq1'], grads['qs1'])
    grads['zs0'] += np.dot(model['Wxk1'], grads['ks1'])
  else:
    grads['Wzy'] = np.dot(states['zs0'], dy.T)
    grads['zs0'] = np.dot(model['Wzy'], dy)

  grads['zs0'] = dsigmoid(states['zs0'], grads['zs0'])
  datt0_sm = np.dot(grads['zs0'], states['vs0'].T)

  # backprop through softmax
  datt0 = dsoftmax(datt0_sm, states['att0_sm'])
  grads['datt0_sm'] = datt0_sm
  #grads['datt0'] = datt0

  grads['vs0'] = np.dot(states['att0_sm'].T, grads['zs0'])
  grads['qs0'] = np.dot(datt0, states['ks0'])
  grads['ks0'] = np.dot(datt0.T, states['qs0'])

  grads['Wxv0'] = np.dot(states['xs'], grads['vs0'].T)
  grads['Wxq0'] = np.dot(states['xs'], grads['qs0'].T)
  grads['Wxk0'] = np.dot(states['xs'], grads['ks0'].T)

  #grads['xs']  = np.dot(model['Wxv0'], grads['vs0'])
  #grads['xs'] += np.dot(model['Wxq0'], grads['qs0'])
  #grads['xs'] += np.dot(model['Wxk0'], grads['ks0'])

  return grads

# adadelta ?
def apply_grads(model, mem, grads, lr):

  rho = 0.9
  for t in model:
    mem[t] = rho * mem[t] + (1-rho) * grads[t] * grads[t]
    model[t] -= grads[t] * lr / np.sqrt(mem[t] + 1e-5)
  return model

# SGD
# def apply_grads(model, grads, lr):
# 
#   for t in model:
#     model[t] -= grads[t] * lr
# 
#   return model

def dict_zeros_like(x):
  y = {}
  for x_id,arr in enumerate(x):
    y[arr] = np.zeros_like(x[arr])
  return y

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-S', type=int, default = 5, help='seq len')
  parser.add_argument('-M', type=int, default = 8, help='input width')
  parser.add_argument('-N', type=int, default = 8, help='inner width')
  parser.add_argument('-r', type=float, default = 1e-3, help='learning rate')
  parser.add_argument('-v', type=int, default = 10000, help='show interval')
  parser.add_argument('-c', type=int, default = 100000, help='checkpoint interval')
  parser.add_argument('-t', type=str, default = 'copy', help='task')
  parser.add_argument('-l', type=str, default = None, help='load from')
  parser.add_argument('-g', action='store_true', help='gradcheck')

  opt = parser.parse_args()

  save_img = True
  do_gradcheck = opt.g
  S = opt.S # seq len
  HN = opt.N # the code layer
  M = opt.M # input / output size
  lr = opt.r # learning rate
  show_interval = opt.v # stats
  checkpoint = opt.c # save model
  max_iters=1000000000

  print(opt)

  datatype = np.float64 if do_gradcheck else np.float32

  model = OrderedDict()
  if opt.l:
    model = load_dict(opt.l)
  else:

    model['Wxv0'] = np.random.randn(M*2, HN).astype(datatype) * 0.02
    model['Wxk0'] = np.random.randn(M*2, HN).astype(datatype) * 0.02
    model['Wxq0'] = np.random.randn(M*2, HN).astype(datatype) * 0.02

    if multi_layer:
      model['Wxv1'] = np.random.randn(M, HN).astype(datatype) * 0.02
      model['Wxk1'] = np.random.randn(M, HN).astype(datatype) * 0.02
      model['Wxq1'] = np.random.randn(M, HN).astype(datatype) * 0.02

    model['Wzy'] = np.random.randn(HN, M).astype(datatype) * 0.02

  mem = dict_zeros_like(model)

  i=0;
  smooth_loss = None

  ts = np.zeros((M,S), dtype=int)
  xs = np.zeros((M,S), dtype=int)
  p = 0

  I = np.eye(M).astype(datatype) # 1-hot encoding
  pos = pos_table(S, M) # positional encoding

  fwd_time, bwd_time, app_time = 0,0,0

  while i<max_iters:

    ix = np.random.randint(low=1, high=(M-1), size=(S))
    ts = np.zeros((M,S), dtype=datatype)
    xs = np.zeros((M*2,S), dtype=datatype) # input = [ val, position ]

    #xs[:M, :] = I[ix, :].T
    xs[:M, :] = np.random.rand(M, S).astype(datatype) > 0.75
    xs_raw = xs[:M, :] # only value
    xs[M:, :] = pos.T
    zero = np.zeros((M,1)).astype(datatype)
    tasks = ['copy', 'rotate', 'reverse', 'filter', 'filter2', 'filter3', 'filter4', 'sub']
    if opt.t in tasks:
      if opt.t == 'copy':
        for j in range(S): ts[:,j] = xs_raw[:,j]
      if opt.t == 'rotate':
        for j in range(S): ts[:,j] = xs_raw[:,(j+1)%S]
      if opt.t == 'reverse':
        for j in range(S): ts[:,j] = xs_raw[:,S-j-1]
      if opt.t == 'filter': # only copy some bits
        for j in range(S): ts[:,j] = xs_raw[:,j]; ts[:3,j]=0; ts[6:,j]=0
      if opt.t == 'filter2': # only copy some if one of the first bits set
        for j in range(S): ts[:,j] = xs_raw[:,j] if xs_raw[1,j]==1 or xs_raw[0,j]==1 else 0
      if opt.t == 'filter3': # only copy some if the first bits set and range
        for j in range(S):
          ts[:,j] = xs_raw[:,j] if xs_raw[0,j]==1 else 0
          ts[:2,j]=0; ts[M-4:j]=0
      if opt.t == 'filter4': # only copy some if the sum of the bits set and range
        for j in range(S):
          ts[:,j] = xs_raw[:,j] if np.sum(xs_raw[:,j])>3 else 0
          ts[:2,j]=0; ts[M-4:j]=0
      if opt.t == 'sub': # max of odd and even
        for j in range(S//2):
          ts[:,2*j] = xs_raw[:,2*j] - xs_raw[:,2*j+1]
          ts[:,2*j+1] = xs_raw[:, 2*j+1]
          ts[ts<0] = 0
    else:
      print('unknown task {}'.format(opt.t))
      print('defined tasks {}'.format(tasks))

    t0 = time.perf_counter()
    states = forward(xs, model)
    t1 = time.perf_counter()
    fwd_time += t1 - t0
    states['xs'] = xs
    states['pos'] = pos.T

    states['xs_raw'] = xs_raw
    states['ts'] = ts

    loss = lossfun(states, ts)
    smooth_loss = loss * 0.99999 + smooth_loss * 0.00001 if smooth_loss else loss

    t0 = time.perf_counter()
    grads = backward(states, model, ts)
    t1 = time.perf_counter()
    bwd_time += t1 - t0

    if (checkpoint > 0 and (i%checkpoint)==0):
      if do_gradcheck:
        err=checkgrads(xs, model, forward, lossfun, ts, grads)
        print('\ngradcheck err {:2.9f}'.format(err))
        if err>1e-7:
          print('!!!')
        else: print('OK')
      else: print('')

      if save_img: save_dict_img('checkpoint_{}_{}.pdf'.format(i, opt.t), [model, states, grads])
      save_dict('checkpoint_{}_{}.pkl'.format(opt.t, i), model)

    if (i>0 and i%show_interval==0):
      print('iter {:6}, loss = {:5.5f}, {:4.9f} {:4.9f} {:4.9f}'.format(i, smooth_loss, fwd_time/i, bwd_time/i, app_time/i))

    t0 = time.perf_counter()
    model = apply_grads(model, mem, grads, lr)
    t1 = time.perf_counter()
    app_time += t1 - t0
    i=i+1
