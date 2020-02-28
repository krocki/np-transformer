# -*- coding: utf-8 -*-
# author: krocki
import numpy as np

from gradcheck import *
from utils import *
import argparse

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
def dident(dy): return dy

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

  states = {}; states['xs'] = xs
  vs = np.dot(model['Wxv'].T, xs)
  ks = np.dot(model['Wxk'].T, xs)
  qs = np.dot(model['Wxq'].T, xs)

  states['vs'] = vs
  states['qs'] = qs
  states['ks'] = ks

  states['att'] = np.dot(qs, ks.T)
  states['att_sm'] = softmax(states['att'])
  zs0 = np.dot(states['att_sm'], vs)
  zs0 = sigmoid(zs0);
  #residual
  #zs = xs + zs0
  zs = zs0

  states['zs0'] = zs0
  states['zs'] = zs

  ys = np.dot(model['Wzy'].T, zs)
  ps = softmax(ys);
  states['ps'] = ps
  states['ys'] = ys

  return states

def lossfun(states, ts):
  ce = -np.log(states['ps'][ts>0])
  return np.sum(ce)

def backward(states, model, ts):

  grads = {}
  dy = states['ps'] - ts
  states['dy'] = dy

  grads['Wzy'] = np.dot(states['zs'], dy.T)
  grads['zs'] = np.dot(model['Wzy'], dy)

  grads['zs'] = dsigmoid(states['zs0'], grads['zs'])

  datt_sm = np.dot(grads['zs'], states['vs'].T)

  # backprop through softmax
  datt = dsoftmax(datt_sm, states['att_sm'])
  grads['datt_sm'] = datt_sm
  grads['datt'] = datt

  grads['vs'] = np.dot(states['att_sm'].T, grads['zs'])
  grads['qs'] = np.dot(datt, states['ks'])
  grads['ks'] = np.dot(datt.T, states['qs'])

  grads['Wxv'] = np.dot(states['xs'], grads['vs'].T)
  grads['Wxq'] = np.dot(states['xs'], grads['qs'].T)
  grads['Wxk'] = np.dot(states['xs'], grads['ks'].T)

  return grads

# adadelta ?
def apply_grads(model, mem, grads, lr):

  rho = 0.9
  for t in model:
    mem[t] = rho * mem[t] + (1-rho) * grads[t] * grads[t]
    model[t] -= grads[t] * lr / np.sqrt(mem[t] + 1e-4)
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
  parser.add_argument('-t', type=str, default = 'copy', help='task')
  parser.add_argument('-l', type=str, default = None, help='load from')
  parser.add_argument('-g', action='store_true', help='gradcheck')

  opt = parser.parse_args()

  save_img = True
  do_gradcheck = opt.g
  S = opt.S # seq len
  HN = 16 # the code layer
  M = 8 # input / output size
  lr = 1e-3 # learning rate
  max_iters=1000000000
  show_interval = 10000 # stats
  checkpoint = 100000 # save model

  print(opt)

  datatype = np.float64 if do_gradcheck else np.float32

  model = {}
  if opt.l:
    model = load_dict(opt.l)
  else:
    model['Wxv'] = np.random.randn(M*2, HN).astype(datatype) * 0.01
    model['Wxk'] = np.random.randn(M*2, HN).astype(datatype) * 0.01
    model['Wxq'] = np.random.randn(M*2, HN).astype(datatype) * 0.01
    model['Wzy'] = np.random.randn(HN, M).astype(datatype) * 0.01

  mem = dict_zeros_like(model)

  i=0;
  smooth_loss = None

  ts = np.zeros((M,S), dtype=int)
  xs = np.zeros((M,S), dtype=int)
  p = 0

  I = np.eye(M).astype(datatype) # 1-hot encoding
  pos = pos_table(S, M) # positional encoding

  # I[0] is an 'invalid' token
  while i<max_iters:

    ix = np.random.randint(low=1, high=(M-1), size=(S))
    ts = np.zeros((M,S), dtype=datatype)
    xs = np.zeros((M*2,S), dtype=datatype) # input = [ val, position ]

    xs[:M, :] = I[ix, :].T
    xs_raw = xs[:M, :] # only value
    xs[M:, :] = pos.T

    if opt.t in ['copy', 'rotate', 'reverse', 'filter']:
      if opt.t == 'copy':
        for j in range(S): ts[:,j] = I[ix[j]]
      if opt.t == 'rotate':
        for j in range(S): ts[:,j] = I[ix[(j+1)%S]]
      if opt.t == 'reverse':
        for j in range(S): ts[:,j] = I[ix[S-j-1]]
      if opt.t == 'filter': # only copy values < M/2
        for j in range(S): ts[:,j] = I[ix[j]] if (ix[j]<M//2) else I[0]

    else:
      print('unknown task {}'.format(opt.t))

    states = forward(xs, model)
    states['xs'] = xs
    states['pos'] = pos.T

    states['xs_raw'] = xs_raw
    states['ts'] = ts

    loss = lossfun(states, ts)
    smooth_loss = loss * 0.999 + smooth_loss * 0.001 if smooth_loss else loss

    grads = backward(states, model, ts)

    if (checkpoint > 0 and (i%checkpoint)==0):
      if save_img: save_dict_img('checkpoint_{}_{}.pdf'.format(i, opt.t), [model, grads, states])
      save_dict('checkpoint_{}_{}.pkl'.format(opt.t, i), model)

      if do_gradcheck:
        err=checkgrads(xs, model, forward, lossfun, ts, grads)
        print('\ngradcheck err {:2.9f}'.format(err))
        if err>1e-7:
          print('!!!')
        else: print('OK')
      else: print('')

    if (i%show_interval==0):
      print('iter {:6}, loss = {:5.5f}'.format(i, smooth_loss))

    model = apply_grads(model, mem, grads, lr)
    i=i+1
