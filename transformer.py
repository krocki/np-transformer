# -*- coding: utf-8 -*-
# author: krocki
import numpy as np

from gradcheck import *
from utils import *
import argparse

HN = 10 # the code layer
M = 10 # input / output size
lr = 1e-3 # learning rate
max_iters=1000000000

show_interval = 10000 # stats
checkpoint = 1000000 # save model

def pos_table(N, dim):
  def get_angle(x, h):
    return x / np.power(10000, 2 * (h // 2) / dim)
  def get_angle_vec(x):
    return [get_angle(x, j) for j in range(dim)]
  tab = np.array([get_angle_vec(i) for i in range(N)]).astype(float)
  tab[:, 0::2] = np.sin(tab[:, 0::2])
  tab[:, 1::2] = np.cos(tab[:, 1::2])
  return tab


def relu(x):
  y = x
  y[y<0] = 0
  return y

def sigmoid(hs):
  return 1.0/(1.0 + np.exp(-hs))

def ident(x):
  return x

def dident(dy):
  return dy

def softmax(ys):
  m0 = np.max(ys, axis=0)
  ps = np.exp(ys-m0)
  sums = np.sum(ps, axis=0)
  ps = ps / sums
  return ps

def dsigmoid(h, dh):
  return dh * h * (1.0 - h)

def drelu(y, dy):
  return y * dy

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
  #####

  states['att'] = np.dot(qs, ks.T)
  states['att_sm'] = softmax(states['att'])
  #states['att_sm'] = ident(states['att'])
  #states['att_sm'] = ident(states['att'])
  zs0 = np.dot(states['att_sm'], vs)
  zs0 = sigmoid(zs0);
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
  #dy = dsigmoid(states['ps'], dy)

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

#def apply_grads(model, grads, lr):
#
#  #rho = 0.9
#  for t in model:
#    model[t] -= grads[t] * lr
#  #  mem[t] = rho * mem[t] + (1-rho) * grads[t] * grads[t]
#  #  model[t] -= grads[t] * lr / np.sqrt(mem[t] + 1e-4)
def apply_grads(model, grads, lr):

  for t in model:
    model[t] -= grads[t] * lr

  return model

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-S', type=int, default = 5, help='seq len')
  parser.add_argument('-t', type=str, default = 'copy', help='task')
  parser.add_argument('-g', action='store_true', help='gradcheck')

  opt = parser.parse_args()

  save_img = True
  do_gradcheck = opt.g
  S = opt.S # seq len

  print(opt)

  datatype = np.float64 if do_gradcheck else np.float32

  model = {}
  Wxv = np.random.randn(M*2, HN).astype(datatype) * 0.01
  Wxk = np.random.randn(M*2, HN).astype(datatype) * 0.01
  Wxq = np.random.randn(M*2, HN).astype(datatype) * 0.01
  Wzy = np.random.randn(HN, M).astype(datatype) * 0.01

  model['Wxv']=Wxv
  model['Wxk']=Wxk
  model['Wxq']=Wxq
  model['Wzy']=Wzy

  i=0;
  smooth_loss = None

  ts = np.zeros((M,S), dtype=int)
  xs = np.zeros((M,S), dtype=int)
  p = 0

  I = np.eye(M).astype(datatype)
  pos = pos_table(S, M)

  while i<max_iters:

    ix = np.random.randint(low=1, high=(M-1), size=(S))
    ts = np.zeros((M,S), dtype=datatype)
    xs = np.zeros((M*2,S), dtype=datatype)

    for j in range(S):
      xs[:M, j] = I[ix[j]]

    xs_raw = xs
    xs[M:, :] = pos.T

    if opt.t in ['copy', 'rotate', 'reverse', 'filter']:
      if opt.t == 'copy':
        for j in range(S): ts[:,j] = I[ix[j]]
      if opt.t == 'rotate':
        for j in range(S): ts[:,j] = I[ix[(j+1)%S]]
      if opt.t == 'reverse':
        for j in range(S): ts[:,j] = I[ix[S-j-1]]
      if opt.t == 'filter':
        for j in range(S): ts[:,j] = I[ix[j]] if ix[j] > 4 and ix[j] < 6 else I[ix[0]]

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
      save_dict('checkpoint_{}_{}.npy'.format(i, opt.t), model)

      if do_gradcheck:
        err=checkgrads(xs, model, forward, lossfun, ts, grads)
        print('\ngradcheck err {:2.9f}'.format(err))
        if err>1e-7:
          print('!!!')
          #input('')
        else: print('OK')
      else: print('')

    if (i%show_interval==0):
      print('iter {:6}, loss = {:5.5f}'.format(i, smooth_loss))

    model = apply_grads(model, grads, lr)
    i=i+1
