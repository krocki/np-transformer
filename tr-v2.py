# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
import pprint

HN = 2
M = 3
S = 5

datatype = np.float64

def softmax(ys):
  m0 = np.max(ys, axis=0)
  ps = np.exp(ys-m0)
  sums = np.sum(ps, axis=0)
  ps = ps / sums
  return ps

def forward(xs, model):
  hs = np.dot(model['Wxh'].T, xs)
  ys = np.dot(model['Why'].T, hs)
  ps = softmax(ys)
  states = {}
  states['xs'] = xs
  states['hs'] = hs
  states['ps'] = ps
  return states

def lossfun(states, ts):
  ce = -np.log(states['ps'][ts>0])
  return np.sum(ce)

def backward(states, model, ts):
  grads = {}
  dy = states['ps'] - ts
  grads['Why'] = np.dot(states['hs'], dy.T)
  grads['h'] = np.dot(model['Why'], dy)
  grads['Wxh'] = np.dot(states['xs'], grads['h'].T)

  return grads

def samplegrad(xs, model, ts):

  samples = 1
  delta = 1e-5

  grads = {}

  for layer, params in enumerate(model):
    p = model[params]
    grads[params] = {}
    p_shape=p.size
    #R = np.random.randint(0, p_shape, size=samples)
    for r in range(p_shape):
      p_old = p.flat[r]
      p.flat[r] = p_old + delta
      y0 = lossfun(forward(xs, model), ts)
      p.flat[r] = p_old - delta
      y1 = lossfun(forward(xs, model), ts)
      gn = (y0 - y1) / (2 * delta)
      p.flat[r] = p_old
      grads[params][r] = gn

  return grads

if __name__ == "__main__":

  encoder = {}
  #xs = np.zeros((S,M), dtype=datatype)
  xs = np.random.rand(M, S).astype(datatype)
  ts = np.zeros((M, S), dtype=datatype)

  for s in range(S):
    m = s % M
    ts[m,s] = 1

  # V Q K
  #WxVQK = np.random.randn(M, 3 * HN).astype(datatype) * 0.01

  Wxh = np.random.randn(M, HN).astype(datatype) * 0.01
  Why = np.random.randn(HN, M).astype(datatype) * 0.01
  encoder['Wxh']=Wxh
  encoder['Why']=Why

  states = forward(xs, encoder)
  loss = lossfun(states, ts)
  analytical = backward(states, encoder, ts)
  numerical = samplegrad(xs, encoder, ts)

  #pprint.pprint(numerical)
  nWhy = numerical['Why']
  nWhx = numerical['Wxh']
  dWhy = analytical['Why']
  dWhx = analytical['Wxh']

  err = {}
  num = {}

  err['Why'] = np.zeros_like(encoder['Why'])
  err['Wxh'] = np.zeros_like(encoder['Wxh'])
  num['Why'] = np.zeros_like(encoder['Why'])
  num['Wxh'] = np.zeros_like(encoder['Wxh'])

  for p in numerical['Why']:
    err['Why'].flat[p]=np.fabs(numerical['Why'][p] - analytical['Why'].flat[p])
    num['Why'].flat[p]=numerical['Why'][p]
    #print(p - analytical.flatten())

  print('num')
  print(num['Why'])
  print('err')
  print(err['Why'])
  print('max err = {}'.format(np.max(np.fabs(err['Why']))))

  for p in numerical['Wxh']:
    err['Wxh'].flat[p]=np.fabs(numerical['Wxh'][p] - analytical['Wxh'].flat[p])
    num['Wxh'].flat[p]=numerical['Wxh'][p]
    #print(p - analytical.flatten())

  print('num')
  print(num['Wxh'])
  print('err')
  print(err['Wxh'])
  print('max err = {}'.format(np.max(np.fabs(err['Wxh']))))
