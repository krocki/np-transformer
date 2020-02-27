# -*- coding: utf-8 -*-
# author: krocki
import numpy as np

HN = 8
M = 10
S = 5

datatype = np.float64

def softmax(ys):
  m0 = np.max(ys, axis=0)
  ps = np.exp(ys-m0)
  sums = np.sum(ps, axis=0)
  ps = ps / sums
  return ps

def forward(xs, model):
  zs = np.dot(model['WxVQK'].T, xs)
  ys = np.dot(model['Wzy'].T, zs)
  ps = softmax(ys)
  states = {}
  states['xs'] = xs
  states['zs'] = zs
  states['ps'] = ps
  return states

def lossfun(states, ts):
  ce = -np.log(states['ps'][ts>0])
  return np.sum(ce)

def backward(states, model, ts):
  grads = {}
  dy = states['ps'] - ts
  grads['Wzy'] = np.dot(states['zs'], dy.T)
  grads['z'] = np.dot(model['Wzy'], dy)
  grads['WxVQK'] = np.dot(states['xs'], grads['z'].T)

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

def checkgrads(xs, encoder, ts, analytical):
  numerical = samplegrad(xs, encoder, ts)
  errors = {}

  for i,k in enumerate(numerical):
    n = numerical[k]
    a = analytical[k]
    e = np.zeros_like(a)
    nmin, nmax = None, None
    for j,m in enumerate(n):
      e.flat[m] = np.fabs(n[m] - a.flat[m])
      if nmin: nmin = min(nmin, n[m])
      else: nmin = n[m]
      if nmax: nmax = max(nmax, n[m])
      else: nmax = n[m]
    errors[k] = e
    print('{}: samples {}\n\trange n [{}, {}] \
    \n\trange a [{}, {}]\n\tmax err = {}'.format(k, len(n), \
    nmin, nmax, np.min(a), np.max(a), np.max(errors[k])))

if __name__ == "__main__":

  encoder = {}
  #xs = np.zeros((S,M), dtype=datatype)
  xs = np.random.rand(M, S).astype(datatype)
  ts = np.zeros((M, S), dtype=datatype)

  for s in range(S):
    m = s % M
    ts[m,s] = 1

  # V Q K
  WxVQK = np.random.randn(M, 3*HN).astype(datatype) * 0.01
  Wzy = np.random.randn(HN*3, M).astype(datatype) * 0.01
  encoder['WxVQK']=WxVQK
  encoder['Wzy']=Wzy

  states = forward(xs, encoder)
  loss = lossfun(states, ts)
  analytical = backward(states, encoder, ts)
  checkgrads(xs, encoder, ts, analytical)
