# -*- coding: utf-8 -*-
# author: krocki
import numpy as np
import pprint

HN = 8
M = 4
S = 5

datatype = np.float64

def samplegrad(xs, model):

  samples = 10
  delta = 1e-5

  grads = {}

  for p in model:
    p_shape=p.size
    R = np.random.randint(0, p_shape, size=samples)
    for r in R:
      p_old = p.flat[r]
#
      p.flat[r] = p_old + delta
      y0 = forward(xs, model)
      print('0 p[{}]={}, y0={}'.format(r, p.flat[r], y0))
      p.flat[r] = p_old - delta
      y1 = forward(xs, model)
      print('1 p[{}]={}, y0={}'.format(r, p.flat[r], y1))
      gn = (y0 - y1) / (2 * delta)
      print('grad n={}'.format(gn))
      p.flat[r] = p_old
      print('diff')
      print(np.fabs(p.flat[r] - p_old))
      print(r, p_shape)

      grads[r] = gn

  return grads

def forward(xs, model):
  y = np.dot(model[0].T, xs)
  return np.linalg.norm(y)

if __name__ == "__main__":

  encoder = []
  #xs = np.zeros((S,M), dtype=datatype)
  xs = np.random.rand(M, S).astype(datatype)

  # V Q K
  WxVQK = np.random.randn(M, 3 * HN).astype(datatype) * 0.01
  encoder.append(WxVQK)

  numerical = samplegrad(xs, encoder)
  pprint.pprint(numerical)
