# -*- coding: utf-8 -*-
# author: kmrocki
# based on the original code by A.Karpathy (char-rnn)

from __future__ import print_function
import numpy as np
import argparse, sys
import datetime, time
import random
from random import uniform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

def save_arr(fname, arr):
  img = plt.imshow((arr), cmap='viridis', interpolation='nearest')
  plt.xticks([]),plt.yticks([])
  plt.savefig("{:}".format(fname))

def pos_table(N, dim):
  def get_angle(x, h):
    return x / np.power(10000, 2 * (h // 2) / dim)
  def get_angle_vec(x):
    return [get_angle(x, j) for j in range(dim)]
  tab = np.array([get_angle_vec(i) for i in range(N)]).astype(float)
  tab[:, 0::2] = np.sin(tab[:, 0::2])
  tab[:, 1::2] = np.cos(tab[:, 1::2])
  return tab

def softmax(x):
  y = x
  mx = np.max(y, axis=0)
  y -= mx # normalize
  y = np.exp(y) / np.sum(np.exp(y), axis=0)
  return y

def train(inputs, targets):

  xs, pos = {}, {}

  loss = 0
  return loss

if __name__ == "__main__":
  ### parse args
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-S', type=int, default = 10, help='seq len')
  parser.add_argument('-N', type=int, default = 32, help='hiddens')

  opt = parser.parse_args()
  S = opt.S
  HN = opt.N

  print(opt)

  # data I/O
  data = open('./alice29.txt', 'r').read()

  chars = list(set(data))
  data_size, M = len(data), len(chars)
  print('data has %d characters, %d unique.' % (data_size, M))
  char_to_ix = { ch:i for i,ch in enumerate(chars) }
  ix_to_char = { i:ch for i,ch in enumerate(chars) }

  ######
  Wev = np.random.randn(HN, M)*0.01 # emb -> value
  Weq = np.random.randn(HN, M)*0.01 # emb -> query
  Wek = np.random.randn(HN, M)*0.01 # emb -> key
  ######

  char_mat = np.eye(M)
  n = 0
  #p = np.random.randint(len(data)-1-S)
  p = 380
  inputs = np.zeros((S,M), dtype=int)
  targets = np.zeros((S,M), dtype=int)

  # pos encoding
  pos = pos_table(S, M)

  ######
  cs = data[p:p+S]
  print(cs)
  inputs[:,:] = [char_mat[char_to_ix[ch]] for ch in cs]

  cs = data[p+1:p+S+1]
  targets[:,:] = [char_mat[char_to_ix[ch]] for ch in cs]

  start = time.time()

  # combine x + positional
  xs = inputs
  save_arr('xs.png', xs.T)
  es = inputs + pos
  es = es.T
  save_arr('es.png', es)

  vs = np.dot(Wev, es)
  qs = np.dot(Weq, es)
  ks = np.dot(Wek, es)

  att = np.dot(qs, ks.T) / np.sqrt(HN)
  att_sm = softmax(att)
  zs = np.dot(att_sm, vs)

  save_arr('vs.png', vs)
  save_arr('qs.png', qs)
  save_arr('ks.png', ks)
  save_arr('att.png', att)
  save_arr('att_sm.png', att_sm)
  save_arr('zs.png', zs)
