# -*- coding: utf-8 -*-
# author: krocki

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 2})

def save_arr_img(fname, arr):
  img = plt.imshow((arr), cmap='viridis', interpolation='nearest')
  plt.xticks([]),plt.yticks([])
  plt.savefig("{:}".format(fname))

def save_dict(fname, d):
  np.save(fname, d)

def save_dict_img(fname, dicts):

  total_len = sum([len(d) for d in dicts])
  strs = ['model', 'grads', 'states']
  q=3
  w=total_len//(q+1)
  h=total_len//w

  fig, axs = plt.subplots(h, w, constrained_layout=True)

  n=0
  for idx,d in enumerate(dicts):
    for name in d:
      data = d[name]
      axs[n//w, n%w].imshow(data, cmap='viridis', interpolation='nearest')
      axs[n//w, n%w].set_title('{}: {} [{},{}]'.format(strs[idx], name, data.shape[0], data.shape[1]))
      axs[n//w, n%w].get_xaxis().set_ticks([])
      axs[n//w, n%w].get_yaxis().set_ticks([])
      n+=1

  plt.savefig('{:}'.format(fname))

