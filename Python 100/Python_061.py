#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: test.py
@time: 2017/8/27 19:45
author: vincent.chen
contact: vincentchan.sysu@gmail.com

打印出杨辉三角形（要求打印出10行如下图）。　

"""
import numpy as np

n_line = 10
n_row = n_line

Pascal_triangle = np.zeros((n_line, n_row))
Pascal_triangle[:, 0] = 1

for i in np.arange(1, n_line):
    for j in np.arange(1, i+1):
        Pascal_triangle[i, j] = Pascal_triangle[i-1, j-1] + Pascal_triangle[i-1, j]

for i in np.arange(0, n_line):
    for j in np.arange(0, i+1):
        print("{0:d}\t".format(int(Pascal_triangle[i, j])), end='')
        if j == i:
            print('\n')
