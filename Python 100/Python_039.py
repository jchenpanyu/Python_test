# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:12:23 2017

http://www.runoob.com/python/python-exercise-example39.html
Python 练习实例39
Python 100例 Python 100例
题目：有一个已经排好序的数组。现输入一个数，要求按原来的规律将它插入数组中。
程序分析：首先判断此数是否大于最后一个数，然后再考虑插入中间的数的情况，插入后此元素之后的数，依次后移一个位置。

@author: vincchen
"""

num_set = [2, 6, 15, 45, 67, 88, 89, 93]
print num_set

insert_num = int(raw_input('input a number between 0-100:'))

n_min = 0
n_max = len(num_set)
n_mid = (n_min + n_max) / 2

while True:
    if insert_num <= num_set[n_mid] and insert_num >= num_set[n_mid-1]:
        num_set.insert(n_mid, insert_num)
        break
    elif insert_num < num_set[n_mid]:
        n_max = n_mid
        n_mid = (n_min + n_max) / 2
    else:
        n_min = n_mid
        n_mid = (n_min + n_max) / 2
    
    if n_mid == 0:
        num_set.insert(0, insert_num)
        break
    elif n_mid == len(num_set)-1:
        num_set.append(insert_num)
        break
        
print num_set
    