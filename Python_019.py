# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:37:46 2017

http://www.runoob.com/python/python-exercise-example19.html
Python 练习实例19
Python 100例 Python 100例
题目：一个数如果恰好等于它的因子之和，这个数就称为"完数"。例如6=1＋2＋3.编程找出1000以内的所有完数。

@author: vincchen
"""

def factorization(n):
    factorization = []
    for i in range(1, n):
        if n % i == 0:
            factorization.append(i)
    return factorization


for n in range(2, 1001):
    dec_n = 0
    for i in factorization(n):
        dec_n = dec_n + i
    if n == dec_n:
        print n