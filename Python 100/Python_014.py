#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@file: Python_014.py
@time: 2017/8/14 22:35
author: vincent.chen
contact: vincentchan.sysu@gmail.com

http://www.runoob.com/python/python-exercise-example14.html
Python 练习实例14
Python 100例 Python 100例
题目：将一个正整数分解质因数。例如：输入90,打印出90=2*3*3*5。
程序分析：对n进行分解质因数，应先找到一个最小的质数k，然后按下述步骤完成：
(1)如果这个质数恰等于n，则说明分解质因数的过程已经结束，打印出即可。
(2)如果n<>k，但n能被k整除，则应打印出k的值，并用n除以k的商,作为新的正整数你n,重复执行第一步。
(3)如果n不能被k整除，则用k+1作为k的值,重复执行第一步。
程序源代码：
"""
prime_factor = []

n = int(raw_input('input a positive integer:'))

while True:
    for i in range(2, n + 1):
        if n % i == 0:
            prime_factor.append(i)
            n = n / i
            break
    if n == 1:
        break

print 'prime_factor', prime_factor