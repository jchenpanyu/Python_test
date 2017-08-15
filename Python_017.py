# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:27:23 2017

http://www.runoob.com/python/python-exercise-example17.html
Python 练习实例17
Python 100例
题目：输入一行字符，分别统计出其中英文字母、空格、数字和其它字符的个数。
程序分析：利用while语句,条件为输入的字符不为'\n'。

@author: vincchen
"""

s = raw_input('input a line of string:')

n_letter = 0
n_space = 0
n_number = 0
n_others = 0

for c in s:
    if c.isalpha():
        n_letter += 1
    elif c.isspace():
        n_space += 1
    elif c.isdigit():
        n_number += 1
    else:
        n_others += 1

print 'char = %d,space = %d,digit = %d,others = %d' % (n_letter,n_space,n_number,n_others)