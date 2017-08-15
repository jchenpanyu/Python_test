# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:23:15 2017

http://www.runoob.com/python/python-exercise-example27.html
Python 练习实例27
Python 100例 Python 100例
题目：利用递归函数调用方式，将所输入的5个字符，以相反顺序打印出来。
程序分析：无。

@author: vincchen
"""

s=raw_input('input string')
for i in s[::-1]:
    print i