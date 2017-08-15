# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:07:49 2017

http://www.runoob.com/python/python-exercise-example35.html
Python 练习实例35
Python 100例 Python 100例
题目：文本颜色设置。
程序分析：无。

@author: vincchen
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
print bcolors.WARNING + "警告的颜色字体?" + bcolors.ENDC