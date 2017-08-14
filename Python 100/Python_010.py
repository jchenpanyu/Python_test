# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:24:13 2017

http://www.runoob.com/python/python-exercise-example10.html
Python 练习实例10
Python 100例 Python 100例
题目：暂停一秒输出，并格式化当前时间

@author: vincchen
"""

import time
 
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
 
# 暂停一秒
time.sleep(1)
 
print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))