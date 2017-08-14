#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
http://www.runoob.com/python/python-exercise-example13.html
Python 练习实例13
Python 100例 Python 100例
题目：打印出所有的"水仙花数"，所谓"水仙花数"是指一个三位数，其各位数字立方和等于该数本身。
例如：153是一个"水仙花数"，因为153=1的三次方＋5的三次方＋3的三次方。
程序分析：利用for循环控制100-999个数，每个数分解出个位，十位，百位。
"""

narcissistic_number = []

for n in range(100, 1000):
    n_2 = n / 100
    n_1 = (n - n_2 * 100) / 10 # n / 10 % 10
    n_0 = n - n_2 * 100 - n_1 * 10 # n % 10
    if n == n_2**3 + n_1**3 + n_0**3:
        narcissistic_number.append(n)

print 'narcissistic_number in 100-999: ', narcissistic_number
