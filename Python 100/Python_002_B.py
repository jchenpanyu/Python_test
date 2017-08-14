# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:12:01 2017

http://www.runoob.com/python/python-exercise-example2.html
Python 练习实例2
Python 100例 Python 100例
题目：企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提0.1；利润高于10万元，低于20万元时，低于10万元的部分按0.1提成，高于10万元的部分，可提成0.075；20万到40万之间时，高于20万元的部分，可提成0.05；40万到60万之间时高于40万元的部分，可提成0.03；60万到100万之间时，高于60万元的部分，可提成0.015，高于100万元时，超过100万元的部分按0.01提成，从键盘输入当月利润I，求应发放奖金总数？
程序分析：请利用数轴来分界，定位。注意定义时需把奖金定义成长整型。

@author: vincchen
"""

import numpy as np

profit = float(raw_input('input profit:'))
bonus = 0

profit_set = [0, 1*10**5, 2*10**5, 4*10**5, 6*10**5, 10*10**5]
profit_set_BonusRate = [0.1, 0.075, 0.05, 0.03, 0.015, 0.01]

for i in np.arange(5, -1, -1):
    if profit > profit_set[i]:
        bonus = bonus + (profit - profit_set[i]) * profit_set_BonusRate[i]
        profit = profit_set[i]

print 'profit is: %f' %profit
print 'bonus is: %f' %bonus
