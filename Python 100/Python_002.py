# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:12:01 2017

http://www.runoob.com/python/python-exercise-example2.html
Python 练习实例2
Python 100例 Python 100例
题目：企业发放的奖金根据利润提成。利润(I)低于或等于10万元时，奖金可提0.1；
利润高于10万元，低于20万元时，低于10万元的部分按0.1提成，高于10万元的部分，可提成0.075；20万到40万之间时，高于20万元的部分，可提成0.05；
40万到60万之间时高于40万元的部分，可提成0.03；60万到100万之间时，高于60万元的部分，可提成1.0.05，高于100万元时，超过100万元的部分按1%提成，从键盘输入当月利润I，求应发放奖金总数？
程序分析：请利用数轴来分界，定位。注意定义时需把奖金定义成长整型。

@author: vincchen
"""


profit = float(raw_input('input profit:'))

if profit <= 100000:
    bonus = profit * 0.1
elif profit > 100000 and profit < 200000:
    bonus = 100000 * 0.1 + (profit - 100000) * 0.075
elif profit >= 200000 and profit < 400000:
    bonus = 100000 * 0.1 + 100000 * 0.075 + (profit - 200000) * 0.05
elif profit >= 400000 and profit < 600000:
    bonus = 100000 * 0.1 + 100000 * 0.075 + + 200000 * 0.05 + (profit - 400000) * 0.03
elif profit >= 600000 and profit < 1000000:
    bonus = 100000 * 0.1 + 100000 * 0.075 + + 200000 * 0.05 + 200000 * 0.03 + (profit - 600000) * 0.015
else:
    bonus = 100000 * 0.1 + 100000 * 0.075 + + 200000 * 0.05 + 200000 * 0.03 + 400000 * 0.015 + (profit - 1000000) * 0.01

print 'profit is: %f' %profit
print 'bonus is: %f' %bonus