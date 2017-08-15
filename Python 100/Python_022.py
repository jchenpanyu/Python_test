# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:57:03 2017

http://www.runoob.com/python/python-exercise-example22.html
Python 练习实例22
Python 100例 Python 100例
题目：两个乒乓球队进行比赛，各出三人。甲队为a,b,c三人，乙队为x,y,z三人。已抽签决定比赛名单。
有人向队员打听比赛的名单。a说他不和x比，c说他不和x,z比，请编程序找出三队赛手的名单。

@author: vincchen
"""

import numpy as np
team_1 = ['a', 'b', 'c']
team_2 = ['x', 'y', 'z']

team = []

for a in team_1:
    for b in team_2:
        team.append(a+b)

for i, s in zip(range(len(team)), team):
    if s == 'ax':
        team[i] = '-'
    if s == 'cx':
        team[i] = '-'
    if s == 'cz':
        team[i] = '-'

for i in range(len(team_1)):
    for j in range(len(team_2)):
        print team[i*3+j], '\t',
        if j == 2:
            print '\n'