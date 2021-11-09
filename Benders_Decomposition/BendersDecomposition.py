# -*- coding: utf-8 -*-
'''
Created on Wed Feb  3 16:08:35 2021

@author: LI Zhengyang
'''
'''
Mike has 1000$ to make a finance plan. There are two options:
1. Saving account
    The interest rate 4.5%.
    Do not accept fractional amount.
2. Mutual funds
    10 available funds F1, F2, ..., F10, which have the rate of return at 1%, 2%, ..., 10%, respectively.
    For each fund, Mike can only purchase 100$ at maximum.

What is the best finance plan?
The obvious answer: buy 100$ of F5, F6, ..., F10 and put the rest of the money (400$) in saving account.
'''

import numpy as np
import gurobipy as gp
from gurobipy import GRB

'''
Data structure
'''
class Bank:
    def __init__(self, name, rate, var_type):
        self.name = name
        self.rate = rate
        self.var_type = var_type
        self.amount = 0

class Fund:
    def __init__(self, name, rate, var_type):
        self.name = name
        self.rate = rate
        self.upLimit = 100
        self.var_type = var_type
        self.amount = 0

'''
Data
'''
total_amount = 1000
bankSet = {}
fundSet = {}
bankSet[0] = Bank('Saving', 0.045, 'I')
for i in range(0,10):
    fundSet[i] = Fund('Fund'+str(i+1), (i+1)*0.01, 'C')

'''
Matrix
'''
# obj. coefficient vector of fund variable
c = np.zeros(len(fundSet))
n = 0
for i in fundSet:
    c[n] = fundSet[i].rate + 1
    n += 1
# obj. coefficient vector of saving account variable
f = np.array([bankSet[0].rate + 1])
# constraint matrix of fund variable
A = np.vstack([np.ones([1, len(fundSet)]), np.identity(len(fundSet))])
# constraint matrix of saving account variable
B = np.zeros([len(fundSet)+len(bankSet), 1])
B[0] = 1
# right hand side value vector
b = 100 * np.ones([len(fundSet)+len(bankSet), 1])
b[0] = total_amount

'''
Initialization
'''
y_value = 1500
UB = float('inf')
LB = -float('inf')
accuracy = 10 ** -6
iter = 0

'''
Master problem (variables and obj. function)
'''
MP = gp.Model('Master problem')
z = MP.addMVar(1, vtype='C', lb=0, name='Profit')
y = MP.addMVar(1, vtype='I', lb=0, name='Saving')
# obj. function
MP.setObjective(z, GRB.MAXIMIZE)

'''
Benders Decomposition
'''
while UB - LB > accuracy:
    print('*******************Iteration ' + str(iter) + '*********************')
    iter += 1

    '''
    Dual sub-problem (generate cut)
    '''
    # solve the dual model
    dual_coefficient = b - np.dot(B, np.array([[y_value]]))
    extreme_ray = np.zeros(np.shape(dual_coefficient))
    extreme_point = np.zeros(np.shape(dual_coefficient))

    DSP = gp.Model('Dual sub-problem')
    mu = DSP.addMVar(len(b), vtype='C', name="Dual variables")
    # obj. function
    DSP.setObjective(np.transpose(dual_coefficient) @ mu, GRB.MINIMIZE)
    # Constraints
    DSP.addConstr(np.transpose(A) @ mu >= c)
    DSP.setParam('LogToConsole', 0)
    DSP.setParam('InfUnbdInfo', 1)
    DSP.optimize()
    x_value = DSP.pi

    if DSP.objVal <= -1 * 10 ** -30:
        # unbounded
        extreme_ray = mu.x / sum(mu.x)
        extreme_ray = np.reshape(extreme_ray, [len(b), 1])
    else:
        # bounded
        extreme_point = np.reshape(mu.x, [len(b), 1])

    '''
    Master problem (add cut)
    '''
    if sum(extreme_ray) > 0:
        # feasibility cut
        MP.addConstr(np.dot(np.transpose(b), extreme_ray) - np.dot(np.transpose(B), extreme_ray) @ y >= 0)
    else:
        # optimality cut
        MP.addConstr(z <= f @ y + np.dot(np.transpose(b), extreme_point) - np.dot(np.transpose(B), extreme_point) @ y)
        LB = max(LB, (np.dot(f, y_value) + np.dot(np.transpose(b - np.dot(B, y_value)), extreme_point))[0][0])
    MP.setParam('LogToConsole', 0)
    MP.optimize()
    UB = MP.objVal
    y_value = y.x[0]

    '''
    Output information
    '''
    print('UB =', UB)
    print('LB =', LB)
    print('Gap =', (UB - LB)/UB)
    print('Fund invest =', x_value)
    print('Saving amount =', y_value)
    print('Profit =', UB)
