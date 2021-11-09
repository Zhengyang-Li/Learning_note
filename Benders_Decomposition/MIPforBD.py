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
b = 100 * np.ones(len(fundSet)+len(bankSet))
b[0] = total_amount

'''
Model
'''
# variable
m = gp.Model("MIP")
x = m.addMVar( len(fundSet), vtype='C', name="Funds")
y = m.addMVar( 1, vtype='I', name='Saving')

# obj. function
m.setObjective(c @ x + f @ y, GRB.MAXIMIZE)

# Constraints
m.addConstr(A @ x + B @ y <= b)
m.optimize()

print('Fund invest =', x.x)
print('Saving amount =', y.x)
print('Profit =', m.objVal)
