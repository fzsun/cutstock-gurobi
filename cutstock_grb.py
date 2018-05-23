#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 22:02:13 2018

@author: leosun
"""

import numpy as np
import logging
from gurobipy import *

def keyboard_terminate(model, where):
    try:
        pass
    except KeyboardInterrupt:
        model.terminate()

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('RunLog.log')
    formatter = logging.Formatter(
        fmt='%(asctime)s %(filename)s:%(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
logger.info('Begin')

# =========================== Parameters ======================================
np.random.seed(1)
TOL = 1e-6
W_roll = 100 # stock width
I = list(range(5)) # item set
w = np.random.randint(1, 50, len(I)).tolist() # width of each item
d = np.random.randint(1, 50, len(I)).tolist() # demand of each item
patterns = np.zeros((len(I), len(I))).astype(int).tolist() # initial pattern
for i in I:
    patterns[i][i] = W_roll//w[i]

# ========================= Master Problem ====================================
m = Model('cutstock') # master problem
x = m.addVars(I, obj=1, vtype='C', name='x')
c1 = m.addConstrs((patterns[i][i] * x[i] >= d[i] for i in I), name='c1')
m.ModelSense = GRB.MINIMIZE

# ======================= Subproblem and Iteration ============================
min_rc = -100 # minimum reduced cost
n_iter = 0
while min_rc < -TOL:
    m.update()
    m.write('cutstock.lp')
    m.optimize(keyboard_terminate)
    price = [c1[i].pi for i in I]
    print(f'Price = {price}')

    sp = Model('subproblem')
    use = sp.addVars(I, obj=price, vtype='I', name='use')
    c2 = sp.addConstr(quicksum(w[i]*use[i] for i in I) <= W_roll)
    sp.ModelSense = GRB.MAXIMIZE
    sp.write('subproblem.lp')
    sp.optimize(keyboard_terminate)
    min_rc = 1 - sp.objVal
    if min_rc < -TOL:
        pattern_new = [int(use[i].x) for i in I]
        patterns.append(pattern_new)
        logger.info(f'min reduced cost = {min_rc:.2f}; new pattern: {pattern_new}')
        n_iter += 1
        x[n_iter+len(I)-1] = m.addVar(obj=1, vtype='C',
                                      column=Column(pattern_new, c1.values()))

# ====================== Relaxed Model Result =================================
logger.info(f'min reduced cost = {min_rc:.2f} >= 0')
relaxed_result = [f'{v.x:.4f}: {patterns[p]}' for
                  p, v in enumerate(m.getVars()) if v.x > TOL]
relaxed_result.insert(0, f'Relaxed result = {m.objVal:.2f} rolls')
logger.info('\n\t'.join(relaxed_result))

# ====================== Integer Model Result =================================
m.setAttr('VType', x.values(), 'I'*len(x))
m.update()
m.write('cutstock.lp')
m.optimize(keyboard_terminate)
m_result = [f'{int(v.x)}: {patterns[p]}' for
            p, v in enumerate(m.getVars()) if v.x > TOL]
m_result.insert(0, f'Integer result = {int(m.objVal)} rolls')
logger.info('\n\t'.join(m_result))
