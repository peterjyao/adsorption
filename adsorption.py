# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:09:01 2018

@author: pyao1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.integrate import odeint
from scipy.optimize import fmin
from scipy.stats import linregress
#from lmfit import minimize, Parameters, Parameter, report_fit
import os

os.chdir('C:\\Users\\pyao1\\python\\adsorption')

time = np.array([0,30,60,90,120,150,180,210,240,270,
                 300,330,360,390,420,450,480,510,540,
                 570,600,630,660,690,720,750,780,810,
                 840,870,900,930,960,990,1020,1050,1080,
                 1110,1140,1170,1200,1230,1260,1290,
                 1320,1350,1380,1410,1440,1470,1500])

volume = np.array([0,13.2522,22.8151,35.4172,48.2249,
                   63.3103,73.558,86.1327,96.7225,
                   107.9742,118.6576,128.7267,140.6426,
                   152.4861,162.8935,174.1857,185.7268,
                   197.1192,209.5242,221.4678,232.2868,
                   247.6275,259.4228,270.9724,281.8857,
                   293.4212,304.6134,315.6425,326.8809,
                   338.671,348.127,363.4559,374.8323,
                   385.8517,396.7044,408.1897,420.0901,
                   427.0895,437.5906,448.138,459.2603,
                   469.2068,480.2171,491.9888,502.4742,
                   513.0803,524.2078,535.2854,545.8359,
                   555.0863,564.3464])    
    
concentration = np.array([3.023798609,3.114830065,3.193132997,
                 3.344098032,3.545952177,3.71131258,
                 3.880179942,4.051567078,4.145738602,
                 4.204649925,4.310641861,4.373957515,
                 4.449874592,4.523475265,4.557658434,
                 4.621247387,4.665646744,4.688958049,
                 4.724696445,4.763434982,4.777761936,
                 4.817210579,4.840685558,4.846937061,
                 4.861563492,4.862069321,4.865633011,
                 4.883685875,4.888158989,4.893807769,
                 4.89773283,4.902150154,4.908982992,
                 4.91435442,4.919867516,4.916160679,
                 4.928572059,4.917313385,4.922553825,
                 4.919561267,4.92966497,4.934463024,
                 4.935445404,4.937192059,4.930878401,
                 4.932299042,4.933749437,4.929975319,
                 4.935555362,4.937137485,4.940053844])    
    
# data = pd.read_csv('data.txt',delimiter='\t',index_col=None)

def rates(state, t, paras):
    c, q = state

#    paras['k1'].value = k1
#    paras['k2'].value = k2
#    paras['qmax'].value

#    k1 = paras['k1'].values[0]
#    k2 = paras['k2'].values[0]
#    qmax = paras['qmax'].values[0]

    k1 = paras[0]
    k2 = paras[1]
    qmax = paras[2]

    dcdt = Vdot/V*(c_input - c) - k1*c*(qmax - q) + k2*q
    dqdt = k1*c*(qmax - q) - k2*q
    return [dcdt, dqdt]

def conc(t, init_state, paras):
    x = odeint(rates, init_state, t, args=(paras,))
    return(x)
    
def residual_sq(paras, t, init_state, data):
    model = conc(t, init_state, paras)
    c_model = model[:,0]
    e = ((c_model - data) **2 ).sum()
    return(e)
    
#k1, k2, qmax = 0.0001, 0.0001, 0.01
#p_fit = pd.DataFrame([k1, k2, qmax])
#p_fit = p_fit.T
#p_fit.columns = ['k1','k2','qmax']
    
V = 50
Vdot = linregress(time, volume).slope
c_input = 5    
    
c0 = concentration[0]
q0 = 30
init_state = c0, 30

p_guess = [0.01, 0.01, 20]

k1fit, k2fit, qmaxfit = fmin(residual_sq,p_guess,args=(time,init_state,concentration))

plt.plot(time,concentration)
plt.plot(time,conc(time,init_state,[k1fit,k2fit,qmaxfit])[:,0])
print([k1fit,k2fit,qmaxfit])
print(qmaxfit * (k1fit/k2fit*c_input)/(1+(k1fit/k2fit*c_input)))

plt.plot(time,conc(time,init_state,[k1fit,k2fit,qmaxfit])[:,1])

