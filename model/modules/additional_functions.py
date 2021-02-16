# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:47:03 2021

@author: TabernaA
"""

import numpy as np
import matplotlib.pyplot as plt

def mean_variable(dataset, variable, drop = 0, drop_end = 0):
    variable_all = dataset.filter(like = variable)
    variable_all = variable_all.drop(variable_all.index[:drop], axis = 0)
    if drop_end != 0:
        variable_all = variable_all.drop(variable_all.index[drop_end:], axis = 0)
    variable_mean = variable_all.mean(axis = 1)
    return variable_mean
    
def mean_variable_log(dataset, variable, drop= 0, drop_end = 0):
    variable_all = dataset.filter(like = variable)
    variable_all = variable_all.drop(variable_all.index[:drop], axis = 0)
    if drop_end != 0:
        variable_all = variable_all.drop(variable_all.index[drop_end:], axis = 0)
    variable_mean = variable_all.mean(axis = 1)
    variable_mean_log = np.log(variable_mean)
    return variable_mean_log

def variable_growth_rate(dataset, variable, transition, end):
    variable_all = dataset.filter(like = variable)
    variable_mean = variable_all.mean(axis = 1)
    variable_mean_log = np.log(variable_mean)
    ref = variable_mean_log.iloc[transition]
    #print(ref)
    growth_rate = (variable_mean_log.iloc[end] -  ref)/ (end - transition + 1)
    return growth_rate


def plot(variables, transition, end,  xlabel = 'Time step', ylabel = 'Population' ):
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)
    for i in range(len(variables)):
        ax.plot( variables[i][0],  label = variables[i][1] , color =  variables[i][2])	

	

    ax.set_xlabel( xlabel, fontsize = 16)	
    ax.set_ylabel(ylabel, fontsize = 16)	

    plt.legend()
    plt.show()

def plot_list(variable, steps, title="Graph"):
    data0 = []
    data1 = []
    for i in steps:
        data0.append(variable[i][0])
        data1.append(variable[i][1])
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle(title)

    ax0.plot(data0)
    ax0.set_ylabel("Coastal Region")

    ax1.plot(data1)
    ax1.set_ylabel("Inland Region")

    ax1.set_xlabel('steps')
    plt.show() #optional
    
    
'''
Here below is just some plots for the results work in progress
''' 
def plot_check(investment, gdp_cap, steps, title="Comparison orders"):
    data_inv = []
    data_cap = []
    for i in steps:
        data_inv.append(investment[i][2])
        data_cap.append(gdp_cap[i][2])
    fig, ax0 = plt.subplots(1, 1)
    fig.suptitle(title)
    ax0.plot(data_inv, label = 'data 1')
    ax0.plot(data_cap, label = 'data 2')
    ax0.set_yscale('log')
    ax0.legend()
    plt.legend()
    plt.show()
    

'''
Plotting micro variables that are stored as [region0, region1] lists
'''
def plot_list(variable, steps, title="Graph"):
    data0 = []
    data1 = []
    for i in steps:
        data0.append(variable[i][0])
        data1.append(variable[i][1])
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle(title)

    ax0.plot(data0)
    ax0.set_ylabel("Coastal Region")

    ax1.plot(data1)
    ax1.set_ylabel("Inland Region")

    ax1.set_xlabel('steps')
    plt.show() #optional
    

def plot_list_log(variable, steps, title="Graph"):
    data0 = []
    data1 = []
    for i in steps:
        data0.append(variable[i][0])
        data1.append(variable[i][1])
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle(title)

    ax0.plot(data0)
    ax0.set_yscale("log")
    ax0.set_ylabel("Coastal Region")

    ax1.plot(data1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Inland Region")

    ax1.set_xlabel('steps')
    plt.show() #optional
    
    
def plot_list_2var_reg(variable1,variable2, steps, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    for i in steps:
        data0.append(variable1[i][0])
        data1.append(variable1[i][1])
        data2.append(variable2[i][0])
        data3.append(variable2[i][1])
    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    ax0.plot(data0, label = "Variable 1")
    ax0.plot(data2, label = "Variable 2")
    ax0.legend()
    
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data1, label = "Variable 1")
    ax1.plot(data3, label = "Variable 2")
    ax1.legend()
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
    plt.legend()
    plt.show() #optional


def plot_list_2var_comp(variable1,variable2, steps, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    for i in steps:
        data0.append(variable1[i][0])
        data1.append(variable1[i][1])
        data2.append(variable2[i][0])
        data3.append(variable2[i][1])
    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    ax0.plot(data0,  )
    ax0.plot(data1)
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data2)
    ax1.plot(data3)
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
    plt.show() #optional
    
    
def plot_list_2var_comp_log(variable1,variable2, steps,step_start, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    for i in range(step_start ,steps):
        data0.append(variable1[i][0])
        data1.append(variable1[i][1])
        data2.append(variable2[i][0]) # - variable2[i -1][0])
        data3.append(variable2[i][1]) #- variable2[i -1][1])
    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    x = np.arange(step_start, steps, 1)
    ax0.plot(x, data0, label = "Region 0")
    ax0.plot(x, data1, label = "Region 1")
    ax0.set_yscale("log")
    ax0.set_ylabel("variable1")
    #ax0.axes.yaxis.set_visible(False)
    

    ax1.plot(x, data2)
    ax1.plot(x, data3)
    ax1.set_ylabel("variable2")
    ax1.set_yscale("log")
    ax1.set_xlabel('steps')
    ax0.legend( loc='best')
    plt.legend(fontsize=8)
    plt.show() #optional
    
    
    

 

    