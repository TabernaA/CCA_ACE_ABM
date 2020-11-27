# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:39:43 2020

@author: TabernaA
"""

#sfrom model.Batch_run import run_data

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats as st
import math


'''
Format data for plotting
data : dataframe resulting from a batch run
mv   : desired macro variable to extract
runs, steps, means : pass True if you want the data to be collected
'''
def extract_data(data, mv, runs=True, steps=True, means=True):
    return_data = {}
    if runs:
        data_per_run = [[],[]] #[[region0], [region1]]
        # arrange the data run-wise for raw plotting
        runs = len(data)
        for i in range(runs):
            data_per_run[0].append([x[0] for x in data[i][mv]])
            data_per_run[1].append([x[1] for x in data[i][mv]])
        return_data['runs'] = data_per_run

    if steps or means:
        # arrange the data stepwise for mean_ci plotting
        data_stepwise = [[],[]]  #[[region0], [region1]]
        for i in range(len(data_per_run[0])):
            data_stepwise[0].append([x[i] for x in data_per_run[0]])
            data_stepwise[1].append([x[i] for x in data_per_run[1]])
        return_data['steps'] = data_stepwise

    if means:
        # get mean value per step
        means = [[], []]
        for i in range(len(data_stepwise[0])):
            means[0].append(np.mean(data_stepwise[0][i]))
            means[1].append(np.mean(data_stepwise[1][i]))
        return_data['means'] = means


    return return_data


'''
def extract_data(data, mv, runs=True, steps=True, means=True):
    return_data = {}
    if runs:
        data_per_run = [[],[]] #[[region0], [region1]]
        # arrange the data run-wise for raw plotting
        runs = len(data)
        for i in range(runs):
            data_per_run[0].append([math.log(x[0]) for x in data[i][mv]])
            data_per_run[1].append([math.log(x[1]) for x in data[i][mv]])
        return_data['runs'] = data_per_run

    if steps or means:
        # arrange the data stepwise for mean_ci plotting
        data_stepwise = [[],[]]  #[[region0], [region1]]
        for i in range(len(data_per_run[0])):
            data_stepwise[0].append([x[i] for x in data_per_run[0]])
            data_stepwise[1].append([x[i] for x in data_per_run[1]])
        return_data['steps'] = data_stepwise
are tff 
    if means:
        # get mean value per step
        means = [[], []]
        for i in range(len(data_stepwise[0])):
            means[0].append(np.mean(data_stepwise[0][i]))
            means[1].append(np.mean(data_stepwise[1][i]))
        return_data['means'] = means


    return return_data
'''
'''
Plot raw data as scatterplot or graph, and in 1 or 2 plots
data  : run-wise data [[region0], [region1]]
title : title of the graph
mode  : 'scatter' = scatterplot, 'graph' = line graph
plotnum : number of plots, 1 = both regions on the same plot, 2 = each region gets its own plot
'''
def plot_raw(data, title, mode='scatter', plotnum=2):
    runs = len(data[0])
    steps = len(data[0][0])
    cons = list(range(steps))
    x=[]
    for i in range(runs):
        x.append(cons)

    if plotnum == 2:
        fig, (ax0, ax1) = plt.subplots(2, 1)
        fig.suptitle(title)
        if mode == 'scatter':
            # plot raw data per region
            ax0.scatter(x, data[0], s=8)
            ax0.set_ylabel("Region 0")

            ax1.scatter(x, data[1], s=8)
            ax1.set_ylabel("Region 1")
        elif mode == 'graph':
            for i in range(runs):
                ax0.plot(x[0], data[0][i])
                ax1.plot(x[0], data[1][i])
            ax0.set_ylabel("Region 0")
            ax1.set_ylabel("Region 1")

    elif plotnum == 1:
        fig = plt.figure()
        fig.suptitle(title)
        ax0 = fig.add_subplot(111)
        if mode == 'scatter':
            ax0.scatter(x, data[0], s=8, c='r', label='Region 0')
            ax0.scatter(x, data[1], s=8, c='b', label='Region 1')
        elif mode == 'graph':
            for i in range(runs):
                ax0.plot(x[0], data[0][i], c='r', label='Region 0')
                ax0.plot(x[0], data[1][i], c='b', label='Region 1')
        plt.legend(loc='upper right')
    plt.show()


'''
Plot mean values and confidence intervals
data : stepwise data from both regions
title : title of the graph
mode  : 'boxplot' or 'graph'
plotnum : 1 or 2
ci : confidence interval, 0.95 default
'''
def plot_mean_ci(data, title, mode='boxplot', plotnum=2, ci=0.95):
    intervals = [[],[]]
    means = [[],[]]
    lowbounds = [[],[]]
    upbounds = [[],[]]
    x = list(range(len(data[0])))
    # calculate mean and confidence interval for each step
    for i in range(len(data[0])):
        mean0 = np.mean(data[0][i])
        mean1 = np.mean(data[1][i])
        (lowbound0, upbound0) = st.t.interval(alpha=ci, df=len(data[0][i])-1, loc=np.mean(data[0][i]), scale=st.sem(data[0][i]))
        (lowbound1, upbound1) = st.t.interval(alpha=ci, df=len(data[1][i])-1, loc=np.mean(data[1][i]), scale=st.sem(data[1][i]))
        if math.isnan(lowbound0):
            (lowbound0, upbound0) = (mean0, mean0)
        if math.isnan(lowbound1):
            (lowbound1, upbound1) = (mean1, mean1)

        if mode == 'boxplot':
            intervals[0].append([lowbound0, mean0, upbound0])
            intervals[1].append([lowbound1, mean1, upbound1])
        elif mode == 'graph':
            means[0].append(mean0)
            means[1].append(mean1)
            lowbounds[0].append(lowbound0)
            lowbounds[1].append(lowbound1)
            upbounds[0].append(upbound0)
            upbounds[1].append(upbound1)

    # make plots depending on input parameters
    if plotnum == 2:
        fig, (ax0, ax1) = plt.subplots(2, 1)
        fig.suptitle(title)
        if mode == 'boxplot':
            # make boxplots per region
            ax0.boxplot(intervals[0])
            ax0.set_ylabel("Region 0")

            ax1.boxplot(intervals[1])
            ax1.set_ylabel("Region 1")
            ax1.set_xlabel("time step")
        elif mode == 'graph':
            ax0.plot(x, means[0], c='r', label='Region 0')
            ax0.fill_between(x, lowbounds[0], upbounds[0], color='r', alpha=.1)
            ax1.plot(x, means[1], c='b', label='Region 1')
            ax1.fill_between(x, lowbounds[1], upbounds[1], color='b', alpha=.1)
    elif plotnum == 1:
        fig = plt.figure()
        fig.suptitle(title)
        ax0 = fig.add_subplot(111)
        if mode == 'graph':
            ax0.plot(x, means[0], c='r', label='Region 0')
            ax0.fill_between(x, lowbounds[0], upbounds[0], color='r', alpha=.1)
            ax0.plot(x, means[1], c='b', label='Region 1')
            ax0.fill_between(x, lowbounds[1], upbounds[1], color='b', alpha=.1)
        plt.legend(loc='upper right')
    plt.show()


'''
Plot two variables in the same plot. Meant to be used with mean data
data_var1 : stepwise data for the first variable for both regions [[region0], [region1]]
data_var2 : stepwise data for the second variable for both regions [[region0], [region1]]
var1, var2 : variable names to be displayed in the legend
'''
def plot_compare(data_var1, data_var2, title, var1, var2):
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle(title)
    x = list(range(len(data_var1[0])))

    l1 = ax0.plot(x, data_var1[0], c='r')[0]
    l2 = ax0.plot(x, data_var2[0], c='b')[0]
    ax0.set_ylabel("Region 0")

    ax1.plot(x, data_var1[1], c='r')
    ax1.plot(x, data_var2[1], c='b')
    ax1.set_ylabel("Region 1")

    fig.legend([l1,l2],
               labels=[var1, var2],
               loc='center right',
               borderaxespad=0.1,
               title="Legend")
    plt.subplots_adjust(right=0.85)
    plt.show()
    
    
    

def plot_compare(data_var1, data_var2, title, var1, var2):
    fig, (ax0, ax1) = plt.subplots(2, 1)
    fig.suptitle(title)
    x = list(range(len(data_var1[0])))

    l1 = ax0.plot(x, data_var1[0], c='r')[0]
    l2 = ax0.plot(x, data_var2[0], c='b')[0]
    ax0.set_ylabel("Region 0")
    
    ax1.plot(x, data_var1[1], c='r')
    ax1.plot(x, data_var2[1], c='b')
    ax1.set_ylabel("Region 1")

    fig.legend([l1,l2],
               labels=[var1, var2],
               loc='center right',
               borderaxespad=0.1,
               title="Legend")
    plt.subplots_adjust(right=0.85)
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
    ax0.set_ylabel("Region 0")

    ax1.plot(data1)
    ax1.set_ylabel("Region 1")

    ax1.set_xlabel('steps')
    plt.show() #optional
    
def plot_list_3var_comp_first_difference(variable1,variable2, variable3,  steps, step_start, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    for i in range(step_start ,steps):
        data0.append(variable1[i][0])
        data1.append(variable1[i][1] ) #- variable1[i -1][1])
        data2.append(variable2[i][0] ) #- variable3[i -1][0])
        data3.append(variable2[i][1] ) #- variable3[i -1][1])
        data4.append(variable3[i][0] ) #- variable4[i -1][0])
        data5.append(variable3[i][1] ) #- variable4[i -1][1])
        
    fig, (ax0, ax1, ax2) = plt.subplots(3,1)
    fig.suptitle(title)
    x = np.arange(step_start, steps, 1)
    ax0.plot( data0 )
    ax0.plot( data1)
    ax0.set_ylabel("Cons firms")
    
    ax0.axes.xaxis.set_visible(False)

    

    ax1.plot(data2, label = "Region 0")
    ax1.plot(data3 , label = "Region 1")
    ax1.set_ylabel("Cap firms")
    #ax1.legend( loc='center left')
    ax1.axes.xaxis.set_visible(False)
    
    ax2.plot(x,data4)
    ax2.plot(x, data5 )
    ax2.set_ylabel("Households")
    
    
    #ax0.set_yscale("log")
    


    
    #ax1.set_yscale("log")
    ax1.set_xlabel('steps')
    #plt.legend(fontsize=8)
    plt.show() #optional
    
def plot_list_2var_comp_1_log(variable1,variable2, steps, step_start, title="Graph"):
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
    ax0.plot(data0)
    ax0.plot(data1)
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data2)
    ax1.plot(data3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
    plt.show() #optional
    
   
    
'''
Plotting micro variables that are stored as [region0, region1] lists
'''
def plot_list_2var_log(variable1,variable2, steps, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    for i in steps:
        data0.append(variable1[i][0])
        data1.append(variable1[i][1])
        data2.append(variable2[i][0] - variable2[i -1][0])
        data3.append( variable2[i][1] - variable2[i -1][1])
    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    ax0.plot(data0)
    ax0.plot(data2)
    ax0.set_yscale("log")
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data1)
    ax1.plot(data3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
    plt.show() #optional
    
def plot_list_comparison(variable, steps, title="Graph"):
    data0 = []
    data1 = []
    for i in steps:
        data0.append(variable[i][0])
        data1.append(variable[i][1])
    data0.plot()
    data1.plot()


    plt.show() #optional
    

    
    
def plot_list_scatter(fixed_parameter, variable_parameter,  title="Graph"):
    data0 = []
    data1 = []
    data2 = [] 
    for i in range(len(variable_parameter)):
        data0.append(fixed_parameter[i][0])
        data1.append(fixed_parameter[i][1])
        #data2.append(variable_parameter)

    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    ax0.scatter(variable_parameter, data0 )
    #ax0.xscale("log")
    ax0.set_ylabel("Region 0")
    ax1.scatter(variable_parameter, data1)
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('variable parameter')
   # ax1.yscale("log")
    plt.show() #optional
'''  
def first_difference(variable, steps):
    first_difference0 = zeros(steps)
    first_difference1 = zeros(steps)
    for i in range(1, steps ):
        first_difference0[i] =  variable[i][0] - variable[i -1][0]
        #first_difference1[i] = variable[i][1] - variable[i -1][1]
    return first_difference0 #, first_difference1]
'''  

def plot_list_3var_comp_first_difference_comp(variable1,variable2, variable3, variable4, variable5, variable6, variable7, variable8,   steps, step_start, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    for i in range(step_start ,steps):
        data0.append(variable1[i][0] )
        data1.append(variable2[i][0] ) #- variable2[i -1][0])
        data2.append(variable3[i][0]) #- variable3[i -1][0])
        data3.append(variable4[i][0] )
        data4.append(variable5[i][0]) #) # - variable5[i -1][0])
        data5.append(variable6[i][0] )
        data6.append(variable7[i][0]) #) # - variable5[i -1][0])
        data7.append(variable8[i][0] ) #- variable6[i -1][0])

        
    fig, (ax2, ax3) = plt.subplots(2,1)
    fig.suptitle(title)
    x = np.arange(step_start, steps, 1)
    '''
    ax0.plot( x,  data0, label = "A" )
    ax0.plot( x, data3, label = "B" )
    ax0.set_ylabel("Region 0 H")
    
    #ax0.set_yscale("log")
    ax0.axes.xaxis.set_visible(False)
    ax1.plot(x,  data1, label = "A")
    ax1.plot(x,  data4, label = "B")
    ax1.set_ylabel("Reg 0 Cons")
    #ax1.set_yscale("log")
    #ax1.axes.xaxis.set_visible(False)

    '''
    ax2.plot(x, data2, label = "A")
    ax2.plot(x, data5, label = "B")
    ax2.set_ylabel("Reg 0 Cap")
    ax2.axes.xaxis.set_visible(False)
    
    
    
    ax3.plot(x, data6, label = "A")
    ax3.plot(x, data7, label = "B")
    ax3.set_ylabel("Reg 0 R coeff")
    #ax2.axes.xaxis.set_visible(False)
    #ax1.set_yscale("log")
    


    
    #ax1.set_yscale("log")
    ax2.set_xlabel('steps')
   
    plt.legend(fontsize=8)
    plt.show() #optional



def plot_list_3var_comp_first_difference_comp2(variable1,variable2, variable3, variable4,   steps, step_start, title="Graph"):
    data0 = []
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    for i in range(step_start ,steps):
        #data0.append(variable1[i])
        #data1.append(variable2[i]) #- variable2[i -1][0])
        data2.append(variable3[i][0]) #- variable3[i -1][0])
        data3.append(variable3[i][1] )
        data4.append(variable4[i][0]) #) # - variable5[i -1][0])
        data5.append(variable4[i][1] ) #- variable6[i -1][0])

        
    fig, (ax0, ax1) = plt.subplots(2,1)
    fig.suptitle(title)
    x = np.arange(step_start, steps, 1)
    #ax0.plot(  data0, label = "CCA investment" )
    ax0.plot( data2, label = "CCA labour coeff")
    ax0.plot(data3, label = "CCA capital coeff")
    ax0.set_ylabel("Region 0 CCA data")
    #ax0.set_yscale("log")
    ax0.axes.xaxis.set_visible(False)


    
    #ax1.plot(  data1, label = "CCA investment" )
    ax1.plot( data4, label = "CCA labour coeff")
    ax1.plot(data5, label = "CCA capital coeff")
    #ax0.set_yscale("log")
    ax1.set_ylabel("Reg 0 CCA ")
    #ax1.set_yscale("log")
    
    
    #ax0.set_yscale("log")
    


    
    #ax1.set_yscale("log")
    ax1.set_xlabel('steps')
    plt.legend(fontsize=8)
    plt.show() #optional


def plot_list_2var_comp_first_difference(variable1,variable2, steps,step_start, title="Graph"):
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
    ax0.set_ylabel("GDP (log)")
    #ax0.axes.yaxis.set_visible(False)
    

    ax1.plot(x, data2)
    ax1.plot(x, data3)
    ax1.set_ylabel("Average productivity (log)")
    ax1.set_yscale("log")
    ax1.set_xlabel('steps')
    ax0.legend( loc='best')
    plt.legend(fontsize=8)
    plt.show() #optional