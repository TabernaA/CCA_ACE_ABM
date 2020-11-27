# model/app.py
# bringing all elements of the model together
# contains run method

from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.classes.household import Household
from model.classes.model import KSModel
from model.modules.data_collection import *
from model.modules.data_collection_2 import *
import matplotlib.pyplot as plt
import model.modules.data_analysis as da
import scipy.stats as st        
import seaborn as sns
import math
from numpy import *
import pandas as pd



steps = 180


model = KSModel(F1 = 4, F2 = 20, H = 300, B= 1, S = 0, T= 0.01)
for i in range(steps):
    print("#------------ step", i+1, "------------#")
    model.step()
macro_variable = model.datacollector.get_model_vars_dataframe()



'''
Here below is just some plots for the results work in progress
''' 

'''
Plotting micro variables that are stored as [region0, region1] lists
'''
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
    ax0.plot(data0)
    ax0.plot(data2)
    
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data1)
    ax1.plot(data3)
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
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
    ax0.plot(data0)
    ax0.plot(data1)
    
    ax0.set_ylabel("Region 0")
    ax1.plot(data2)
    ax1.plot(data3)
    ax1.set_ylabel("Region 1")
    ax1.set_xlabel('steps')
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
    
    
    

 
data0 = [[],[]]
data1 = [[],[]]
for i in range(steps):
   data0[0].append(macro_variable.Consumption_firms_av_prod[i][0])
   data0[1].append('Region0')
   data1[0].append(macro_variable.Consumption_firms_av_prod[i][1])
   data1[1].append('Region1')

#for i in range (step):
prod = data0[0] + data1[0]
region =  data0[1] + data1[1]


macro_variable[['Cons region 0','Cons region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cons_Firms.to_list(), index= macro_variable.index)
macro_variable[['CCA coeff 0','CCA coeff 1']] = pd.DataFrame(macro_variable.Average_CCA_coeff.to_list(), index= macro_variable.index)
macro_variable[['Prod cons region 0','Prod cons region 1']] = pd.DataFrame(macro_variable.Consumption_firms_av_prod.to_list(), index= macro_variable.index)  
regional_prod_df = pd.DataFrame({"prod":prod, "region":region})  

sns.set_style('prod regions')
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)



ax = sns.lineplot(x=macro_variable.index, y='CCA coeff 0', color = "red", data=macro_variable)
ax2 = ax.twinx()
ax2 = sns.regplot(x=macro_variable.index, y='Average_CCA_coeff', color = "blue", data=macro_variable,
                  x_estimator=np.mean)

df = DataFrame (macro_variable.Consumption_firms_av_prod, columns=["wei"])

     
plot_list_3var_comp_first_difference(macro_variable.Population_Regional_Cons_Firms, macro_variable.Population_Regional_Cap_Firms, macro_variable.Population_Regional_Households, 200, 1  )     
plot_list(macro_variable.Unemployment_Regional, range(20 ,steps), "Unemployment rate") 

plot_list_2var_comp_first_difference(macro_variable.RD_CCA_INVESTMENT,macro_variable.Average_CCA_coeff,250 , 50, "Turning the tide of agglomeration")

macro_variable.Average_CCA_coeff.plot()


plot_list_2var_comp_first_difference(macro_variable.INVESTMENT,macro_variable.Average_Salary,200 , 50, "Turning the tide of agglomeration")

plot_list_2var_comp_first_difference(macro_variable.GDP,macro_variable.Consumption_firms_av_prod ,200 , 50, "Turning the tide of agglomeration")
plot_list(macro_variable.INVESTMENT, range(steps), "Investment")
plot_list(macro_variable.Competitiveness_Regional, range( 5,steps), "Competitiveness")
plot_list(macro_variable.Aggregate_Employment, range(steps), "Aggregate Employment")
#plot_list(macro_variable.Population_Regional, range(steps), "Population")
plot_list(macro_variable.Average_Salary, range(steps) , "Average Salary")
plot_list(macro_variable.Population_Regional_Households, range(steps), "Number of households")
plot_list(macro_variable.Cosumption_price_average,  range( 20, steps) , "Consumption price average")
plot_list(macro_variable.Population_Regional_Cons_Firms, range(steps), "Number of consumption firms")
plot_list(macro_variable.Capital_firms_av_prod, range(steps), " Average productivity Cap firms")
plot_list(macro_variable.Population_Regional_Cap_Firms, range(steps), "Number of capital  firms")
plot_list(macro_variable.Consumption_firms_av_prod, range(steps), " Average productivity Cons firms")
#(macro_variable.Regional_fiscal_balance, range(steps), "Regional fiscal balance")  
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
