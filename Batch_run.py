# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:07:58 2020

@author: TabernaA
"""
from mesa.batchrunner import BatchRunner
from model.app import *
from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.classes.household import Household
from model.classes.model import KSModel
from model.modules.data_collection import *
import matplotlib.pyplot as plt
import seaborn as sns

fixed_params = {"width": 1,
               "height": 2,
               "F1" : 40,
               "F2" : 200,
                "H" : 10000,
                "B" : 1/2} 
variable_params = {"T" : range(0, 100, 10)}

batch_run = BatchRunner(KSModel,
                        variable_params,
                        fixed_params,
                        iterations=1,
                        max_steps=250,
                        model_reporters={
                        "Unemployment_Regional" : regional_unemployment_rate,
                        "Consumption_firms_av_prod" : productivity_consumption_firms_average,
                        "Cosumption_price_average" : price_average,
                        "Capital_firms_av_prod" : productivity_capital_firms_average,
                         "Average_Salary_Cons" : regional_average_salary_cons,
                         "GDP": gdp,
                         "CONSUMPTION" : consumption,
                         "Population_Regional_Cap_Firms" : regional_population_cap,
                         "Population_Regional_Cons_Firms" : regional_population_cons,
                         "Population_Regional_Households" : regional_population_households,
                         "Competitiveness_Regional" : regional_average_competitiveness
                         
                        })
batch_run.run_all()

run_data = batch_run.get_model_vars_dataframe()
run_data.head()

plot_list(run_data.Unemployment_Regional, run_data.T, "Unemployment Rate ")
plot_list(run_data.Cosumption_price_average, run_data.T, "Price ")
plot_list(run_data.Capital_firms_av_prod, run_data.T, "Capital  firms average production ")
plot_list(run_data.Consumption_firms_av_prod, run_data.T, "Consumption  firms average production ")
plot_list(run_data.GDP, run_data.T, "GDP ")
#plot_list(run_data., len(run_data.H), "Consumption  firms average production ")



    

data0 = []
data1 = []
for i in range(0, len(run_data.Consumption_firms_av_prod)):
    data0.append(run_data.Consumption_firms_av_prod[i][0])
    data1.append(run_data.Consumption_firms_av_prod[i][1])

plt.scatter( data0, run_data.T)

plt.scatter()

plot_list_scatter(run_data.Consumption_firms_av_prod, run_data.T,  "Consumption  firms average productivity ")
plot_list_scatter(run_data.Capital_firms_av_prod, run_data.T,  "Capital  firms average productivity ")
plot_list_scatter(run_data.Average_Salary_Cons, run_data.T,  "Consumption  firms average salary ")
plot_list_scatter(run_data.Cosumption_price_average, run_data.T,  "Consumption  firms average price ")
plot_list_scatter(run_data.Unemployment_Regional, run_data.T,  " Unemployment rate  ")



plt.scatter(run_data.H, run_data.Consumption)
plt.scatter(run_data.s, run_data.Unemployment_Region0)
plot_list(macro_variable.Consumption_firms_av_prod, steps, " Average productivity Cons firms")
#plot_list(macro_variable.Capital_firms_av_prod, steps, " Average productivity Cap firms")
#plot_list(macro_variable.Regional_fiscal_balance, steps, "Regional fiscal balance")
