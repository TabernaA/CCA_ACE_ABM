# model/app.py
# bringing all elements of the model together
# contains run method
seed_value = 12345678
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed=seed_value)
#from model.classes.capital_good_firm import CapitalGoodFirm
#from model.classes.consumption_good_firm import ConsumptionGoodFirm
#from model.classes.household import Household
from model.classes.model import KSModel
#from model.modules.data_collection import *
#from model.modules.data_collection_2 import *
import matplotlib.pyplot as plt
#import model.modules.data_analysis as da
import model.modules.additional_functions as af
#import scipy.stats as st        
import seaborn as sns
#import random

import pandas as pd
import statsmodels.api as sm


steps =  300


model = KSModel(F1 = 50, F2 =250, H = 3500, B= 1,  T= 2, S = 0)
model.reset_randomizer(seed_value)

for i in range(steps):
    print("#------------ step", i+1, "------------#")
    model.step()
macro_variable = model.datacollector.get_model_vars_dataframe()
micro_variable = model.datacollector.get_agent_vars_dataframe()

micro_variable = micro_variable.dropna()


macro_variable.info()


#macro_variable[['LD cons 0', 'LD cons 1']] = pd.DataFrame(macro_variable.LD_cons.to_list(), index= macro_variable.index)
macro_variable[['Cons region 0','Cons region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cons_Firms.to_list(), index= macro_variable.index)
macro_variable[['Cap region 0','Cap region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cap_Firms.to_list(), index= macro_variable.index)
macro_variable[['Households region 0','Households region 1']] = pd.DataFrame(macro_variable.Population_Regional_Households.to_list(), index= macro_variable.index)
macro_variable[['Cons price region 0','Cons price region 1']] = pd.DataFrame(macro_variable.Cosumption_price_average.to_list(), index= macro_variable.index)
#macro_variable[['CCA coeff 0','CCA coeff 1']] = pd.DataFrame(macro_variable.Average_CCA_coeff.to_list(), index= macro_variable.index)
#macro_variable[['Prod cons region 0','Prod cons region 1', 'Delta prod cons region 0', 'Delta prod cons region 1']] = pd.DataFrame(macro_variable.Consumption_firms_av_prod.to_list(), index= macro_variable.index)
macro_variable[['Prod region 0','Prod region 1', 'Delta prod region 0', 'Delta prod region 1' ]] = pd.DataFrame(macro_variable.Regional_average_productivity.to_list(), index= macro_variable.index)    
macro_variable[['GDP region 0','GDP region 1', 'GDP total']] = pd.DataFrame(macro_variable.GDP.to_list(), index= macro_variable.index)
macro_variable[['Unemployment region 0','Unemployment region 1', 'Unemployment diff 0','Unemployment diff 1', 'Unemployment total' ]] = pd.DataFrame(macro_variable.Unemployment_Regional.to_list(), index= macro_variable.index)
#macro_variable[['MS track 0', 'MS track 1']] = pd.DataFrame(macro_variable.MS_track.to_list(), index= macro_variable.index)
macro_variable[['CONS 0', 'CONS 1', 'CONS Total', 'Export']] = pd.DataFrame(macro_variable.CONSUMPTION.to_list(), index= macro_variable.index)
macro_variable[['INV 0', 'INV 1', 'INV Total']] = pd.DataFrame(macro_variable.INVESTMENT.to_list(), index= macro_variable.index)
macro_variable[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame(macro_variable.GDP_cons.to_list(), index= macro_variable.index)
macro_variable_csv_data = macro_variable.to_csv('data_model_.csv', index  = True)
macro_variable[['Aggr unemployment region 0','Aggr unemployment region 1']] = pd.DataFrame(macro_variable.Aggregate_Unemployment.to_list(), index= macro_variable.index)
macro_variable['Aggr unemployment'] = macro_variable['Aggr unemployment region 0'] + macro_variable['Aggr unemployment region 1']
macro_variable[['Aggr employment region 0','Aggr employment region 1']] = pd.DataFrame(macro_variable.Aggregate_Employment.to_list(), index= macro_variable.index)
macro_variable['Aggr employment'] = macro_variable['Aggr employment region 0'] + macro_variable['Aggr employment region 1']
macro_variable['Unemployment rate'] = macro_variable['Aggr unemployment'] / (macro_variable['Aggr unemployment'] + macro_variable['Aggr employment'])
macro_variable['Labor demand 0'] =  macro_variable['GDP region 0'] / macro_variable['Prod region 0']  /  macro_variable['Cons price region 0'] 
macro_variable['Labor demand 0'] =  macro_variable['GDP region 0'] / macro_variable['Prod region 0']  /  macro_variable['Cons price region 0'] 
macro_variable[['Wages region 0','Wages region 1','Wage diff 0', 'Wage diff 1']] = pd.DataFrame(macro_variable.Average_Salary.to_list(), index= macro_variable.index)

#cycles = sm.tsa.filters.bkfilter(macro_variable[['INV Total','CONS Total', 'GDP total']], 6, 32, 12)
#macro_variable['Households region 0 rate'] = macro_variable['Households region 0'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['Households region 1 rate'] = macro_variable['Households region 1'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['INV region 1 rate'] = macro_variable['INV 1'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['INV region 0 rate'] = macro_variable['INV 0'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['GDP region 0 rate'] = macro_variable['GDP region 0'] / (macro_variable['GDP region 0'] + macro_variable['GDP region 1'])
#fig, ax = plt.subplots()
#cycles.plot(ax=ax, style=['r--', 'b-'])
#plt.show()

transition = 60
'''
bankruptcy =  micro_variable.loc[(micro_variable['Bankrupt'] > 0)]
bankruptcy = bankruptcy.loc[140:, ]
bankruptcy['Bankrupt'].hist()
'''
#mv = micro_variable.loc[(micro_variable['Price'] > 5) & (micro_variable.index.get_level_values('Step') > 275)] 


##------log variable -----#
'''
INV
'''
inv_gr = af.variable_growth_rate(macro_variable, 'INV Total', transition, steps)
inv_gr_0 = af.variable_growth_rate(macro_variable, 'INV 0', transition, steps)
inv_gr_1 = af.variable_growth_rate(macro_variable, 'INV 1', transition, steps)
std_inv = af.mean_variable_log(macro_variable, 'INV Total', 50, 50).std()



'''
GDP 
'''


gdp_gr = af.variable_growth_rate( macro_variable, 'GDP total', transition, steps)
gdp_gr_0 = af.variable_growth_rate(macro_variable, 'GDP region 0', transition, steps)
gdo_gr_1 =af.variable_growth_rate(macro_variable, 'GDP region 1', transition, steps)

std_gdp = af.mean_variable_log(macro_variable, 'GDP total', transition, 50).std()
rel_std_inv = std_inv / std_gdp

mean_gdp_0 = af.mean_variable_log(macro_variable, 'GDP region 0')
mean_gdp_1 = af.mean_variable_log(macro_variable, 'GDP region 1')

gdp_plot = [[mean_gdp_0, 'Region 0', 'black'], [mean_gdp_1, 'Region 1', 'red']]
af.plot(gdp_plot, 0, 350)


'''
CONS
'''
cons_gr = af.variable_growth_rate(macro_variable, 'CONS Total', transition, steps)
cons_gr_0 = af.variable_growth_rate(macro_variable, 'CONS 0', transition, steps)
cons_gr_1 = af.variable_growth_rate(macro_variable, 'CONS 1', transition, steps)
std_cons = af.mean_variable_log(macro_variable, 'CONS Total', transition, 50).std()
rel_std_cons = std_cons / std_gdp

'''
UNEMPLOYMENT RATE
'''

unemployment_rate_0 = af.mean_variable(macro_variable,'Unemployment region 0', 100).mean()
unemployment_rate_1 = af.mean_variable(macro_variable,'Unemployment region 1', 100).mean()
unemployment_rate = af.mean_variable(macro_variable,'Unemployment rate', 100).mean()

af.mean_variable(macro_variable,'Unemployment region 0', 100).plot()
af.mean_variable(macro_variable,'Unemployment region 1', 100).plot()
af.mean_variable(macro_variable,'Unemployment rate', 100).plot()




af.plot_check(macro_variable.INVESTMENT, macro_variable.GDP_cap, range(steps), "Checking cap ")
af.plot_check(macro_variable.CONSUMPTION, macro_variable.GDP_cons, range(steps), "Checking cons")


macro_variable['Debt'].plot()
#for i in range (step):

'''
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


plot_list_2var_comp_first_difference(macro_variable.RD_CCA_INVESTMENT,macro_variable.Average_CCA_coeff,250 , 50, "Turning the tide of agglomeration")

#macro_variable.Average_CCA_coeff.plot()


plot_list_2var_comp_first_difference(macro_variable.INVESTMENT,macro_variable.Average_Salary,200 , 50, "Turning the tide of agglomeration")
'''
#macro_variable = macro_variables[2]

transition = 10
#plot_list_2var_comp_first_difference(macro_variable.GDP,macro_variable.Consumption_firms_av_prod ,200 , 50, "Turning the tide of agglomeration")
af.plot_list_log(macro_variable.INVESTMENT, range(transition, steps), "Investment")
af.plot_list_log(macro_variable.CONSUMPTION, range(transition, steps), "Consumption")
af.plot_list(macro_variable.Competitiveness_Regional, range(transition,  steps), "Competitiveness")
#plot_list(macro_variable.Aggregate_Employment, range(transition, steps), "Aggregate Employment")
#plot_list(macro_variable.Population_Regional, range(steps), "Population")
af.plot_list_log(macro_variable.Average_Salary, range(transition, steps) , "Average Salary")
af.plot_list(macro_variable.Population_Regional_Households, range(transition,steps), "Number of households")
af.plot_list_log(macro_variable.Cosumption_price_average,  range(transition,  steps) , "Consumption price average")
af.plot_list(macro_variable.Population_Regional_Cons_Firms, range(transition, steps), "Number of consumption firms")
#af.plot_list_log(macro_variable.Capital_firms_av_prod, range(transition, steps), " Average productivity Cap firms")
af.plot_list(macro_variable.Population_Regional_Cap_Firms, range(transition,  steps), "Number of capital  firms")
#af.plot_list_log(macro_variable.Consumption_firms_av_prod, range(transition, steps ), " Average productivity Cons firms")
af.plot_list(macro_variable.Unemployment_Regional, range( transition, steps), "Unemployment rate") 
af.plot_list_log(macro_variable.GDP, range( steps), "GDP") 

af.plot_list_2var_reg(macro_variable.GDP_cap, macro_variable.INVESTMENT, range(steps), "Check account identity cap ")
af.plot_list_2var_reg(macro_variable.GDP_cons, macro_variable.CONSUMPTION, range(steps), "Check account IDENTITY CONS  ")
#af.plot_list_2var_reg(macro_variable.Capital_firms_av_prod, macro_variable.Consumption_firms_av_prod, range(steps), "Productivity ")

af.plot_list_2var_reg(macro_variable.Average_Salary_Capital, macro_variable.Average_Salary_Cons, range(steps), "Wages")
#(macro_variable.Regional_fiscal_balance, range(steps), "Regional fiscal balance")  
af.plot_list_2var_reg(macro_variable.Average_Salary_Cons, macro_variable.Consumption_firms_av_prod, range(steps), "Wages")

comp_1 = []
comp_2 = []
comp_3 = []
for i in range(len(model.firms2)):
    comp_2.append(model.firms2[i].market_share[2])

migr_par = []
mp = []
region = []
demand_distance = []
wage_distance = []
delta_pr = []
cons_firms = model.firms2
households = model.households
migr_par_h = []

for i in range(len(households)):
    migr_par_h.append(households[i].migration_pr)

for i in range(len(cons_firms)):
    migr_par.append(cons_firms[i].distances_mig)
    
    
    
    
    mp.append(cons_firms[i].distances_mig[0])
    region.append(cons_firms[i].region_history[1])
    demand_distance.append(cons_firms[i].distances_mig[0])
    wage_distance.append(cons_firms[i].distances_mig[1])
    delta_pr.append(cons_firms[i].distances_mig[2])
    

'''
Band pass filter
'''

gdp = af.mean_variable_log(macro_variable, 'GDP total', 50)
inv = af.mean_variable_log(macro_variable, 'INV Total', 50)
cons = af.mean_variable_log(macro_variable, 'CONS Total', 50)

bp = pd.concat([gdp, inv, cons], axis = 1)
bp.columns = ['GDP', 'INV', 'CONS']
cycles = sm.tsa.filters.bkfilter(bp[['INV','CONS', 'GDP']], 6, 32, 12)
#macro_variable['Log GDP cons total']= np.log(macro_variable['GDP cons total'])
#log_mv = macro_variable.filter(like = 'Log')
#log_mv = log_mv.drop(log_mv.index[:100])#
#log_mv = log_mv.drop(log_mv.index[200:])



#cf_cycles, cf_trend = sm.tsa.filters.cffilter(log_mv[['Log INV','Log CONS', 'Log GDP cons total']])
std_bp_gdp = cycles['GDP_cycle'].std() #/ abs(cycles['GDP_cycle'].mean()) 
std_bp__cons =  cycles['CONS_cycle'].std()# / abs(cycles['CONS_cycle'].mean())
std_bp_inv =  cycles['INV_cycle'].std()# / abs(cycles['INV_cycle'].mean())

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
cycles.plot(ax=ax, style=['r--','b-'])
print(cf_cycles.head(10))
#macro_variable['Households region 0 rate'] = macro_variable['Households region 0'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['Households region 1 rate'] = macro_variable['Households region 1'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['INV region 1 rate'] = macro_variable['INV 1'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['INV region 0 rate'] = macro_variable['INV 0'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['GDP region 0 rate'] = macro_variable['GDP region 0'] / (macro_variable['GDP region 0'] + macro_variable['GDP region 1'])
fig, ax = plt.subplots()
cycles.plot(ax=ax, style=['r--', 'b-'], label = 'INV')
plt.show()


'''
PLOT SIZE DISTRIBUTION
'''

agents = model.firms1
demand_ditance_list = []

for i in range(len(agents)):
    if agents[i].region == 1:
        demand_ditance_list.append(agents[i])
    

households = model.households
for i in range(len(households)):
    demand_ditance_list.append(households[i].migration_pr)
    
    
'''
CORRELATION STRUCTURE
'''


drop= 50
drop_end = 0
#df = df_t_001
df = macro_variable
output = af.mean_variable_log(df, 'GDP total', drop, drop_end)
cons = af.mean_variable_log(df, 'CONS Total', drop, drop_end )
inv =  af.mean_variable_log(df, 'INV Total', drop, drop_end)
prices = af.mean_variable_log(df, 'Cons price region 1', drop, drop_end)
unemployment = af.mean_variable(df, 'Unemployment rate', drop, drop_end)

corr_list  = [ unemployment,  prices , inv , cons ,output ]
df_corr = pd.concat(corr_list, axis = 1)
df_corr.columns = [ 'Unemployment','Price',  'Inv',  'Cons','Output']
cycles = sm.tsa.filters.bkfilter(df_corr[['Output', 'Cons', 'Inv', 'Price', 'Unemployment']], 6, 32, 12)
cycles.columns = [    'Output' , 'Cons', 'Inv', 'Price','Unemployment']
#corrMatrix = df_corr.corr(method='pearson')
corrMatrix = cycles.corr(method='pearson')
sns.heatmap(corrMatrix, annot=True)
plt.show()
fig, ax = plt.subplots(figsize=(6, 6)) 
mask = np.zeros_like(corrMatrix.corr())
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corrMatrix, mask= mask, ax= ax, annot= True)


firm_list = model.firms1


#price_time_firm_list = model.list_firms[200]
price_ms_list = []
price_ms_list_cap = []
demand_cap = []
orders_cons = []
for i in range(len(firm_list)):
    agent = firm_list[i]
   # if agent.type == 'Cap' and agent.region ==1:
    #    price_ms_list_cap.append([agent, agent.price, agent.net_worth])
        #if sum(agent.real_demand_cap) != 0: 
           # demand_cap.append( [agent, agent.real_demand_cap, agent.unique_id])
    if agent.type == 'Cap': # and agent.region == 1:
        price_ms_list.append([agent, agent.price, agent.productivity])
        #if agent.quantity_ordered != 0 and agent.supplier_id == 24:
           # orders_cons.append([agent, agent.quantity_ordered, agent.supplier_id])
            
        
        
    
    #price_ms_list.append([price_time_firm_list[i].price, price_time_firm_list[i].market_share, price_time_firm_list[i].wage, price_time_firm_list[i].lifecycle, price_time_firm_list[i].region, price_time_firm_list[i].type])