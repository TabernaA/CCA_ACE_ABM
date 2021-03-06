# model/app.py
# bringing all elements of the model together
# contains run method

#from model.classes.capital_good_firm import CapitalGoodFirm
#from model.classes.consumption_good_firm import ConsumptionGoodFirm
#from model.classes.household import Household
from model.classes.model import KSModel
#from model.modules.data_collection import *
#from model.modules.data_collection_2 import *
import matplotlib.pyplot as plt
#import model.modules.data_analysis as da
import model.modules.additional_functions as af
#import random
import seaborn as sns
import numpy as np
import pandas as pd   




runs= 10
steps = 500
macro_variables = []
micro_variables = []
model_lists = []
for i in range(runs):
    
    model = KSModel(F1 = 50, F2 =250, H = 3500, B= 1,  T= 2, S = 5)
    model.reset_randomizer(i )
    print("#-------------- iteration", i+1, "---------------#")
    for j in range(steps):
        #print("#------------ step", j+1, "------------#")
        model.step()
    run_data = model.datacollector.get_model_vars_dataframe()
    micro_data = model.datacollector.get_agent_vars_dataframe()
    micro_data = micro_data.dropna()
    #model_lists.append(model)
    '''
    ### PLOT ##
    af.plot_list(run_data.Unemployment_Regional, range(20 ,steps), "Unemployment rate")

    af.plot_list(run_data.GDP, range(steps), "GDP")
    af.plot_list(run_data.INVESTMENT, range(steps), "Investment")
    af.plot_list(run_data.Competitiveness_Regional, range( 5,steps), "Competitiveness")
    af.plot_list(run_data.Aggregate_Employment, range(steps), "Aggregate Employment")
    #plot_list(macro_variable.Population_Regional, range(steps), "Population")
    af.plot_list(run_data.Average_Salary, range(steps) , "Average Salary")
    af.plot_list(run_data.Population_Regional_Households, range(steps), "Number of households")
    af.plot_list(run_data.Cosumption_price_average,  range( 20, steps) , "Consumption price average")
    af.plot_list(run_data.Population_Regional_Cons_Firms, range(steps), "Number of consumption firms")
    af.plot_list(run_data.Capital_firms_av_prod, range(steps), " Average productivity Cap firms")
    af.plot_list(run_data.Population_Regional_Cap_Firms, range(steps), "Number of capital  firms")
    af.plot_list(run_data.Consumption_firms_av_prod, range(steps), " Average productivity Cons firms")
    '''
    ## TRANSOFOR MICRO ##
    micro_data[['Prod 0','Prod 1']] = pd.DataFrame(micro_data.Prod.to_list(), index= micro_data.index)
   # micro_data[['MS 0','MS 1', 'MS Exp']] = pd.DataFrame(micro_data.Ms.to_list(), index= micro_data.index)
    micro_variables.append(micro_data)
    
    ## TRANSFORM MACRO##
    
    macro_variable = run_data
    
    macro_variable[['Av wage region 0','Av wage region 1', 'Wage diff 0', 'Wage diff 1']] = pd.DataFrame(macro_variable.Average_Salary.to_list(), index= macro_variable.index)
    macro_variable[['Cons region 0','Cons region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cons_Firms.to_list(), index= macro_variable.index)
    macro_variable[['Cap region 0','Cap region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cap_Firms.to_list(), index= macro_variable.index) 
    macro_variable[['Households region 0','Households region 1']] = pd.DataFrame(macro_variable.Population_Regional_Households.to_list(), index= macro_variable.index)
    macro_variable[['Cons price region 0','Cons price region 1']] = pd.DataFrame(macro_variable.Cosumption_price_average.to_list(), index= macro_variable.index)
#macro_variable[['CCA coeff 0','CCA coeff 1']] = pd.DataFrame(macro_variable.Average_CCA_coeff.to_list(), index= macro_variable.index)
   # macro_variable[['Prod cons region 0','Prod cons region 1', 'Delta prod cons region 0', 'Delta prod cons region 1']] = pd.DataFrame(macro_variable.Consumption_firms_av_prod.to_list(), index= macro_variable.index)
    macro_variable[['Prod region 0','Prod region 1', 'Delta prod region 0', 'Delta prod region 1' ]] = pd.DataFrame(macro_variable.Regional_average_productivity.to_list(), index= macro_variable.index)    
    macro_variable[['GDP region 0','GDP region 1', 'GDP total']] = pd.DataFrame(macro_variable.GDP.to_list(), index= macro_variable.index)
    macro_variable[['Unemployment region 0','Unemployment region 1', 'Unemployment diff 0','Unemployment diff 1','Unemployment total' ]] = pd.DataFrame(macro_variable.Unemployment_Regional.to_list(), index= macro_variable.index)
#macro_variable[['MS track 0', 'MS track 1']] = pd.DataFrame(mac 7V'CONS difference 1']] = pd.DataFrame(macro_variable.CONSUMPTION.to_list(), index= macro_variable.index)
    macro_variable[['CONS 0', 'CONS 1', 'CONS Total', 'Export']] = pd.DataFrame(macro_variable.CONSUMPTION.to_list(), index= macro_variable.index)
    macro_variable[['INV 0', 'INV 1', 'INV Total']] = pd.DataFrame(macro_variable.INVESTMENT.to_list(), index= macro_variable.index)
    macro_variable[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame(macro_variable.GDP_cons.to_list(), index= macro_variable.index)
    macro_variable_csv_data = macro_variable.to_csv('data_model_.csv', index  = True)
    macro_variable[['Aggr unemployment region 0','Aggr unemployment region 1']] = pd.DataFrame(macro_variable.Aggregate_Unemployment.to_list(), index= macro_variable.index)
    macro_variable['Aggr unemployment'] = macro_variable['Aggr unemployment region 0'] + macro_variable['Aggr unemployment region 1']
    macro_variable[['Aggr employment region 0','Aggr employment region 1']] = pd.DataFrame(macro_variable.Aggregate_Employment.to_list(), index= macro_variable.index)
    macro_variable['Aggr employment'] = macro_variable['Aggr employment region 0'] + macro_variable['Aggr employment region 1']
    macro_variable['Unemployment rate'] = macro_variable['Aggr unemployment'] / (macro_variable['Aggr unemployment'] + macro_variable['Aggr employment'])
    macro_variable[['Sum MS region 0','Sum MS region 1', 'Sum MS exp', 'Exp share region 0', 'Exp share region 1']] = pd.DataFrame(macro_variable.Regional_sum_market_share.to_list(), index= macro_variable.index)
    macro_variable['Wage prod ratio 0']= macro_variable['Av wage region 0']/ macro_variable['Prod region 0']
    macro_variables.append(macro_variable)
#df_multiple_runs = pd.DataFrame(macro_variables) 




result_1 = pd.concat(macro_variables, axis = 1 , copy = False)
micro_result_1 = pd.concat(micro_variables, axis = 0)


bankruptcy =  micro_result_1.loc[(micro_result_1['Bankrupt'] > 0)]
mv =  bankruptcy.loc[(bankruptcy.index.get_level_values('Step') > 150) & (bankruptcy.index.get_level_values('Step') < 170)]
mv['Size'] = mv['Bankrupt']
mv['Firms'] = 1
a = mv.groupby(by="Size").count()
total = 2918
a['Firms'] = a['Firms'] / total

a =   a.loc[(a['Firms'] >= 0.01)]
a.rename(columns = {'Firms':'% over the total of post-disaster bankruptcies'}, inplace = True)
a.rename(columns = {'Size':'Size of the firm'}, inplace = True)
sns.barplot(x = a.index,  y = '% over the total of post-disaster bankruptcies',   color = 'cyan' , data = a)

transition = 15
#result_1.to_csv(r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\T002_S01_100RUNSb.csv')         
inv_gr_0 = af.variable_growth_rate(result_1, 'INV 0', transition, steps)
inv_gr_1 = af.variable_growth_rate(result_1, 'INV 1', transition, steps)
#steps = 150



bankruptcy =  micro_result_1.loc[(micro_result_1['Bankrupt'] > 0)]
bankruptcy = bankruptcy.loc[140:, ]bankruptcy

sns.distplot(mv["Bankrupt"], bins=4)
mv = bankruptcy.loc[(bankruptcy.index.get_level_values('Step') > 150) & (bankruptcy.index.get_level_values('Step') < 170)] 
mv.drop(mv['AgentID'])
mv.loc[ mv.index.get_level_values(0) > 150, 'Step'] = 1

sns.barplot(x ='Year post flood', y = "", data = mv)
plt.show()
mv['Bankrupt av'] = mv['Bankrupt'] / 5 
mv['Bankrupt av'].hist()
mv= mv.loc[(mv.index.get_level_values('Step') < 180)]
mv['Year post flood'] = (mv.index.get_level_values('Step') - 150) // 4
mv['Firms'] = 1 

a = mv.groupby(by="Year post flood").sum()
sns.barplot(x = a.index.get_level_values('Year post flood') , y = "Firms", hue = 'Bankrupt', data =  a)
plt.show()



'''
GDP 
'''



gdp_gr = af.variable_growth_rate(result_1, 'GDP cons total', transition, steps)
gdp_gr_0 = af.variable_growth_rate(result_1, 'GDP cons region 0', transition, steps)
gdo_gr_1 = af.variable_growth_rate(result_1, 'GDP cons region 1', transition, steps)

'''
df_gdp_pct = result_1.filter(like = 'GDP total').pct_change()
df_gdp = result_1.filter(like = 'GDP total')

df_gdp_aggl_preflood = df_gdp.iloc[:, [0,1,2,3,4,7,8,10,11,13,14]]
last_raw = df_gdp_aggl_preflood.iloc[300, :]
'''

mean_gdp_0 = af.mean_variable_log(result_1, 'GDP region 0')
mean_gdp_1 = af.mean_variable_log(result_1, 'GDP region 1')

gdp_plot = [[mean_gdp_0, 'Coastal region', 'blue'], [mean_gdp_1, 'Inland region', 'green']]
af.plot(gdp_plot, 50, 250)


'''
CONS
'''
cons_gr = af.variable_growth_rate(result_1, 'CONS Total', transition, steps)
cons_gr_0 = af.variable_growth_rate(result_1, 'CONS 0', transition, steps)
cons_gr_1 = af.variable_growth_rate(result_1, 'CONS 1', transition, steps)

'''
UNEMPLOYMENT RATE
'''

unemployment_rate_0 = af.mean_variable(result_1,'Unemployment region 0', 100).mean()
unemployment_rate_1 = af.mean_variable(result_1,'Unemployment region 1', 100).mean()
unemployment_rate = af.mean_variable(result_1,'Unemployment rate', 100).mean()

af.mean_variable(result_1,'Unemployment region 0', 100).plot()
af.mean_variable(result_1,'Unemployment region 1', 100).plot()
af.mean_variable(result_1,'Unemployment rate', 100).plot()


'''
CONSUMPTION FIRMS 
'''

firms_gr_0 = af.variable_growth_rate(result_1, 'Cons region 0', transition, steps)
firms_gr_1 = af.variable_growth_rate(result_1, 'Cons region 1', transition, steps)

mean_cons_0 = af.mean_variable(result_1, 'Cons region 0')
mean_cons_1 = af.mean_variable(result_1, 'Cons region 1')

df = result_1.filter(like = 'Cons region 0')
df.loc[ : ,  df.loc[40, :] == 0 ]

firms_plot = [[mean_cons_0, 'Coastal region', 'blue'], [mean_cons_1, 'Inland region', 'green']]
af.plot(firms_plot, 0, 350)

'''
HOUSEHOLDS
'''

h_gr_0 = af.variable_growth_rate(result_1, 'Households region 0', transition, steps)
h_gr_1 = af.variable_growth_rate(result_1, 'Households region 1', transition, steps)

mean_hous_0 = af.mean_variable(result_1, 'Households region 0')
mean_hous_1 = af.mean_variable(result_1, 'Households region 1')

households_plot = [[mean_hous_0, 'Coastal region', 'blue'], [mean_hous_1, 'Inland region', 'green']]
af.plot(households_plot, 0, 350)


'''
PRICE
'''


##--Price --##
# I
price_growth_rate = af.variable_growth_rate(result_1 ,'Cons price region 0', transition, steps)
mean_price_0 = af.mean_variable_log(result_1, 'Cons price region 0').plot()
mean_price_1 = af.mean_variable_log(result_1, 'Cons price region 1').plot()
price_plot = [[mean_price_0, 'Region 0', 'black'], [mean_hous_1, 'Region 1', 'red']]
af.plot(households_plot, 0, 350)

#df_unemployment_0 = result_1.filter(like = 'Unemployment region 0')
'''
unemployment_mean = df_unemployment_total['Mean Unemployment rate'].mean()
unemployment_mean_0 = df_unemployment_0['Mean Unemployment region 0'].mean()
unemployment_mean_1 = df_unemployment_1['Mean Unemployment region 1'].mean()
'''
'''
BANDPASS FILTER
'''
import statsmodels.api as sm

#result_1 = macro_variables[50]

gdp = af.mean_variable_log(result_1, 'GDP total',100)
inv = af.mean_variable_log(result_1, 'INV Total', 100)
cons = af.mean_variable_log(result_1, 'CONS Total', 100)

bp = pd.concat([gdp, inv, cons], axis = 1)
bp.columns = ['GDP', 'INV', 'CONS']
cycles = sm.tsa.filters.bkfilter(bp[['INV','CONS', 'GDP']], 6, 32, 12)

#cycles = cycles.drop(cycles.index[200:])
cycles['INV_cycle'].std()
cycles['CONS_cycle'].std()
cycles['GDP_cycle'].std()


#cycles['CONS_cycle'].std()
rsd_gdp = cycles['GDP_cycle'].std() / abs(cycles['GDP_cycle'].mean()) 
rsd_cons =  cycles['CONS_cycle'].std() / abs(cycles['CONS_cycle'].mean())
rsd_inv =  cycles['INV_cycle'].std() / abs(cycles['INV_cycle'].mean())
cf_cycles, cf_trend = sm.tsa.filters.cffilter(bp[['INV','CONS', 'GDP']])

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
cf_cycles.plot(ax=ax, style=['r--','b-'])
print(cf_cycles.head(10))
#macro_variable['Households region 0 rate'] = macro_variable['Households region 0'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['Households region 1 rate'] = macro_variable['Households region 1'] / (macro_variable['Households region 0'] + macro_variable['Households region 1'])
#macro_variable['INV region 1 rate'] = macro_variable['INV 1'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['INV region 0 rate'] = macro_variable['INV 0'] / (macro_variable['INV 0'] + macro_variable['INV 1'])
#macro_variable['GDP region 0 rate'] = macro_variable['GDP region 0'] / (macro_variable['GDP region 0'] + macro_variable['GDP region 1'])

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
cycles.plot(ax=ax, style=['r--', 'b-'])
plt.show()


#df_2  = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\Migration_beginning\model\multiple_runs_flood_hor.csv')
#result_1 = df_2

'''
FLOOD ANALYSIS 
'''


'''FIRMS'''



## ---FLOOD --- ##
df_s_0a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S0_T002_500RUNS.csv', skiprows=2)
#df_s_0 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S0_T002_250RUNS_500steps.csv', skiprows=2)

df_s_0b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S00_T002_100RUNS.csv', skiprows=2)
df_s_050a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S05_T002_500RUNS.csv', skiprows=2)
#df_s_050 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S05_T002_500RUNS_500steps.csv', skiprows=2)
df_s_050b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S05_T002_100RUNS.csv', skiprows=2)
df_s_050c = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S05_T002_100RUNS.csv', skiprows=2)
#df_s_025a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\30runs_T003_S025.csv')
#df_s_025b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\10runs_T003_S025.csv')
df_s_010a =  pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\T002_S01_100RUNS.csv')
df_s_010b =  pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\T002_S01_100RUNSb.csv')
#df_s_025b.merge(df_s_025a)  
df_s_030a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S03_T002_500RUNS.csv', skiprows=2)
df_s_030b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\results\S03_T002_100RUNS.csv', skiprows=2)
df_s_0 = pd.concat([df_s_0a, df_s_0b], axis = 1)
df_s_010 = pd.concat([df_s_010a, df_s_010b], axis = 1)
df_s_050 = pd.concat([df_s_050a, df_s_050b, df_s_050c, df_s_050c, df_s_050c, df_s_050c], axis = 1)
df_s_030 = pd.concat([df_s_030a, df_s_030b], axis = 1)

f_all_050 = df_s_050.filter(like = "Population_Region_0_Cons_Firms")

time_intervals = [150, 250]
raws_50 = []
count_values = []
for i in time_intervals:
    raws_50.append(f_all_050.iloc[i, :])
    count_values.append(f_all_050.iloc[i, :].value_counts().sort_index(ascending = False)/ 1000)


count_f_50 = pd.concat(count_values, axis = 1)
count_f_50.columns = ['Before flood', 'After flood']

count_f_50['Net after flood'] = count_f_50['After flood'] -  count_f_50['Before flood'] 
#agl_1 = count_f_50.loc[:124,:]
agl_1 = count_f_50.loc[:124, :].sum()
agl_0 = count_f_50.loc[125:, :].sum()

f_all_0 = df_s_0.filter(like = "Population_Region_0_Cons_Firms")
raws_0 = []
count_values_0 = []
for i in time_intervals:
    raws_0.append(f_all_0.iloc[i, :])
    count_values_0.append(f_all_0.iloc[i, :].value_counts().sort_index(ascending = False)/ 600)
    
count_f_0 = pd.concat(count_values_0, axis = 1)
agl_1_0 = count_f_0.loc[:124, :].sum()
agl_0_0 = count_f_0.loc[125:, :].sum()


f_all_0 = df_s_0.filter(like = "Population_Region_0_Cons_Firms")

time_intervals = [150, 250]
raws_0 = []
count_values = []
for i in time_intervals:
    raws_0.append(f_all_0.iloc[i, :])
    count_values.append(f_all_0.iloc[i, :].value_counts().sort_index(ascending = False)/ 600)


count_f_0 = pd.concat(count_values, axis = 1)
count_f_0.columns = ['Before flood', 'After flood']
count_f_0['Net after flood'] = count_f_0['After flood'] -  count_f_0['Before flood'] 

agl_1_before = 
agl_0 = count_f_50.iloc[104:, :].sum()

f_all_0 = df_s_0.filter(like = "Population_Region_0_Cons_Firms")
raws_0 = []
count_values_0 = []
for i in time_intervals:
    raws_0.append(f_all_0.iloc[i, :])
    count_values_0.append(f_all_0.iloc[i, :].value_counts().sort_index(ascending = False)/ 600)
    
count_f_0 = pd.concat(count_values_0, axis = 1)
agl_0_1 = count_f_0.iloc[:1, :].sum()
agl_0_0 = count_f_0.iloc[90:, :].sum()




ast_raw.value_counts()


drop_firm = 0

firms_0_s050 =  af.mean_variable(df_s_050, 'Population_Region_0_Cons_Firms', drop_firm) /250
firms_0_s050_std =  af.std_dev_variable(df_s_050, 'Population_Region_0_Cons_Firms', drop_firm)  / 250  / np.sqrt(800)# / 250 #* 2
#firms_1_s050 =  af.mean_variable(df_s_050, 'Cons region 1', drop_firm) * 2
#firms_0_s025a =  mean_variable(df_s_025a, 'Cons region 0', drop_firm)
#firms_1_s025a =  mean_variable(df_s_025a, 'Cons region 1', drop_firm)
firms_0_s030 =  af.mean_variable(df_s_030 , 'Population_Region_0_Cons_Firms', drop_firm) / 250 
firms_0_s030_std =  af.std_dev_variable(df_s_030, 'Population_Region_0_Cons_Firms', drop_firm)  / 250  / np.sqrt(600) #* 2
#firms_1_s025 =  af.mean_variable(df_s_025, 'Cons region 1', drop_firm) * 2

#firms_0_s025 = ( firms_0_s025a + firms_0_s025b)/ 2
#firms_1_s025 = ( firms_1_s025a + firms_1_s025b)/ 2
firms_0_s0 =  af.mean_variable(df_s_0, "Population_Region_0_Cons_Firms", drop_firm) / 250
firms_0_s0_std =  af.std_dev_variable(df_s_0, 'Population_Region_0_Cons_Firms', drop_firm)  / 250  / np.sqrt(600)
firms_0_s010 =  af.mean_variable(df_s_010, 'Cons region 0', drop_firm) / 250
#firms_1_s010 =  mean_variable(df_s_010, 'Cons region 1', drop_firm) * 2


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])



#ax.plot( firms_0_t001,  label = 'F = 0', color = 'black')	
#ax.plot( firms_1_t001, '--', color = 'black', )	
ax.plot( firms_0_s050,  label = 'Dc = 0.5', color = 'black')	
ax.fill_between( firms_0_s050.index, firms_0_s050 - firms_0_s050_std, firms_0_s050 + firms_0_s050_std, color = 'black', alpha = .1 )
#ax.plot( firms_1_s050, label = 'F = 0.5', color = 'red', )	
ax.plot(  firms_0_s030,  label = 'Dc = 0.3', color = 'purple')	
ax.fill_between( firms_0_s030.index, firms_0_s030 - firms_0_s030_std, firms_0_s030 + firms_0_s030_std , color = 'purple', alpha = .1)
#ax.plot(  firms_1_s025,'--', label = 'F = 0.25',  color = 'red', )	
ax.plot(  firms_0_s0, label = 'Dc = 0', color = 'gold')
ax.fill_between( firms_0_s0.index, firms_0_s0 - firms_0_s0_std, firms_0_s0 + firms_0_s0_std , color = 'gold', alpha = .1)	
#ax.plot(  firms_0_s010, 'o', label = 'F = 0.1', color = 'grey', )	

ax.set_xlabel("Time step", fontsize = 16)	
ax.set_ylabel("% Firms population in Coastal region", fontsize = 16)	

plt.legend()
plt.show()

firms_0_s0 = firms_0_s0.rename('Firms population')


fig, axes = plt.subplots(1, 1, figsize=(15, 5), sharey=True)
fig.suptitle('Firms population')
sns.lineplot(x=  firms_0_s0.index , y = firms_0_s0 , data =  firms_0_s0)



data = {'Agglomeration':[0.77, 0.75, 0.70, 0.18, 0.20, 0.23],
        'Damage coeff' :[0, 0.3, 0.5, 0, 0.3, 0.5],
        'Region': ['Coastal','Coastal', 'Coastal' , 'Internal', 'Internal', 'Internal']}

df = pd.DataFrame(data)
sns.barplot(x = "Region", y = "Agglomeration", hue = "Damage coeff", data = df)
plt.show()

'''Households'''
drop_h = 0

drop_firm = 0

households_0_s050 =  af.mean_variable(df_s_050, "Population_Region_0_Households", drop_firm) /250
households_0_s050_std =  af.std_dev_variable(df_s_050, "Population_Region_0_Households" , drop_firm)  / 250  / np.sqrt(800)# / 250 #* 2
#firms_1_s050 =  af.mean_variable(df_s_050, 'Cons region 1', drop_firm) * 2
#firms_0_s025a =  mean_variable(df_s_025a, 'Cons region 0', drop_firm)
#firms_1_s025a =  mean_variable(df_s_025a, 'Cons region 1', drop_firm)
households_0_s030 =  af.mean_variable(df_s_030 , "Population_Region_0_Households", drop_firm) / 250 
households_0_s030_std =  af.std_dev_variable(df_s_030, "Population_Region_0_Households", drop_firm)  / 250  / np.sqrt(600) #* 2
#firms_1_s025 =  af.mean_variable(df_s_025, 'Cons region 1', drop_firm) * 2

#firms_0_s025 = ( firms_0_s025a + firms_0_s025b)/ 2
#firms_1_s025 = ( firms_1_s025a + firms_1_s025b)/ 2
households_0_s0 =  af.mean_variable(df_s_0, "Population_Region_0_Households", drop_firm) / 250
households_0_s0_std =  af.std_dev_variable(df_s_0, "Population_Region_0_Households", drop_firm)  / 250  / np.sqrt(600)
households_0_s010 =  af.mean_variable(df_s_010, "Population_Region_0_Households", drop_firm) / 250
#firms_1_s010 =  mean_variable(df_s_010, 'Cons region 1', drop_firm) * 2


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])



#ax.plot( firms_0_t001,  label = 'F = 0', color = 'black')	
#ax.plot( firms_1_t001, '--', color = 'black', )	
ax.plot( households_0_s050,  label = 'Dc = 0.5', color = 'black')	
ax.fill_between( households_0_s050.index, households_0_s050 - households_0_s050_std, households_0_s050 + households_0_s050_std, color = 'black', alpha = .1 )
#ax.plot( firms_1_s050, label = 'F = 0.5', color = 'red', )	
ax.plot(  households_0_s030,  label = 'Dc = 0.3', color = 'purple')	
ax.fill_between( households_0_s030.index, households_0_s030 - households_0_s030_std, households_0_s030 + households_0_s030_std , color = 'purple', alpha = .1)
#ax.plot(  firms_1_s025,'--', label = 'F = 0.25',  color = 'red', )	
ax.plot(  households_0_s0, label = 'Dc = 0', color = 'gold')
ax.fill_between( households_0_s0.index, households_0_s0 - households_0_s0_std, households_0_s0 + households_0_s0_std , color = 'gold', alpha = .1)	
#ax.plot(  firms_0_s010, 'o', label = 'F = 0.1', color = 'grey', )	

ax.set_xlabel("Time step", fontsize = 16)	
ax.set_ylabel("% Households population in Coastal region", fontsize = 16)	

plt.legend()
plt.show()



'''
c = [households_0, households_0_flood, households_1, households_1_flood]
c_1 = [households_0, households_1]
df_households_concat = pd.concat(c, axis = 1)
df_households_concat.columns = ['Coastal no-flood', 'Coastal flood', 'Inland no-flood', 'Inland flood'] 
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
df_households_concat.plot(ax=ax, style='r--')


#rank_data, size_data = rank_size_data(pd_sales['Sales'])


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( households_0,  label = 'T = 0.01', color = 'green')	
ax.plot( households_1  , color = 'green')	
ax.plot( firms_1,  label = 'T = 02' , color = 'blue')	
ax.set_xlabel("log size")	
ax.set_ylabel("log rank")	

plt.legend()
plt.show()
'''


'''GDP'''

drop_gdp = 120
drop_end = 70
   

gdp_all_0 = df_s_0a.filter(like = "GDP_cons")
gdp_all_0.iloc[:,0][['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame(gdp_all_0.iloc[:,0].to_list(), index= gdp_all_0.index)

gdp_all_0[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame( gdp_all_0.GDP_cons.to_list(), index= gdp_all_0.index)

GDP_all_0[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame(GDP_all_0.GDP_cons.to_list(), index= GDP_all_0.index)
 macro_variable[['Av wage region 0','Av wage region 1', 'Wage diff 0', 'Wage diff 1']] = pd.DataFrame(macro_variable.Average_Salary.to_list(), index= macro_variable.index)



GDP_all_0[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = pd.DataFrame(GDP_all_0.GDP_cons.to_list(), index= GDP_all_0.index)


gdp_all_0[0]    
for column in gdp_all_0.columns:
        print([f"{column}"])
        gdp_all_0[[f"{column}"]] = gdp_all_0[f"{column}"].astype(int)
        df[[f"{column}"] = pd.to_numeric(df[f"{column}"]) 
        df1 = pd.DataFrame( gdp_all_0[f"{column}"].to_list(), columns=['1', '2', '3'])
        
        #gdp_all_0[['GDP cons region 0','GDP cons region 1', 'GDP cons total']] = , index= gdp_all_0.index)
 df['DataFrame Column'] = df['DataFrame Column'].astype(int)
(2) The to_numeric method:

df['DataFrame Column'] = pd.to_numeric(df['DataFrame Column'])       




gdp_s050 =  af.mean_variable_log(df_s_050, 'GDP total', drop_gdp, drop_end)
gdp_0_s050 =  af.mean_variable_log(df_s_050, 'GDP region 0', drop_gdp,drop_end)
gdp_1_s050 =  af.mean_variable_log(df_s_050, 'GDP region 1', drop_gdp, drop_end)
gdp_s025 =  af.mean_variable_log(df_s_025, 'GDP total', drop_gdp, drop_end)
gdp_0_s025 =  af.mean_variable_log(df_s_025, 'GDP region 0', drop_gdp, drop_end)
gdp_1_s025 =  af.mean_variable_log(df_s_025, 'GDP region 1', drop_gdp, drop_end)
gdp_s010 =  mean_variable_log(df_s_010, 'GDP total', drop_gdp, drop_end)
gdp_0_s010 =  af.mean_variable_log(df_s_010, 'GDP region 0', drop_gdp, drop_end)
gdp_1_s010 =  af.mean_variable_log(df_s_010, 'GDP region 1', drop_gdp, drop_end)

## GDP TOTAL PLOT ##
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( gdp_s050,  label = 'F = 0.5', color = 'black')	
ax.plot(  gdp_s025, label = 'F = 0.25', color = 'red')	
#ax.plot( gdp_s010, label = 'F = 0.1', color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()

## GDP REGION 0 ##

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( gdp_0_s050,  label = 'F = 0.5', color = 'green')	
ax.plot(  gdp_0_s025, label = 'F = 0.25', color = 'blue')	
ax.plot( gdp_0_s010, label = 'F = 0.1', color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP REGION 0')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()


## GDP REGION 1 ##

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( gdp_1_s050,  label = 'F = 0.5', color = 'green')	
ax.plot(  gdp_1_s025, label = 'F = 0.25', color = 'blue')	
ax.plot( gdp_1_s010, label = 'F = 0.1', color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP REGION 1')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()




## GDP REGION 0  and 1 ##

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( gdp_0_s050,  label = 'F = 0.5', color = 'black')	
ax.plot( gdp_1_s050, color = 'red')	
ax.plot( gdp_0_s025, '--', label = 'F = 0.25', color = 'black')	
ax.plot( gdp_1_s025 , '--' ,  color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP REGION 0 and REGION 1')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()




fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
ax.plot( gdp_0_s025, label = 'Region 0', color = 'green')	
ax.plot( gdp_1_s025 , label = 'Region 1',  color = 'blue')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP REGION 0 and REGION 1')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
ax.plot( gdp_0_s010, label = 'Region 0', color = 'green')	
ax.plot( gdp_1_s010 , label = 'Region 1',  color = 'blue')	
#plt.xticks(np.arange(0,50,1))
ax.set_xlabel("Time step")	
ax.set_ylabel('GDP REGION 0 and REGION 1')	
#plt.xticks(range(1, 50))

plt.legend()
plt.show()




'''
gdp_0_list = [gdp_0, gdp_flood_0]
df_gdp_0_concat = pd.concat(gdp_0_list, axis = 1)
df_gdp_0_concat.columns = ['GDP Coastal flood', 'GDP Coastal no-flood'] 
#df_households_concat.columns = ['Coastal no-flood', 'Coastal flood', 'Internal no-flood', 'Internal flood'] 
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
gdp_0.plot(ax=ax, style=['r--','b-'])

gdp_1_list = [gdp_1, gdp_flood_1]
df_gdp_1_concat = pd.concat(gdp_1_list, axis = 1)
df_gdp_1_concat.columns = ['GDP region 0 flood', 'GDP no-flood'] 
#df_households_concat.columns = ['Coastal no-flood', 'Coastal flood', 'Internal no-flood', 'Internal flood'] 
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
gdp_1.plot(ax=ax, style=['r--','b-'])
'''

'''
Price analysis 
'''

price_drop = 120
price_end = 70

price_0_s050 = af.mean_variable_log(df_s_050 , 'Cons price region 0', price_drop, price_end)
price_1_s050 = af.mean_variable_log(df_s_050 , 'Cons price region 1', price_drop, price_end)
price_0_s025 = af.mean_variable_log(df_s_025 , 'Cons price region 0', price_drop, price_end)
price_1_s025 = af.mean_variable_log(df_s_025 , 'Cons price region 1', price_drop, price_end)
#price_s010 = mean_variable_log(df_s_010 , 'Cons price region 0', price_drop, price_end)

'''
price_s050 = mean_variable_log(df_s_050 , 'Cons price region 1', price_drop, price_end)
price_s025 = mean_variable_log(df_s_025 , 'Cons price region 1', price_drop, price_end)
price_s010 = mean_variable_log(df_s_010 , 'Cons price region 1', price_drop, price_end)
'''

## PRICE REGION 1  and 0##

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(211)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( price_0_s050,  label = 'Region 0', color = 'black')
ax.plot( price_1_s050,  label = 'Region 1', color = 'red')	
#ax.set_xlabel("Time step")	
ax.set_ylabel('Price')
ax.set_title('F = 0.5')	
ax.legend()
ax1 = fig.add_subplot(212)
ax1.plot(  price_0_s025, label = 'Region 0', color = 'black')	
ax1.plot(  price_1_s025, label = 'Region 1', color = 'red')	
#ax.plot( price_s010, label = 'F = 0.1', color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax1.set_xlabel("Time step")	
ax1.set_ylabel('Price ')	
ax1.set_title('F = 0.25')	
#ax1.title('Region 1')
#plt.xticks(range(1, 50))

plt.legend()
plt.show()


price_list = [price_flood, price_no_flood]
df_price_concat = pd.concat(price_list, axis = 1)
df_price_concat.columns = ['Price flood', 'Price NO flood'] 
#df_households_concat.columns = ['Coastal no-flood', 'Coastal flood', 'Internal no-flood', 'Internal flood'] 
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
price_cons_0.plot(ax=ax, style=['r--','b-'])



'''
Unemployment rate analysis 
'''
unemployment_drop = 120
unemployment_end = 70

unem_0_s050 = af.mean_variable(df_s_050 , 'Unemployment region 0', unemployment_drop, unemployment_end)
unem_1_s050 = af.mean_variable(df_s_050 , 'Unemployment region 1', unemployment_drop, unemployment_end)
unem_0_s025 = af.mean_variable(df_s_025 , 'Unemployment region 0', unemployment_drop, unemployment_end)
unem_1_s025 = af.mean_variable(df_s_025 , 'Unemployment region 1', unemployment_drop, unemployment_end)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(211)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( unem_0_s050,  label = 'Region 0', color = 'black')
ax.plot( unem_1_s050,  label = 'Region 1', color = 'red')	
#ax.set_xlabel("Time step")	
ax.set_ylabel('Unemployment rate')
ax.set_title('F = 0.5')	
ax.legend()
ax1 = fig.add_subplot(212)
ax1.plot(  unem_0_s025, label = 'Region 0', color = 'black')	
ax1.plot(  unem_1_s025, label = 'Region 1', color = 'red')	
#ax.plot( price_s010, label = 'F = 0.1', color = 'red')	
#plt.xticks(np.arange(0,50,1))
ax1.set_xlabel("Time step")	
ax1.set_ylabel('Unemployment rate')	
ax1.set_title('F = 0.25')	
#ax1.title('Region 1')
#plt.xticks(range(1, 50))

plt.legend()
plt.show()

'''
TRANSPORT COST

'''
df_t_000_a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\50RUNS_T00_S0.csv')
df_t_000_b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\20RUNS_T00_S0.csv')
#df_t_001 = pd.read_csv(  r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\transport_cost_analysis\40runs_t001.csv')
df_t_002_a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\75RUNS_T002_S0.csv')
df_t_002_b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\50RUNS_T002_S0.csv')
df_t_005 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\50RUNS_T005_S0.csv')
#df_t_006 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\transport_cost_analysis\30run_t006.csv')
df_t_01_a = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\50RUNS_T01_S0.csv')
df_t_01_b = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\20RUNS_T01_S0.csv')
#df_t_025 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\transport_cost_analysis\40run_t025.csv')
#df_t_003 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\Migration_beginning\model\data_results\multiple_runs_S_0_T003_hor.csv')
#df_t_025 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\Migration_beginning\model\data_results\multiple_runs_T002_S025b__shuffle_hor.csv')
#df_s_010 = pd.read_csv( r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\Migration_beginning\model\data_results\multiple_runs_T002_S01a__shuffle_hor.csv')
#df_s_025b.merge(df_s_025a)  

df_t_000 = pd.concat([df_t_000_a, df_t_000_b], axis = 1)
df_t_002 = pd.concat([df_t_002_a, df_t_002_b], axis = 1)
df_t_01 = pd.concat([df_t_01_a, df_t_01_b] , axis = 1)



##---FIRMSS --##
drop_firm = 0
firms_pop = 250

firms_0_t000 =  af.mean_variable(df_t_000, 'Cons region 0', drop_firm)  
perc_firms_0_t002 =  af.mean_variable(df_t_002, 'Cons region 0', drop_firm)
perc_firms_0_t005 =  af.mean_variable(df_t_005, 'Cons region 0', drop_firm)
#perc_firms_0_t006 =  af.mean_variable(df_t_006, 'Cons region 0', drop_firm) / firms_pop
perc_firms_0_t01 =  af.mean_variable(df_t_01, 'Cons region 0', drop_firm) 
#perc_firms_0_t025 =  af.mean_variable(df_t_025, 'Cons region 0', drop_firm) / firms_pop
#firms_0_t0 =  mean_variable(df_t_000, 'Cons region 0', drop_firm) / firms_pop
#

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( perc_firms_0_t000 ,  label = 't = 0', color = 'blue')	
ax.plot( perc_firms_0_t002 ,  label = 't = 0.02', color = 'green')	
ax.plot( perc_firms_0_t005 , label = 't = 0.05', color = 'black', )	
#ax.plot( perc_firms_0_t006 , label = 't = 0.06', color = 'blue')	
ax.plot(  perc_firms_0_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( perc_firms_0_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Firms population", fontsize = 14)	

plt.legend()
plt.show()



## --- percentage firms in region 0 --##
#perc_firms_0_t001 =  af.mean_variable(df_t_001, 'Cons region 0', drop_firm) / firms_pop
#perc_firms_0_t002 =  af.mean_variable(df_t_002, 'Cons region 0', drop_firm) / firms_pop
perc_firms_0_t000 =  af.mean_variable(df_t_000_a, 'Cons region 0', drop_firm)  / firms_pop
perc_firms_0_t002 =  af.mean_variable(df_t_002, 'Cons region 0', drop_firm) / firms_pop
perc_firms_0_t005 =  af.mean_variable(df_t_005, 'Cons region 0', drop_firm) / firms_pop
#perc_firms_0_t006 =  af.mean_variable(df_t_006, 'Cons region 0', drop_firm) / firms_pop
perc_firms_0_t01 =  af.mean_variable(df_t_01, 'Cons region 0', drop_firm) / firms_pop
#perc_firms_0_t025 =  af.mean_variable(df_t_025, 'Cons region 0', drop_firm) / firms_pop
#firms_0_t0 =  mean_variable(df_t_000, 'Cons region 0', drop_firm) / firms_pop
#

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( perc_firms_0_t000 ,  label = 't = 0', color = 'blue')	
ax.plot( perc_firms_0_t002 ,  label = 't = 0.02', color = 'green')	
ax.plot( perc_firms_0_t005 , label = 't = 0.05', color = 'black', )	
#ax.plot( perc_firms_0_t006 , label = 't = 0.06', color = 'blue')	
ax.plot(  perc_firms_0_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( perc_firms_0_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel(" % of Firms population in Coastal region", fontsize = 14)	

plt.legend()
plt.show()


## -- producitvity ration between the two regions ---##
prod_ration_firms_t001 =  af.mean_variable(df_t_001, 'Prod region 0', drop_firm) / af.mean_variable(df_t_001, 'Prod region 1', drop_firm)
prod_ration_firms_t002 =  af.mean_variable(df_t_002, 'Prod region 0', drop_firm) / af.mean_variable(df_t_002, 'Prod region 1', drop_firm)
prod_ration_firms_t005 =  af.mean_variable(df_t_005, 'Prod region 0', drop_firm) / af.mean_variable(df_t_005, 'Prod region 1', drop_firm)
prod_ration_firms_t006 =  af.mean_variable(df_t_006, 'Prod region 0', drop_firm) / af.mean_variable(df_t_006, 'Prod region 1', drop_firm)
prod_ration_firms_t01 =  af.mean_variable(df_t_01, 'Prod region 0', drop_firm) / af.mean_variable(df_t_01, 'Prod region 1', drop_firm)
prod_ration_firms_t025 =  af.mean_variable(df_t_025, 'Prod region 0', drop_firm) / af.mean_variable(df_t_025, 'Prod region 1', drop_firm)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( prod_ration_firms_t001 ,  label = 't = 0.01', color = 'red')	
ax.plot( prod_ration_firms_t002 ,  label = 't = 0.02', color = 'green')	
ax.plot(prod_ration_firms_t005 , label = 't = 0.05', color = 'black', )	
ax.plot( prod_ration_firms_t006 ,label = 't = 0.06', color = 'blue')	
ax.plot(  prod_ration_firms_t01 ,label = 't - 0.1',  color = 'yellow', )	
ax.plot( prod_ration_firms_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Productivity ratio", fontsize = 14)	

plt.legend()
plt.show()








#firms_0_s025 = ( firms_0_s025a + firms_0_s025b)/ 2
#firms_1_s025 = ( firms_1_s025a + firms_1_s025b)/ 2

firms_0_t002 =  mean_variable(df_t_002, 'Cons region 0', drop_firm) / firms_pop
#firms_1_t002 =  mean_variable(df_t_002, 'Cons region 1', drop_firm) / firms_pop

#firms_0_t003 =  mean_variable(df_t_003, 'Cons region 0', drop_firm) * 2
#firms_1_t003 =  mean_variable(df_t_003, 'Cons region 1', drop_firm) * 2

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( firms_0_t0,  label = 't = 0', color = 'green')	
ax.plot( firms_1_t0, '--', color = 'green', )	
ax.plot(  firms_0_t001, label = 't = 0.01', color = 'blue')	
ax.plot(  firms_1_t001,'--',  color = 'blue', )	
ax.plot(  firms_0_t002, label = 't = 0.03', color = 'red')	
ax.plot(  firms_1_t002, '--', color = 'red', )	
ax.plot(  firms_0_t003, label = 't = 0.02', color = 'yellow')	
ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Firms population", fontsize = 14)	

plt.legend()
plt.show()



##-- HOUSEHOLDS --##
households_0_t0 =  mean_variable(df_t_000, 'Households region 0', drop_firm)
households_1_t0 =  mean_variable(df_t_000, 'Households region 1', drop_firm)
#firms_0_s025a =  mean_variable(df_s_025a, 'Cons region 0', drop_firm)
#firms_1_s025a =  mean_variable(df_s_025a, 'Cons region 1', drop_firm)
households_0_t001 =  mean_variable(df_t_001, 'Households region 0', drop_firm)
households_1_t001 =  mean_variable(df_t_001, 'Households region 1', drop_firm)

#firms_0_s025 = ( firms_0_s025a + firms_0_s025b)/ 2
#firms_1_s025 = ( firms_1_s025a + firms_1_s025b)/ 2
households_0_t002 =  mean_variable(df_t_002, 'Households region 0', drop_firm)
households_1_t002 =  mean_variable(df_t_002, 'Households region 1', drop_firm)

households_0_t003 =  mean_variable(df_t_003, 'Households region 0', drop_firm)
households_1_t003 =  mean_variable(df_t_003, 'Households region 1', drop_firm)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( households_0_t0,  label = 't = 0', color = 'green')	
ax.plot( households_1_t0, '--', color = 'green', )	
ax.plot(  households_0_t001, label = 't = 0.01', color = 'blue')	
ax.plot(  households_1_t001,'--',  color = 'blue', )	
ax.plot(  households_0_t002, label = 't = 0.02', color = 'red')	
ax.plot(  households_1_t002, '--', color = 'red', )	
ax.plot(  households_0_t003, label = 't = 0.03', color = 'yellow')	
ax.plot(  households_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Households population", fontsize = 14)	

plt.legend()
plt.show()


'''
PLOT SIZE DISTRIBUTION
'''
all_sales = []
df_sales = result_1.filter(like = 'sales')
for i in range(len(model_lists)):
    list_firms = model_lists[i].firms2
    for j in list_firms:
        all_sales.append(j.sales)
all_sales = pd.Series(all_sales)


def rank_size_data(data, c=1.0):	

    w = - np.sort(- data)                  # Reverse sort	
    w = w[:int(len(w) * c)]                # extract top c%	
    rank_data = np.log(np.arange(len(w)) + 1)	
    size_data = np.log(w)	
    
    return rank_data, size_data	

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))	
    axes = axes.flatten()      

#rank_data, size_data = rank_size_data(pd_sales['Sales'])	
rank_data, size_data = rank_size_data(all_sales)	


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.plot( size_data, rank_data, 'o', markersize=3.0, alpha=0.5)	


ax.set_xlabel("log size", fontsize = 16)	
ax.set_ylabel("log rank", fontsize = 16)	


plt.show()

'''
CORRELATION STRUCTURE
'''

import seaborn as sns

def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    

corr = df_corr.corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)    

drop= 50
drop_end = 0
#df = df_t_001
df = result_1
output = af.mean_variable_log(df, 'GDP total', drop, drop_end)
cons = af.mean_variable_log(df, 'CONS Total', drop, drop_end )
inv =  af.mean_variable_log(df, 'INV Total', drop, drop_end)
prices =  af.mean_variable_log(df, 'Cons price region 1', drop, drop_end)
unemployment = af.mean_variable(df, 'Unemployment rate', drop, drop_end)

corr_list  = [ unemployment,  prices , inv , cons ,output ]
df_corr = pd.concat(corr_list, axis = 1)
df_corr.columns = [ 'Unemployment','Price',  'Inv',  'Cons','Output']
cycles = sm.tsa.filters.bkfilter(df_corr[['Output', 'Cons', 'Inv', 'Price', 'Unemployment']], 6, 32, 12)
cycles.columns = [    'Output' , 'Cons', 'Inv', 'Price','Unemployment']
corrMatrix = cycles.corr(method='pearson')
#corrMatrix = df_corr.corr(method='pearson')
sns.heatmap(corrMatrix, annot=True)
plt.show()
fig, ax = plt.subplots(figsize=(6, 6)) 
mask = np.zeros_like(corrMatrix.corr())
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corrMatrix, mask= mask, ax= ax, annot= True)


##--- ANALYZING MULTIPLE RUNS ---##

 
    
#plot_list_2var_comp_first_difference(macro_variable.GDP,macro_variable.Consumption_firms_av_prod ,200 , 50, "Turning the tide of agglomeration")
#af.plot_list_log(macro_variable.INVESTMENT, range(transition, steps), "Investment")
#af.plot_list_log(macro_variable.CONSUMPTION, range(transition, steps), "Consumption")
#af.plot_list(macro_variable.Competitiveness_Regional, range(transition,  steps), "Competitiveness")
#plot_list(macro_variable.Aggregate_Employment, range(transition, steps), "Aggregate Employment")
#plot_list(macro_variable.Population_Regional, range(steps), "Population")
#af.plot_list_log(macro_variable.Average_Salary, range(transition, steps) , "Average Salary")
af.plot_list(macro_variable.Population_Regional_Households, range(transition,steps), "Number of households")
#af.plot_list_log(macro_variable.Cosumption_price_average,  range(transition,  steps) , "Consumption price average")
af.plot_list(macro_variable.Population_Regional_Cons_Firms, range(transition, steps), "Number of consumption firms")
#af.plot_list_log(macro_variable.Capital_firms_av_prod, range(transition, steps), " Average productivity Cap firms")
af.plot_list(macro_variable.Population_Regional_Cap_Firms, range(transition,  steps), "Number of capital  firms")
#af.plot_list_log(macro_variable.Consumption_firms_av_prod, range(transition, steps ), " Average productivity Cons firms")
#af.plot_list(macro_variable.Unemployment_Regional, range( transition, steps), "Unemployment rate") 
#af.plot_list_log(macro_variable.GDP, range( steps), "GDP") 

#af.plot_list_2var_reg(macro_variable.GDP_cap, macro_variable#.INVESTMENT, range(steps), "Check account identity cap ")
#af.plot_list_2var_reg(macro_variable.GDP_cons, macro_variable.CONSUMPTION, range(steps), "Check account IDENTITY CONS  ")
#af.plot_list_2var_reg(macro_variable.Capital_firms_av_prod, macro_variable.Consumption_firms_av_prod, range(steps), "Productivity ")

#af.plot_list_2var_reg(macro_variable.Average_Salary_Capital, macro_variable.Average_Salary_Cons, range(steps), "Wages")


#macro_variable = macro_variables[3]

##--- ANALYZING MULTIPLE RUNS ---##
df_pi = macro_variables[7]
df_pi[['Av wage region 0','Av wage region 1', 'Wage diff 0', 'Wage diff 1']] = pd.DataFrame(df_pi.Average_Salary.to_list(), index= df_pi.index)
df_pi['Wage prod ratio 0']= df_pi['Av wage region 0']/ df_pi['Prod region 0']
df_pi_int = df_pi[['Cons price region 0', 'Prod region 0', 'Av wage region 0', 'Delta prod region 0','Wage prod ratio 0'  ]]
 
for i in range(len(macro_variables)):
    print(i)
    result_1 = macro_variables[i]
    #gdp_gr = af.variable_growth_rate(result_1, 'GDP cons total', transition, steps)
    mean_gdp_0 = af.mean_variable_log(result_1, 'GDP cons region 0', 120) /  af.mean_variable(result_1, 'Households region 0') # *  af.mean_variable(result_1, 'Cons price region 0', 120,80 )[150] /   af.mean_variable(result_1, 'Cons price region 0', 120,80 )
   # mean_gdp_1 = af.mean_variable_log(result_1, 'GDP region 1')

    
    gdp_plot = [[mean_gdp_0, 'Coastal region', 'blue']] #, [mean_gdp_1, 'Inland region', 'green']]
    af.plot(gdp_plot, 100, 250)


macro_variables_1 = macro_variables
drop = 120
drop_end = 85
case_0_mean_gdp_0 = af.mean_variable_log(macro_variables_1[48], 'GDP cons region 0', drop,drop_end) /  af.mean_variable(macro_variables_1[48], 'Households region 0', drop, drop_end) 
case_1_mean_gdp_0 = af.mean_variable_log(macro_variables_1[49], 'GDP cons region 0', drop, drop_end) /  af.mean_variable(macro_variables_1[48], 'Households region 0', drop, drop_end) 
case_3_mean_gdp_0 = af.mean_variable_log(macro_variables_1[8], 'GDP cons region 0', drop,drop_end) /  af.mean_variable(macro_variables_1[8], 'Households region 0', drop, drop_end)
case_2_mean_gdp_0 = af.mean_variable_log(macro_variables_1[37], 'GDP cons region 0', drop,drop_end) /  af.mean_variable(macro_variables_1[37], 'Households region 0', drop, drop_end)  
##case_4_mean_gdp_0 = af.mean_variable_log(macro_variables_1[42], 'GDP cons region 0', drop, drop_end) /  af.mean_variable(macro_variables_1[42], 'Households region 0', drop, drop_end)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot(case_0_mean_gdp_0 ,   color = 'pink')
ax.plot( case_1_mean_gdp_0  ,  color = 'orange')
ax.plot( case_2_mean_gdp_0  ,   color = 'olive')
ax.plot( case_3_mean_gdp_0  ,   color = 'brown')
ax.plot( case_4_mean_gdp_0  ,  color = 'grey')
ax.set_xlabel('Steps')
ax.set_ylabel('GDP per capita (log)')
#ax.plot( perc_cap_0_run_1 , label = 'Cap', color = 'black' )
plt.legend()
plt.show()

'''
df_gdp_pct = result_1.filter(like = 'GDP total').pct_change()
df_gdp = result_1.filter(like = 'GDP total')

df_gdp_aggl_preflood = df_gdp.iloc[:, [0,1,2,3,4,7,8,10,11,13,14]]
last_raw = df_gdp_aggl_preflood.iloc[300, :]
'''


#plot_list_2var_comp_first_difference(macro_variable.GDP,macro_variable.Consumption_firms_av_prod ,200 , 50, "Turning the tide of agglomeration")
#af.plot_list_log(macro_variable.INVESTMENT, range(transition, steps), "Investment")
#af.plot_list_log(macro_variable.CONSUMPTION, range(transition, steps), "Consumption")
#af.plot_list(macro_variable.Competitiveness_Regional, range(transition,  steps), "Competitiveness")
#plot_list(macro_variable.Aggregate_Employment, range(transition, steps), "Aggregate Employment")
#plot_list(macro_variable.Population_Regional, range(steps), "Population")
#af.plot_list_log(macro_variable.Average_Salary, range(transition, steps) , "Average Salary")
    af.plot_list(macro_variable.Population_Regional_Households, range(transition,steps), "Number of households")
   # af.plot_list_log(macro_variable.Cosumption_price_average,  range(transition,  steps) , "Consumption price average")
    af.plot_list(macro_variable.Population_Regional_Cons_Firms, range(transition, steps), "Number of consumption firms")
#af.plot_list_log(macro_variable.Capital_firms_av_prod, range(transition, steps), " Average productivity Cap firms")
    #af.plot_list(macro_variable.Population_Regional_Cap_Firms, range(transition,  steps), "Number of capital  firms")
#af.plot_list_log(macro_variable.Consumption_firms_av_prod, range(transition, steps ), " Average productivity Cons firms")
#af.plot_list(macro_variable.Unemployment_Regional, range( transition, steps), "Unemployment rate") 
#af.plot_list_log(macro_variable.GDP, range( steps), "GDP") 

#af.plot_list_2var_reg(macro_variable.GDP_cap, macro_variable#.INVESTMENT, range(steps), "Check account identity cap ")
#af.plot_list_2var_reg(macro_variable.GDP_cons, macro_variable.CONSUMPTION, range(steps), "Check account IDENTITY CONS  ")
#af.plot_list_2var_reg(macro_variable.Capital_firms_av_prod, macro_variable.Consumption_firms_av_prod, range(steps), "Productivity ")

#af.plot_list_2var_reg(macro_variable.Average_Salary_Capital, macro_variable.Average_Salary_Cons, range(steps), "Wages")


#macro_variable = macro_variables[3]

df_run_1 = macro_variables[0]
df_run_2 = macro_variables[1]
df_run_3 = macro_variables[2]
df_run_4 = macro_variables[3]
df_run_even = macro_variables[41]
df_run_inland = macro_variables[4]

df_t002_different_cases = pd.concat([df_run_1, df_run_even, df_run_inland])

df_t002_different_cases.to_csv(r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\t002_scenario.csv')    

firms_pop = 250
drop = 90
drop_end = 200
households_pop = 4000
cap_pop = 60

## ------ % of consumption firms in Coastal region -----###

perc_firms_0_run_1 =  af.mean_variable(df_run_1, 'Cons region 0', drop) / firms_pop
#perc_firms_0_run_even =  af.mean_variable(df_run_even, 'Cons region 0', drop) / firms_pop
#perc_firms_0_run_inland =  af.mean_variable(df_run_inland, 'Cons region 0', drop) / firms_pop
#perc_firms_0_t002 =  af.mean_variable(df_t_002, 'Cons region 0', drop_firm) / firms_pop
perc_firms_0_run_2 =  af.mean_variable(df_run_2, 'Cons region 0', drop) / firms_pop
perc_firms_0_run_3 =  af.mean_variable(df_run_3, 'Cons region 0', drop) / firms_pop
perc_firms_0_run_4 =  af.mean_variable(df_run_4, 'Cons region 0', drop) / firms_pop


##-----% of households in coastal region ----###
perc_households_0_run_1 =  af.mean_variable(df_run_1, 'Households region 0', drop) / households_pop
perc_households_0_run_even =  af.mean_variable(df_run_even, 'Households region 0', drop) / households_pop
perc_households_0_run_inland =  af.mean_variable(df_run_inland, 'Households region 0', drop) / households_pop
#firms_0_t0 =  mean_variable(df_t_000, 'Cons region 0', drop_firm) / firms_pop

##------ % of Cap firms in coastal region ---###

perc_cap_0_run_1 =  af.mean_variable(df_run_1, 'Cap region 0', drop) / cap_pop
perc_cap_0_run_even =  af.mean_variable(df_run_even, 'Cap region 0', drop) / cap_pop
perc_cap_0_run_inland =  af.mean_variable(df_run_inland, 'Cap region 0', drop) / cap_pop
#


###----- Figure with migrations of all agents --- ###

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(311)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( perc_firms_0_run_1 ,  label = 'Cons', color = 'blue')
ax.plot( perc_households_0_run_1 ,  label = 'Households', color = 'green')
ax.plot( perc_cap_0_run_1 , label = 'Cap', color = 'black' )


ax1 = fig.add_subplot(312)
ax1.plot( perc_firms_0_run_even ,  label = 'Cons', color = 'blue')
ax1.plot( perc_households_0_run_even , label = 'Households', color = 'green')	
ax1.plot( perc_cap_0_run_even ,  label = 'Cap', color = 'black' )
ax1.set_ylabel('% of population residing in Coastal region' , fontsize = 14)	

ax2 = fig.add_subplot(313)
ax2.plot( perc_households_0_run_inland , label = 'Households', color = 'green')	
ax2.plot( perc_firms_0_run_inland ,  label = 'Cons', color = 'blue')		
ax2.plot( perc_cap_0_run_inland ,  label = 'Cap', color = 'black' )			
#ax.plot( perc_firms_0_run_4 , label = 'Run 4', color = 'blue')	
#ax.plot(  perc_firms_0_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( perc_firms_0_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	

ax2.set_xlabel("Time step", fontsize=14)	


plt.legend()
plt.show()


drop_end = 50

####-----Drivers of households migration ---###

wage_ratio_run_1 =  af.mean_variable(df_run_1, 'Wage region 0', drop, drop_end) / af.mean_variable(df_run_1, 'Wage region 1', drop, drop_end)
#unemployment_ratio_run_1 =  af.mean_variable(df_run_1, 'Unemployment region 0', drop_firm) / af.mean_variable(df_run_1, 'Unemployment region 1', drop, drop_end)

wage_ratio_run_even =  af.mean_variable(df_run_even, 'Wage region 0', drop, drop_end) / af.mean_variable(df_run_even, 'Wage region ', drop, drop_end)
#unemployment_ratio_even=  af.mean_variable(df_run_2, 'Unemployment region 0', drop_firm) / af.mean_variable(df_run_2, 'Unemployment region 1', drop, drop_end)
wage_ratio_run_inland =  af.mean_variable(df_run_inland, 'Wage region 0', drop, drop_end) / af.mean_variable(df_run_inland, 'Wage region ', drop, drop_end)

prod_ration_firms_run_1 =  af.mean_variable(df_run_1, 'Prod region 0', drop, drop_end) / af.mean_variable(df_run_1, 'Prod region 1', drop, drop_end)
prod_ration_firms_run_even =  af.mean_variable(df_run_even, 'Prod region 0', drop, drop_end) / af.mean_variable(df_run_even, 'Prod region 1', drop, drop_end)
prod_ration_firms_run_inland =  af.mean_variable(df_run_inland, 'Prod region 0', drop, drop_end) / af.mean_variable(df_run_inland, 'Prod region 1', drop, drop_end)


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( wage_ratio_run_1 ,  label = 'Wage', color = 'blue')
ax.plot( prod_ration_firms_run_1 , '--',  label = 'Productivity', color = 'blue')
	
ax.plot( wage_ratio_run_even ,  label = 'Wage', color = 'black')	
ax.plot( prod_ration_firms_run_even , '--',  label = 'Productivity', color = 'black')

ax.plot( wage_ratio_run_inland ,  label = 'Wage', color = 'green')
ax.plot( prod_ration_firms_run_inland , '--',  label = 'Productivity ', color = 'green')

		

#ax.plot( unemployment_ratio_run_2 ,  label = 'Unemplyoemt ration', color = 'green')	
#ax.plot( perc_cap_0_run_2 , label = 'Cap', color = 'black', )	
#ax.plot( perc_firms_0_run_4 , label = 'Run 4', color = 'blue')	
#ax.plot(  perc_firms_0_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( perc_firms_0_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	

ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Coastal/Inland atio", fontsize = 14)	

plt.legend()
plt.show()


## --- SHARE OF EXPORT ----##





'''
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( perc_firms_0_run_1 ,  label = 'Run 1', color = 'red')	
ax.plot( perc_firms_0_run_2 ,  label = 'Run 2', color = 'green')	
ax.plot( perc_firms_0_run_3 , label = 'Run 3', color = 'black', )	
ax.plot( perc_firms_0_run_4 , label = 'Run 4', color = 'blue')	
#ax.plot(  perc_firms_0_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( perc_firms_0_t025, label = 't = 0.25', color  0-v099 = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Firms population", fontsize = 14)	

plt.legend()
plt.show()
'''

## -- producitvity ration between the two regions ---##

prod_ration_firms_run_4 =  af.mean_variable(df_run_4, 'Prod region 0', drop_firm) / af.mean_variable(df_run_4, 'Prod region 1', drop_firm, drop_end)
#prod_ration_firms_t01 =  af.mean_variable(df_t_01, 'Prod region 0', drop_firm) / af.mean_variable(df_t_01, 'Prod region 1', drop_firm)
#prod_ration_firms_t025 =  af.mean_variable(df_t_025, 'Prod region 0', drop_firm) / af.mean_variable(df_t_025, 'Prod region 1', drop_firm)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( prod_ration_firms_run_1 ,  label = 'Run 1', color = 'red')	
ax.plot( prod_ration_firms_run_2 ,  label = 'Run 2', color = 'green')	
ax.plot(prod_ration_firms_run_3 , label = 'Run 3', color = 'black', )	
ax.plot( prod_ration_firms_run_4 ,label = 'Run 4', color = 'blue')	
#ax.plot(  prod_ration_firms_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( prod_ration_firms_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Productivity ratio", fontsize = 14)	

plt.legend()
plt.show()



d_ration_firms_run_1 =  af.mean_variable(df_run_1, 'CONS 0', drop, drop_end) / af.mean_variable(df_run_1, 'CONS 1', drop, drop_end)
d_ration_firms_run_even =  af.mean_variable(df_run_even, 'CONS 0', drop, drop_end) / af.mean_variable(df_run_even, 'CONS 1', drop, drop_end)
d_ration_firms_run_inland =  af.mean_variable(df_run_inland, 'CONS 0', drop, drop_end) / af.mean_variable(df_run_inland, 'CONS 1', drop, drop_end)
#d_ration_firms_run_2 =  af.mean_variable(df_run_2, 'CONS 0', drop_firm) / af.mean_variable(df_run_2, 'CONS 1', drop_firm , drop_end)
#d_ration_firms_run_3=  af.mean_variable(df_run_3, 'CONS 0', drop_firm) / af.mean_variable(df_run_3, 'CONS 1', drop_firm, drop_end)
#d_ration_firms_run_4 =  af.mean_variable(df_run_4, 'CONS 0', drop_firm) / af.mean_variable(df_run_4, 'CONS 1', drop_firm, drop_end)
#prod_ration_firms_t01 =  af.mean_variable(df_t_01, 'Prod region 0', drop_firm) / af.mean_variable(df_t_01, 'Prod region 1', drop_firm)
#prod_ration_firms_t025 =  af.mean_variable(df_t_025, 'Prod region 0', drop_firm) / af.mean_variable(df_t_025, 'Prod region 1', drop_firm)

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])

ax.plot( d_ration_firms_run_1 ,  label = 'Coastal', color = 'blue')	
ax.plot( d_ration_firms_run_even ,  label = 'Even', color = 'black')
ax.plot( d_ration_firms_run_inland ,  label = 'Even', color = 'green')	
#ax.plot( d_ration_firms_run_2 ,  label = 'Run 2', color = 'green')	
#ax.plot(d_ration_firms_run_3 , label = 'Run 3', color = 'black', )	
#ax.plot( d_ration_firms_run_4 ,label = 'Run 4', color = 'blue')	
#ax.plot(  prod_ration_firms_t01 ,label = 't - 0.1',  color = 'yellow', )	
#ax.plot( prod_ration_firms_t025, label = 't = 0.25', color = 'purple')	
#ax.plot(  perc_firms_0_t002 , '--', color = 'red', )	
#ax.plot(  perc_firms_0_t002 , label = 't = 0.02', color = 'yellow')	
#ax.plot(  firms_1_t003, '--', color = 'yellow', )	


ax.set_xlabel("Time step", fontsize=14)	
ax.set_ylabel("Demand ratio", fontsize = 14)	

plt.legend()
plt.show()


av_exp_share_0 = af.mean_variable(result_1, 'Exp share region 0', 10, 150).plot()