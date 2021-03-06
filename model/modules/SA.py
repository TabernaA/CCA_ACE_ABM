'''
from IPython.display import clear_output
!pip install SALib
import SALib
clear_output()
print("Everything A-Okay!")
'''
seed_value = 1
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed=seed_value)

from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm

from model.classes.model import KSModel
from SALib.sample import saltelli

from mesa.batchrunner import BatchRunnerMP
#from model.modules.batchrunner_local import BatchRunnerMP
from SALib.analyze import sobol
import pandas as pd

import matplotlib.pyplot as plt
from itertools import combinations
import model.modules.additional_functions as af

def gdp_cons(model):
    GDP = 0
    agents = model.firms2 #model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if agents[i].region == 0:
            GDP +=  firm.production_made
    
    return GDP

def regional_unemployment_rate(model):
    gov = model.governments[0]
    unemployment_rate = gov.unemployment_rates[0]
    return round(unemployment_rate, 2)


def price(model):
    price = 0 
    agents = model.firms2 
    for i in range(len(agents)):
        a = agents[i]
        #if a.type == "Cons":
        price += a.price * sum(a.market_share) 
    return round( price/ 2, 2)


    
def regional_population_households_region_0(model):
    gov =  model.governments[0]
    
    households_pop = gov.regional_pop_hous[0] / sum(gov.regional_pop_hous)
    return households_pop

def regional_population_cons_region_0(model):
    r0 = 0
    agents = model.firms2 
    for i in range(len(agents)):
        a = agents[i]
        if a.region == 0:
            r0 += 1
        
    #print("Cap population region 0,1 is ", [r0, r1]) 
    return r0
    
# We define our variables and bounds
'''
problem = {
    'num_vars':2,
    'names': ['T','S'],
    'bounds': [[0, 20], [0, 5]]
   # 'bounds': [[1000000, 20000000], [1000000, 5000000]]
}
'''
problem = {
    'num_vars':1,
    'names': ['S'],
    'bounds': [ [0, 0]]
   # 'bounds': [[1000000, 20000000], [1000000, 5000000]]
}
'''
problem = {
    'num_vars':1,
    'names': ['S'],
    'bounds': [[0, 5], [0, 20]]
   # 'bounds': [[1000000, 20000000], [1000000, 5000000]]
}
'''

# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 50
max_steps = 400
distinct_samples = 1

# Set the outputs
model_reporters = {   "Population_Region_0_Households" :regional_population_households_region_0,
                      "Population_Region_0_Cons_Firms":regional_population_cons_region_0,
                      'Price' : price,
                      "Real GDP": gdp_cons,
                      'Unemployment_rate_0': regional_unemployment_rate}

data = {}

macro_variables = []


for i, var in enumerate(problem['names']):
    # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
    samples = np.linspace(*problem['bounds'][i], num=distinct_samples)
    
    # Keep in mind that wolf_gain_from_food should be integers. You will have to change
    # your code to acommidate for this or sample in such a way that you only get integers.
    if var == 'S':
        samples = np.linspace(*problem['bounds'][i], num=distinct_samples, dtype=int)
    
    batch = BatchRunnerMP(KSModel,nr_processes= 8,
                        #fixed_parameters= fixed_params,
                        max_steps=max_steps,
                        iterations=replicates,
                        variable_parameters={var: samples},
                        model_reporters=model_reporters,
                        display_progress=True)
    
    batch.run_all()
    data[var] = batch.get_model_vars_dataframe()
    run_data = batch.get_collector_model()
    macro_variables.append([run_data, var]) 
    
    
    
def plot_param_var_conf(ax, df, var, param, i):
    """
    Helper function for plot_all_vars. Plots the individual parameter vs
    variables passed.

    Args:
        ax: the axis to plot to
        df: dataframe that holds the data to be plotted
        var: variables to be taken from the dataframe
        param: which output variable to plot
    """
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]

    replicates = df.groupby(var)[param].count()
    err = (1.96 * df.groupby(var)[param].std()) / np.sqrt(replicates)

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)

    ax.set_xlabel(var)
    ax.set_ylabel(param)

def plot_all_vars(df, param):
    """
    Plots the parameters passed vs each of the output variables.

    Args:
        df: dataframe that holds all data
        param: the parameter to be plotted
    """

    f, axs = plt.subplots(2, figsize=(7, 10))
    
    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param, i)

for param in ("Population_Region_0_Households", "Population_Region_0_Cons_Firms",'Price',
                      'GDP',
                      'Unemployment_rate_0'  ):
    plot_all_vars(data, param)
    plt.show()
    
    



    
'''
SOBOL SENSITIVITY
'''  
## --- GETTING THE DATA --##
# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 5
max_steps = 250
distinct_samples = 100

# We get all our samples here
param_values = saltelli.sample(problem, distinct_samples)

# READ NOTE BELOW CODE
batch = BatchRunner(KSModel, 
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    model_reporters=model_reporters)


count = 0
for i in range(replicates):
    for vals in param_values: 
        # Change parameters that should be integers
        vals = list(vals)
        vals[1] = int(vals[1])

        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val

        batch.run_iteration(variable_parameters, tuple(vals), count)
        count += 1

        clear_output()
        print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')
    
data = batch.get_model_vars_dataframe()

print(data)

## ----ANALYZING DATA ---##

# This is not very insightful
Si_hous_pop_coastal = sobol.analyze(problem, data['Population_Region_0_Households'].as_matrix(), print_to_console=True)
Si_firms_pop_coastal = sobol.analyze(problem, data['population_Region_0_Cons_Firms'].as_matrix(), print_to_console=True)
Si_price = sobol.analyze(problem, data['Price'].as_matrix(), print_to_console=True)
Si_gdp = sobol.analyze(problem, data['GDP'].as_matrix(), print_to_console=True)
Si_ur = sobol.analyze(problem, data['Unemployment_rate'].as_matrix(), print_to_console=True)



## ---better do a function to plot it ---##
def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')

# 1st, 2nd and total order sensitivtity analysis

for Si in (Si_hous_pop_coastal, Si_firms_pop_coastal, Si_price, Si_gdp, Si_ur):
    # First order
    plot_index(Si, problem['names'], '1', 'First order sensitivity')
    plt.show()

    # Second order
    plot_index(Si, problem['names'], '2', 'Second order sensitivity')
    plt.show()

    # Total order
    plot_index(Si, problem['names'], 'T', 'Total order sensitivity')
    plt.show()
    
    
    
    
'''
BANDPASS FILTER
'''   
import statsmodels.api as sm

drop = 50
gdp = af.mean_variable_log(macro_variable, 'GDP total', drop)
inv = af.mean_variable_log(macro_variable, 'INVESTMENT total', drop)
cons = af.mean_variable_log(macro_variable, 'CONSUMPTION total', drop)
debt =   af.mean_variable_log(macro_variable, 'Debt', drop)

bp = pd.concat([gdp, inv, cons, debt], axis = 1)
bp.columns = ['GDP', 'INV', 'CONS', 'DEBT']
cycles = sm.tsa.filters.bkfilter(bp[['DEBT','INV','CONS', 'GDP']], 6, 32, 12)
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

plt.show()


'''
CORRELATION STRUCTURE
'''

    
'''
CORRELATION STRUCTURE
'''
import seaborn as sns

drop= 50
drop_end = 0
#df = df_t_001

drop = 50
gdp = af.mean_variable_log(macro_variable, 'GDP total', drop)
inv = af.mean_variable_log(macro_variable, 'INVESTMENT total', drop)
cons = af.mean_variable_log(macro_variable, 'CONSUMPTION total', drop)
debt =   af.mean_variable_log(macro_variable, 'Debt', drop)
prices = af.mean_variable_log(macro_variable, 'Price total', drop, drop_end)
unemployment = af.mean_variable(macro_variable, 'Unemployment total', drop, drop_end)

corr_list  = [ debt , unemployment,  prices , inv , cons ,gdp]
df_corr = pd.concat(corr_list, axis = 1)
df_corr.columns = [ 'Debt',  'Unemployment','Price',  'Inv',  'Cons','Output']
cycles = sm.tsa.filters.bkfilter(df_corr[['Output', 'Cons', 'Inv', 'Price', 'Unemployment', 'Debt']], 6, 32, 12)
cycles.columns = [    'Output' , 'Cons', 'Inv', 'Price','Unemployment', 'Debt']
#corrMatrix = df_corr.corr(method='pearson')
corrMatrix = cycles.corr(method='pearson')
sns.heatmap(corrMatrix, annot=True)
plt.show()
fig, ax = plt.subplots(figsize=(6, 6)) 
mask = np.zeros_like(corrMatrix.corr())
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corrMatrix, mask= mask, ax= ax, annot= True)

result_1 = pd.concat(run_data, axis = 1 )
result_1.to_csv(r'C:\Users\tabernaa\Documents\PHD UTWENTE\Research\first_model\Versions\Last\final\model\data_results\S05_Test.csv')  

df_s_0 = result_1

drop_firm = 0
drop_end = 0
#firms_0_s050 =  af.mean_variable(df_s_050, 'Population_Region_0_Cons_Firms', drop_firm, drop_end) /250
#firms_0_s050_std =  af.std_dev_variable(df_s_050, 'Population_Region_0_Cons_Firms', drop_firm, drop_end)  / 250  / np.sqrt(800)# / 250 #* 2
#firms_1_s050 =  af.mean_variable(df_s_050, 'Cons region 1', drop_firm) * 2
#firms_0_s025a =  mean_variable(df_s_025a, 'Cons region 0', drop_firm)
#firms_1_s025a =  mean_variable(df_s_025a, 'Cons region 1', drop_firm)
#firms_0_s030 =  af.mean_variable(df_s_030 , 'Population_Region_0_Cons_Firms', drop_firm, drop_end) / 250 
#firms_0_s030_std =  af.std_dev_variable(df_s_030, 'Population_Region_0_Cons_Firms', drop_firm, drop_end)  / 250  / np.sqrt(600) #* 2
#firms_1_s025 =  af.mean_variable(df_s_025, 'Cons region 1', drop_firm) * 2

firms_0_s0 =  af.mean_variable(df_s_0, "Population_Region_0_Cons_Firms", drop_firm, drop_end) / 250
#firms_0_s0_std =  af.std_dev_variable(df_s_0, 'Population_Region_0_Cons_Firms', drop_firm, drop_end)  / 250  / np.sqrt(600)

variable_all = result_1.filter(like = "Population_Region_0_Cons_Firms")


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#df_gdp_1_concat.plot(ax=ax, style=['r--','b-'])



#ax.plot( firms_0_t001,  label = 'F = 0', color = 'black')	
#ax.plot( firms_1_t001, '--', color = 'black', )	
#ax.plot( firms_0_s010,  label = 'Dc = 0.1', color = 'blue')	
#ax.fill_between( firms_0_s010.index, firms_0_s010 - firms_0_s010_std, firms_0_s010 + firms_0_s010_std, color = 'blue', alpha = .1 )
#ax.plot( firms_0_s050,  label = 'Dc = 0.5', color = 'black')	
#ax.fill_between( firms_0_s050.index, firms_0_s050 - firms_0_s050_std, firms_0_s050 + firms_0_s050_std, color = 'black', alpha = .1 )
#ax.plot( firms_1_s050, label = 'F = 0.5', color = 'red', )	
#ax.plot(  firms_0_s030,  label = 'Dc = 0.3', color = 'purple')	
#ax.fill_between( firms_0_s030.index, firms_0_s030 - firms_0_s030_std, firms_0_s030 + firms_0_s030_std , color = 'purple', alpha = .1)
#ax.plot(  firms_1_s025,'--', label = 'F = 0.25',  color = 'red', )	
ax.plot(  firms_0_s0, label = 'Dc = 0', color = 'gold')
#ax.fill_between( firms_0_s0.index, firms_0_s0 - firms_0_s0_std, firms_0_s0 + firms_0_s0_std , color = 'gold', alpha = .1)	
#ax.plot(  firms_0_s010, 'o', label = 'F = 0.1', color = 'grey', )	

ax.set_xlabel("Time step", fontsize = 16)	
ax.set_ylabel("Firms population in Coastal region (over total)", fontsize = 16)	

plt.legend()
plt.show()