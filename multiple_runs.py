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
import math
from model.app import plot_list

   

runs=4
steps=250
macro_variables = []
for i in range(runs):
    model = KSModel(F1 = 40, F2=200, H=6000, B=1, T=20, S= 0.5)
    print("#-------------- iteration", i+1, "---------------#")
    for j in range(steps):
        print("#------------ step", j+1, "------------#")
        model.step()
    run_data = model.datacollector.get_model_vars_dataframe()
    macro_variables.append(run_data)
    plot_list(run_data.Unemployment_Regional, range(20 ,steps), "Unemployment rate")

    plot_list(run_data.GDP, range(steps), "GDP")
    plot_list(run_data.INVESTMENT, range(steps), "Investment")
    plot_list(run_data.Competitiveness_Regional, range( 5,steps), "Competitiveness")
    plot_list(run_data.Aggregate_Employment, range(steps), "Aggregate Employment")
    #plot_list(macro_variable.Population_Regional, range(steps), "Population")
    plot_list(run_data.Average_Salary, range(steps) , "Average Salary")
    plot_list(run_data.Population_Regional_Households, range(steps), "Number of households")
    plot_list(run_data.Cosumption_price_average,  range( 20, steps) , "Consumption price average")
    plot_list(run_data.Population_Regional_Cons_Firms, range(steps), "Number of consumption firms")
    plot_list(run_data.Capital_firms_av_prod, range(steps), " Average productivity Cap firms")
    plot_list(run_data.Population_Regional_Cap_Firms, range(steps), "Number of capital  firms")
    plot_list(run_data.Consumption_firms_av_prod, range(steps), " Average productivity Cons firms")
    

 


'''
Here below is just some plots for the results work in progress
'''   
unemployment_d = da.extract_data(macro_variables, "Unemployment_Regional")
average_salary  = da.extract_data(macro_variables, "Average_Salary")
cons_prod = da.extract_data(macro_variables, "Consumption_firms_av_prod")
gdp_d= da.extract_data(macro_variables, "GDP")
price = da.extract_data(macro_variables, "Cosumption_price_average")

resilience_coeff = da.extract_data(macro_variables, "Average_CCA_coeff")

da.plot_raw(unemployment_d['steps'], "Unemployment_Regional", mode='scatter', plotnum=2)
da.plot_mean_ci(unemployment_d['steps'],
                title="Mean Regional Unemployment",
                mode='graph',
                plotnum=2)
da.plot_compare(resilience_coeff, average_salary, title="GDP Cons and Cap firms", var1='Cons', var2='Cap')
    

    

a = macro_variables[0]
b = macro_variables[1]
c = macro_variables[2]
d = macro_variables[3]

df_concat = pd.concat((a, b,c,d))


by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()


macro_variable[['Cons region 0','Cons region 1']] = pd.DataFrame(macro_variable.Population_Regional_Cons_Firms.to_list(), index= macro_variable.index)
a[['Average prod 0','Average prod 1']] = pd.DataFrame(a.Regional_average_productivity.to_list(), index=a.index)
b[['Average prod 0','Average prod 1']] = pd.DataFrame(b.Regional_average_productivity.to_list(), index=b.index)
c[['Average prod 0','Average prod 1']] = pd.DataFrame(c.Regional_average_productivity.to_list(), index=c.index)  
b[['POP 0',' POP 1']] = pd.DataFrame(b.Population_Regional_Households.to_list(), index=b.index) 

a[['GDP 0',' GDP 1', "GDP total"]] = pd.DataFrame(a.GDP.to_list(), index=a.index)
 
b[['Average prod 0','Average prod 1']] = pd.DataFrame(b.Regional_average_productivity.to_list(), index= macro_variable.index)  
regional_prod_df = pd.DataFrame({"prod":prod, "region":region})  
b[['GDP 0','GDP 1', "GDP total "]] = pd.DataFrame(b.GDP.to_list(), index= b.index) 
c[['GDP 0','GDP 1', "GDP total "]] = pd.DataFrame(c.GDP.to_list(), index= c.index) 

c[['Unemployment 0','Unemployment 1']] = pd.DataFrame(c.Unemployment_Regional.to_list(), index= c.index) 
sns.set_style('prod regions')
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)



ax = sns.regplot(x=b.index, y='GDP 1', color = "red", dropna=True,data=c, label= "region 1" ,  x_estimator=np.mean)

ax.legend(loc = "upper center")                 
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y='GDP 0', color = "blue", dropna=True, data=c, label= "region 0",
                  x_estimator=np.mean)
ax2.legend(loc = "upper left")


ax = sns.regplot(x=b.index, y='Unemployment 1', color = "red", dropna=True,data=c, label= "region 1" ,  x_estimator=np.mean)

ax.legend(loc = "upper center")                 
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y='Unemployment 0', color = "blue", dropna=True, data=c, label= "region 0",
                  x_estimator=np.mean)
ax2.legend(loc = "upper left")



ax = sns.regplot(x=b.index, y='Average prod 1', color = "red", data=b,
                  x_estimator=np.mean)
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y='Average prod 0', color = "blue", data=b,
                  x_estimator=np.mean)

ax = sns.regplot(x=b.index, y='Average prod 1', color = "red", dropna=True,data=a, label= "region 1" ,  x_estimator=np.mean)

ax.legend(loc = "upper center")                 
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y='Average prod 0', color = "blue", dropna=True, data=a, label= "region 0",
                  x_estimator=np.mean)
ax2.legend(loc = "upper left")
ax = sns.regplot(x=b.index, y='POP 0', color = "red", data=b,
                  x_estimator=np.mean)
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y=' POP 1', color = "blue", data=b,
                  x_estimator=np.mean)

ax = sns.regplot(x=a.index, y='Average prod 1', color = "red", data=a,
                  x_estimator=np.mean)
ax2 = ax.twinx()
ax2 = sns.regplot(x=a.index, y='Average prod 0', color = "blue", data=a,
                  x_estimator=np.mean)


ax = sns.regplot(x=b.index, y='Average prod 1', color = "red", data=b,
                  x_estimator=np.mean)
ax2 = ax.twinx()
ax2 = sns.regplot(x=b.index, y='Average prod 0', color = "blue", data=b,
                  x_estimator=np.mean)


plot_list_2var_comp_first_difference(d.GDP, d.Consumption_firms_av_prod ,250 , 10, "GDP and productivity")
plot_list_3var_comp_first_difference_comp2( a.RD_CCA_INVESTMENT, b.RD_CCA_INVESTMENT, a.Average_CCA_coeff, b.Average_CCA_coeff,  200, 50  )
plot_list_3var_comp_first_difference_comp( a.Population_Regional_Households, a.Population_Regional_Cons_Firms, a.Population_Regional_Cap_Firms, b.Population_Regional_Households, b.Population_Regional_Cons_Firms, b.Population_Regional_Cap_Firms, b.Average_CCA_coeff, a.Average_CCA_coeff,200, 50  ) 
plot_list_3var_comp_first_difference(b.Population_Regional_Cons_Firms, b.Population_Regional_Cap_Firms, b.Population_Regional_Households, 250, 50  ) 
plot_list_3var_comp_first_difference(c.Population_Regional_Cons_Firms, c.Population_Regional_Cap_Firms, c.Population_Regional_Households, 250, 50  ) 
plot_list_3var_comp_first_difference(d.Population_Regional_Cons_Firms, d.Population_Regional_Cap_Firms, d.Population_Regional_Households, 250, 50, "Regional population over time"  )

