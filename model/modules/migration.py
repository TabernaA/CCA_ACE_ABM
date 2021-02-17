# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:31:50 2020
@author: TabernaA
modules/migration.py
"""

import math 
#import random
import bisect
from scipy.stats import bernoulli



'''
Calculation of households probability to migrate
'''
def households_migration_probability(region, model, w_1 = 1): 
    ##--retrieve relevant parameters --#
    prob_migration = 0
    gov = model.governments[0]
    unemployment_diff = gov.unemployment_rates[2 + region]
    wage_diff=  gov.average_wages[2 + region]
    if  wage_diff < 0 and unemployment_diff < 0:
        prob_migration = 1-math.exp(w_1* wage_diff )
   # migration_pr.append([wage_diff, unemployment_diff, prob_migration, region])
    
    '''
    unemployment = model.datacollector.model_vars['Unemployment_Regional'][int(model.schedule.time)]
    average_wage = model.datacollector.model_vars['Average_Salary'][int(model.schedule.time)]

    #current_average_price = model.datacollector.model_vars["Cosumption_price_average"][int(model.schedule.time)]
    #real_income =round( my_income / (current_average_price[region] + 0.0001), 5)
    #real_income_exp = round(average_wage[1-region]/( current_average_price[ 1 - region] + 0.0001),5)
    wage_diff = (average_wage[region] - average_wage[1 -region]) 
    
    unemployment_diffe =  unemployment[1-region] - unemployment[region]
    if abs(unemployment_diffe) > 0.1:
         unemployment_diff = unemployment_diffe / max(unemployment)                       
    else:
        unemployment_diff = 0#rint("unemployemt", )
    ##---calculation of probability --##
    
    '''
    

    #prob_migration = prob_migration * (1 - mc)
    #print("H region ", region, "real income" , real_income, " real_income exp", real_income_exp, "unemployment rate", unemployment[region],
     #     "Unemployement ex", unemployment[1 - region], " probability", prob_migration)
    
    return prob_migration




'''
households migration procedure
'''
def household_migrate(mp, model, region, unique_id):
    ##-- if the probability is higher than random number between 0,1, leave the job and move out --##
    agent = model.schedule._agents[unique_id]
    if bernoulli.rvs(mp) == 1:
        agent.lifecycle = 0
		# leave the job in the region and become unemployed
        if agent.employer_ID != None:
            employer = model.schedule._agents[agent.employer_ID]
            if unique_id in employer.employees_IDs:
                employer.employees_IDs.remove(unique_id)
                agent.employer_ID = None
        
        # update the lists of ids for each region, keep them ordered
        if region == 0:
            model.ids_region0.remove(unique_id)
            bisect.insort(model.ids_region1, unique_id)
        elif region == 1:
            model.ids_region1.remove(unique_id)
            bisect.insort(model.ids_region0, unique_id)

		# move the agent in the grid
		#model.grid.remove_agent(agent)
		#model.grid.move_agent(agent, (0, 1-region))

		
		#print("Household", unique_id,"migrated to region", 1-region)
        return 1-region
    return region




''' 
	Probability of migration for firms
	PARAMETERS
	D       : demand of firm in both regions [0,1]
	r       : firm's region
   average_profits_other_reg: prfits made on average by the same sector in the other region
	my_wage : wage offered by the firm
'''
def firms_migration_probability(demand_distance, r,   model,  w_1 = 0.5, w_2 = 0.5):
    
    prob_migration = 0

    profitability = model.datacollector.model_vars["Regional_profits_cons"][int(model.schedule.time)][r + 2]
    #gov = model.governments[0]
    #profitability = gov.net_sales_cons_firms[r + 2]
    if profitability < 0: 
          prob_migration =  1 - math.exp(w_1 * demand_distance  +   w_2 * profitability) 
    
    return prob_migration
   
  


'''
firm migration procedure
'''
def firm_migrate(mp, model, region, unique_id, employees_IDs, net_worth, wage, capital_vintage):
    firm = model.schedule._agents[unique_id]
    ##-- if the probability is higher than random number between 0,1, start procedure to leave the region--##
    if bernoulli.rvs(mp) == 1:
        ##--migration costs for each worker --## (need to add one for capital)
        migration_cost = len(employees_IDs) * wage 

        ##--- if migration is affordable --##
        if migration_cost < 0.75 * net_worth:
            ##--remove migration costs from ym net worth, fire employees and move out --##
            net_worth -= migration_cost
            if capital_vintage != 0:
                n = len(firm.capital_vintage)
                if n> 3:
                  firm.capital_vintage = capital_vintage[n//5:n]
           # migration_cost += capital_stock * model.transport_cost
            for id in employees_IDs:
                employee = model.schedule._agents[id]
                employee.employer_ID = None
            employees_IDs = []
            if region == 0:
                model.ids_region0.remove(unique_id)
                bisect.insort(model.ids_region1, unique_id)
            elif region == 1:
                model.ids_region1.remove(unique_id)
                bisect.insort(model.ids_region0, unique_id)
            
           # model.grid.move_agent(firm, (0, 1-region))
            firm.lifecycle = 0
            gov = model.governments[0]
            if firm.type == "Cons":
                wage = gov.salaries_cons[1 - region] #* 1.1
                firm.price = gov.av_price_cons[1 - region]  #* 1.1
        
            if firm.type == "Cap":
                wage = gov.salaries_cap[1 - region] # * 1.05
                firm.price = gov.av_price_cap[1 - region] # * 1.1
            
            
    
            return 1-region, employees_IDs, net_worth, wage
    
    return region, employees_IDs, net_worth, wage
