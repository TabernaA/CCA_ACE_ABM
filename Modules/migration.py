# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:31:50 2020
@author: TabernaA
modules/migration.py
"""

import math 
import random
import bisect



'''
Calculation of households probability to migrate
'''
def households_migration_probability(region, model, my_income, w_1 = 0.5, w_2 = 0.5, mc = 0.05): 
    ##--retrieve relevant parameters --#
    unemployment = model.datacollector.model_vars['Unemployment_Regional'][int(model.schedule.time)]
    average_wage = model.datacollector.model_vars['Average_Salary'][int(model.schedule.time)]
    current_average_price = model.datacollector.model_vars["Cosumption_price_average"][int(model.schedule.time)]
    real_income =round( my_income / (current_average_price[region] + 0.0001), 5)
    real_income_exp = round(average_wage[1-region]/( current_average_price[ 1 - region] + 0.0001),5)
                               
    #rint("unemployemt", )
    ##---calculation of probability --##
    prob_migration = 1-math.exp(w_1* (  real_income - real_income_exp ) / real_income +
                                w_2*(unemployment[1-region] + 0.01 - unemployment[region]) / (unemployment[1-region] + 0.001))
    #prob_migration = prob_migration * (1 - mc)
    #print("H region ", region, "real income" , real_income, " real_income exp", real_income_exp, "unemployment rate", unemployment[region],
     #     "Unemployement ex", unemployment[1 - region], " probability", prob_migration)
    
    return prob_migration if prob_migration > 0 else 0




'''
households migration procedure
'''
def household_migrate(mp, model, region, unique_id):
    ##-- if the probability is higher than random number between 0,1, leave the job and move out --##
	agent = model.schedule._agents[unique_id]
	if random.uniform(0,1) < mp:
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
		model.grid.move_agent(agent, (0, 1-region))

		
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
def firms_migration_probability(D, r, my_wage, profits, model, w_1 = 0.4, w_2 = 0.3, w_3 = 0.4, mc = 0.05):
    
   ##---collect relevant variables for my time step ---##
    time = int(model.schedule.time)
    average_foreign_wage = model.datacollector.model_vars['Average_Salary'][time][1-r]
    regional_unemployment_subsidy = model.datacollector.model_vars["Regional_unemployment_subsidy"][time][r]
    minimum_wage = model.datacollector.model_vars["Minimum_wage"][time ]
    if average_foreign_wage <= minimum_wage:
        average_foreign_wage = minimum_wage#preventing division by zero
    D[r] += 0.00001 
    
    average_profits_other_reg = round( model.datacollector.model_vars['Regional_average_profits_cons'][int(model.schedule.time)][1-r])
    #comp = model.datacollector.model_vars['Competitiveness_Regional'][int(model.schedule.time)]
    #profits = min(0 , profits)
    if average_profits_other_reg  <= 0: 
        delta_profits = 0
    else:
        if profits <= 0:
            profits = 0.0001
        delta_profits = (profits - average_profits_other_reg) / profits
 
  
    demand_distance = (D[r] - D[1-r]) / D[r]
    wage_distance = (average_foreign_wage   - my_wage) / (average_foreign_wage ) 

    prob_migration =  1 - math.exp(w_1 * demand_distance  +  w_2* wage_distance +  w_3 * delta_profits)
    #prob_migration = prob_migration * (1 - mc)
    #print("F cons region ", r,  "my demand", D, " my wage " , my_wage, " avergae wage exp ", average_foreign_wage, " my profits ", profits, "prob migr", prob_migration)
     #     "profits exp ", foreign_average_profits, " profitability", delta_profits, "prob migration ", prob_migration)
	 #prob_migration =  1 - math.exp(w_1 * (D[r] - D[1-r]) / D[r] +  w_2*(average_foreign_wage - my_wage) / average_foreign_wage)
    return prob_migration if prob_migration > 0 else 0

'''
could merge with other, it is the same they just have a differend calcualtion for the demand (different market share)
'''

def cap_firms_migration_probability(D, r, my_wage,profits, model, unemployment_subsidy, w_1 = 0.4, w_2 = 0.3, w_3 = 0.3, mc = 0.05):

    average_foreign_wage = model.datacollector.model_vars['Average_Salary_Capital'][int(model.schedule.time)][1-r]
    regional_unemployment_subsidy = model.datacollector.model_vars["Regional_unemployment_subsidy"][int(model.schedule.time)][r]
    minimum_wage = model.datacollector.model_vars["Minimum_wage"][int(model.schedule.time) ]
    if average_foreign_wage <= minimum_wage:
        average_foreign_wage = minimum_wage#preventing division by zero
    wage_distance = (average_foreign_wage - my_wage ) / (average_foreign_wage )
    #comp = model.datacollector.model_vars['Competitiveness_Regional'][int(model.schedule.time)]
    average_profits_other_reg = round( model.datacollector.model_vars['Regional_average_profits_cap'][int(model.schedule.time)][1-r])
    #comp = model.datacollector.model_vars['Competitiveness_Regional'][int(model.schedule.time)]
    #profits = min(0 , profits)
    if average_profits_other_reg <= 0: 
        delta_profits = 0
    else:
        if profits <= 0:
            profits = 0.0001
        delta_profits = (profits -  average_profits_other_reg) / profits
        
    #profits = min( 0, profits)
    prob_migration = 1 - math.exp(w_1 * D + w_2* wage_distance + w_3 * delta_profits)
    #prob_migration = prob_migration * (1 - mc)
    '''
    print(   "  my demand distance is ", \
          D," my wage is ", my_wage, " average foreign wage is  ", average_foreign_wage, \
              " this my wage distance is ", wage_distance, "final prob is ", prob_migration )
    '''
    return prob_migration if prob_migration > 0 else 0



'''
firm migration procedure
'''
def firm_migrate(mp, model, region, unique_id, employees_IDs, net_worth, wage):
    firm = model.schedule._agents[unique_id]
    ##-- if the probability is higher than random number between 0,1, start procedure to leave the region--##
    if random.uniform(0,1) < mp:
        ##--migration costs for each worker --## (need to add one for capital)
        migration_cost = len(employees_IDs) * wage 
        ##--- if migration is affordable --##
        if migration_cost < 0.5 * net_worth:
            ##--remove migration costs from ym net worth, fire employees and move out --##
            net_worth -= migration_cost
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
            
            model.grid.move_agent(firm, (0, 1-region))
            
    
            return 1-region, employees_IDs, net_worth
    
    return region, employees_IDs, net_worth
