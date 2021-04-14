# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:46:57 2020

@author: TabernaA
"""
seed_value = 12345678
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed=seed_value)
from mesa import Agent
#import numpy as np
#from sklearn import preprocessing
import math 

from scipy.stats import beta
from model.classes.vintage import Vintage
#import bisect
#from scipy.stats import bernoulli

#from model.classes.capital_good_firm import CapitalGoodFirm
#from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.modules import migration as migration

class Government (Agent):
    def __init__(self, unique_id, model, region):
        super().__init__(unique_id, model)
        
        
        self.type = "Gov"
        self.region = region
        self.fiscal_balance = 0
        self.fiscal_revenues = 0 
        self.unemployment_sub =0.5
        self.unemployment_subsidy = [0,0]
        self.tax_rate = 0.2
        self.unemployment_expenditure = [0,0]
        self.market_shares_ids_cons_region0 = []
        self.market_shares_ids_cons_region1 = []
        self.market_shares_normalized = 0
        self.market_shares_normalized_cap = 0
        self.average_normalized_market_share = 0
        self.average_normalized_comp = [0,0,0]
        self.norm_price_unfilled_demand = 0
        #self.initial_minimum_wage = 0.01
        self.minimum_wage_region = 0.5
        self.average_wages =  [ 0, 0, 0, 0]
        self.net_sales_cons_firms = [ 0, 0, 0, 0]
        self.salaries_cap = [1,1]
        self.salaries_cons = [1,1]
        self.av_price_cons = [1,1]
        self.av_price_cap = [1,1]
        
        
        self.aggregate_cons = [ 0, 0, 0, 0]
        self.regional_pop_hous= [self.model.num_households/2, self.model.num_households /2]
        self.aggregate_unemployment = [0,0]
        self.aggregate_employment = [0,0]
        self.regional_av_prod = [ 0, 0, 0, 0]
        self.unemployment_rates =  [ 0, 0, 0, 0, 0]
        self.cap_av_prod = [ 0, 0, 0, 0]
        self.export_demand =  300
        self.export_demand_list = []
        self.fraction_exp = 0.01
        self.best_firm1 = [0,0]
        
        #self.open_vacancies = [ ]
        #self.local_ids = [a.unique_id for a in self.model.schedule.agents if a.region == self.region]

    def minimum_wage(self):
            average_salaries = self.average_wages
           # max_av_salary = max(average_salaries)
            fraction = 0.6
            higher_regional_wage = max(average_salaries) / 2
            self.minimum_wage_region = [max( 1, average_salaries[0] * fraction, higher_regional_wage ), max( 1, average_salaries[1] * fraction, higher_regional_wage)]
            #self.unemployment_subsidy = [ self.minimum_wage_region * 0.9, self.minimum_wage_region * 0.9]
            regional_wage_regions =  self.minimum_wage_region
            fraction_sub = 0.8
            
            #regional_aggr_unemployment =  self.aggregate_unemployment
            self.unemployment_subsidy = [max( 1, regional_wage_regions[0] * fraction_sub), max( 1, regional_wage_regions[1] * fraction_sub)]
            
        #r= self.region
        #current_average_productivty_my_region = self.model.datacollector.model_vars['Regiona_average_productivity'][int(self.model.schedule.time)][r]
       # previous_average_productivty_my_region = self.model.datacollector.model_vars["Regiona_average_productivity"][int(self.model.schedule.time) - 1 ][r]
        #if previous_average_productivity == 0:
         #   return
        #else:
         #   delta_unemployment = (current_unemployment_rate_my_region - previous_unemployment_rate_my_region) / previous_unemployment_rate_my_region
          #  self.minimum_wage_region[r] = self.minimum_wage_region[r] * (1 + 0.5 * delta_unemployment )
    def open_vacancies_list(self):
        
        cons_firms = self.model.firms_1_2
        r = self.region
        self.open_vacancies_cons = []
        for i in range(len(cons_firms)):
            if cons_firms[i].open_vacancies == True and cons_firms[i].region == r:
                self.open_vacancies_cons.append(cons_firms[i])
        '''
        cap_firms = self.model.firms1
        r = self.region
        self.open_vacancies_cap = []
        for i in range(len(cap_firms)):
            if cap_firms[i].open_vacancies == True and cap_firms[i].region == r:
                self.open_vacancies_cap.append(cap_firms[i])
        '''
                
        

     

    def collect_taxes(self):
        self.fiscal_revenues = 0
        pos = (0, self.region)
        regional_population = self.model.grid.get_cell_list_contents([pos])
        for i in range(len(regional_population)):
            if (regional_population[i].type == "Cons" or regional_population[i].type == "Cap" and regional_population[i].profits > 0):
                self.fiscal_revenues += self.tax_rate * regional_population[i].profits
        
       # print( "I am gov of region ", self.region, " my fiscal revenus are", self.fiscal_revenues)
            
        
    def determine_subsidy(self):
        regional_wage_regions =  self.average_wages
        fraction = 0.6
        #regional_aggr_unemployment =  self.aggregate_unemployment
        self.unemployment_subsidy = [max( 1, regional_wage_regions[0] * fraction), max( 1, regional_wage_regions[1] * fraction)]
            #print(self.unemployment_subsidy[i])
            #self.unemployment_expenditure[i] = regional_aggr_unemployment[i] * self.unemployment_subsidy[i]
           ## self.fiscal_balance[i] += self.fiscal_revenues[i] - self.unemployment_expenditure[i]
        #print("my unemployment agg was", regional_aggr_employment, " my total expenses are ", self.unemployment_expenditure, "my balance is ", self.fiscal_balance)

    def market_share_normalizer_cons(self):
        if self.region == 0: 
            MS0 = [a.market_share[0] for a in self.model.firms2 ]
            MS1 = [a.market_share[1] for a in self.model.firms2 ]
            MS2 = [a.market_share[2] for a in self.model.firms2 ]
            ids = self.model.ids_firms2
            
            #ids = [a.unique_id for a in self.model.schedule.agents if (a.type == "Cons")]
            norm0 =  [ round(float(i)/(sum(MS0) + 0.0001) , 5) for i in MS0]
            norm1 =  [round(float(i)/(sum(MS1) + 0.0001), 5) for i in MS1] 
            norm2 =  [round(float(i)/(sum(MS2) + 0.0001), 5) for i in MS2] 
            self.average_normalized_market_share = [round(sum(norm0) / len(ids) , 8), round(sum(norm1) / len(ids) , 8), round(sum(norm2) / len(ids) , 8)]
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            norm_ms = zip(norm0, norm1, norm2)
            norm_ms_dict = dict(zip(ids, norm_ms))
            self.market_shares_normalized = norm_ms_dict
            #print(self.market_shares_normalized)
            
            
            
    def market_share_normalizer_cap(self):
       # if self.region == 0: 
            MS0 = [a.real_demand_cap[0] for a in self.model.firms1 ]
            MS1 = [a.real_demand_cap[1] for a in self.model.firms1 ]
            ids = self.model.ids_firms1
            
            #ids = [a.unique_id for a in self.model.schedule.agents if (a.type == "Cons")]
            norm0 =  [ round(float(i)/(sum(MS0) + 0.0001) , 5) for i in MS0]
            norm1 =  [round(float(i)/(sum(MS1) + 0.0001), 5) for i in MS1] 
            #self.average_normalized_market_share = round ((sum(norm0) + sum(norm1)) / len(ids) , 5)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids, norm_ms))
            self.market_shares_normalized_cap = norm_ms_dict
            #print(self.market_shares_normalized)
    
    def comp_norm(self):
       # if self.region == 0: 
           
            NEG_COMP_all_0 = [ (a.competitiveness[0])  for a in self.model.firms2 ]
            min_0 = abs(min(NEG_COMP_all_0))  + 0.001
            COMP_all_0 = [ a + min_0  for a in NEG_COMP_all_0 ]
            NEG_COMP_all_1 = [ ( a.competitiveness[1]) for a in self.model.firms2 ]
            min_1 = abs(min(NEG_COMP_all_1))  + 0.001
            COMP_all_1 = [ a + min_1  for a in NEG_COMP_all_1 ]
            NEG_COMP_all_2 = [ ( a.competitiveness[2]) for a in self.model.firms2 ]
            min_2 = abs(min(NEG_COMP_all_2)) + 0.001
            COMP_all_2 = [ a + min_2 for a in NEG_COMP_all_2 ]
            ids_all =  [a.unique_id for a in self.model.firms2] 
            
            #norm0 = [(float(i)-min(COMP_all_0))/(max(COMP_all_0 )-min(COMP_all_0) ) for i in COMP_all_0]
            #norm1 = [(float(i)-min(COMP_all_1))/(max(COMP_all_1 )-min(COMP_all_1) ) for i in COMP_all_1]
            norm0 = [ round( float(i)/sum(COMP_all_0 ), 8) for i in COMP_all_0]
            norm1 = [ round(float(i)/sum(COMP_all_1), 8) for i in COMP_all_1]
            norm2 = [ round(float(i)/sum(COMP_all_2), 8) for i in COMP_all_2]
            norm_ms = zip(norm0, norm1, norm2)
            norm_ms_dict = dict(zip(ids_all, norm_ms))

            self.comp_normalized = norm_ms_dict

        
            C0 =  [ a.market_share[0] for a in self.model.firms2 ]
            C1 =  [ a.market_share[1] for a in self.model.firms2 ]
            C2 =  [ a.market_share[2] for a in self.model.firms2 ]
            norm_region_0 = [ norm0[i] * C0[i] for i in range(len(C0))]
            norm_region_1 = [ norm1[i] * C1[i] for i in range(len(C1))]
            norm_region_2 = [ norm2[i] * C2[i] for i in range(len(C2))]
            self.average_normalized_comp = [  round(sum(norm_region_0), 8), round(sum(norm_region_1),8), round(sum(norm_region_2), 8)]


    
    
    def price_and_unfilled_demand_cons(self):
        if self.region == 0: 
            MS0 = [a.price + 0.000001 for a in self.model.firms2 ] 
            #print(self.comp_normalized)rice for a in self.model.schedule.agents if ( a.type == "Cons")]
            MS1 = [a.unfilled_demand + 0.000001 for a in self.model.firms2 ]
            ids = self.model.ids_firms2 
            norm0 = [round( float(i)/sum(MS0) , 8) for i in MS0]
            norm1 = [round( float(i) /sum(MS1)  , 8) for i in MS1]
            #self.average_normalized_market_share = (sum(norm0) + sum(norm1)) / len(ids)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids, norm_ms))
            self.norm_price_unfilled_demand = norm_ms_dict


        

      
            #print(self.market_shares_normalized)


    
    def entry_exit_cap(self):
        all_firms_cap = self.model.firms1
        #all_firms = self.model.firms_1_2
        for firm_cap in all_firms_cap:
            if firm_cap.net_worth  <= 0  and firm_cap.production_made == 0:
               # if firm_cap.lifecycle > 10:
                    
                    ##----removing employees---#
                    for employee_id in firm_cap.employees_IDs:
                        employee = firm_cap.model.schedule.agents[employee_id]
                        employee.employer_ID = None
                    firm_cap.employees_IDs = [] 
                    ##----removing offers -----#
                    for client_id in firm_cap.client_IDs:
                        #print("Cap firm", firm.unique_id,"exiting and removing client", client_id)
                        client = self.model.schedule.agents[client_id]
                        if firm_cap.brochure_regional in client.offers:
                            #print("removing regional brochure")
                            client.offers.remove(firm_cap.brochure_regional)
                        elif firm_cap.brochure_export in client.offers:
                            #print("removing export brochure")
                            client.offers.remove(firm_cap.brochure_export)
                        #else:
                            #print("client had no brochure to remove")
                        if client.supplier_id == firm_cap.unique_id:
                            client.supplier_id == None
                    firm_cap.client_IDs = []
                        
                    firm_cap.production_made = 0
                    
                    r = firm_cap.region
                    #regional_inv =  self.model.datacollector.model_vars["INVESTMENT"][int(self.model.schedule.time)]
                    regional_wage =  self.salaries_cap
                    firm_cap.wage =  regional_wage[r]# * 1.1
                    #average_regional_prof =  self.model.datacollector.model_vars["Regional_profits_cons"][int(self.model.schedule.time)]
                    average_regional_NW = self.model.datacollector.model_vars["Regional_average_NW"][int(self.model.schedule.time)]
                    
                    firm_cap.net_worth = average_regional_NW[r] * 0.6
                    #inv_distance = (regional_inv[r] - regional_inv[1 - r]) /  max(regional_inv)
                    '''
                    inv_distance = 0 #  self.model.datacollector.model_vars["INVESTMENT"][int(self.model.schedule.time)][r + 3]
                    
                    mp = migration.firms_migration_probability( inv_distance, r, self.model, w_1 = 0, w_2 = 1  )
                    #firm_cap.region_history.append([mp, r])
                    firm_cap.region, firm_cap.employees_IDs, firm_cap.net_worth ,  firm_cap.wage= migration.firm_migrate(mp, firm_cap.model, r, firm_cap.unique_id, firm_cap.employees_IDs, firm_cap.net_worth, regional_wage[r],0)
                    '''
                    r = firm_cap.region
                   # current_average_price =  self.model.datacollector.model_vars["Capital_price_average"][int(self.model.schedule.time)][r]
                    
                        
                    top_prod_region = self.model.datacollector.model_vars["Top_prod"][int(self.model.schedule.time )][r]
                        #print(prod0)

                    a = (1 - 0.1 + beta.rvs(3,3)*(0.1)) 
                    firm_cap.productivity[0] = top_prod_region[0] * a 
                    firm_cap.productivity[1] = top_prod_region[1] * a
                    firm_cap.previous_productivity = firm_cap.productivity
                    #firm_cap.price = current_average_price
                        #max_prod1 = max(prod1)
                   #print("new entry prod", firm_cap.unique_id)

                    firm_cap.lifecycle = 0
        
        region0_cap_all = [  a.unique_id  for a in self.model.firms1 if  a.region == 0 ]
        
        if region0_cap_all != []:
        
            region0_cap = random.sample(region0_cap_all , math.ceil(len(region0_cap_all )/5))
        
            productivity_list = []
            for id_0 in region0_cap:
                agent_0 = self.model.schedule.agents[id_0]
                productivity_list.append(agent_0.productivity[0]/agent_0.price)
        
            best_i = productivity_list.index(max(productivity_list))
            self.best_firm1[0] =  self.model.schedule.agents[region0_cap[best_i]]
        
        region1_cap_all = [  a.unique_id for a in self.model.firms1 if  a.region == 1 ]
        if region1_cap_all != []:
        
            productivity_list = []
        
            region1_cap = random.sample(region1_cap_all , math.ceil(len(region1_cap_all )/5))
            for id_1 in region1_cap:
               agent_1 = self.model.schedule.agents[id_1]
               productivity_list.append(agent_1.productivity[0]/ agent_1.price)
       # if productivity_list != []:
               best_i = productivity_list.index(max(productivity_list))
               self.best_firm1[1] =  self.model.schedule.agents[region1_cap[best_i]]
          

    def entry_exit_cons(self):
        # exit firms
        all_firms_cons = self.model.firms2
        
        for firm in all_firms_cons:
            if  sum(firm.market_share) < 0.003 or  firm.labor_demand == 0: #firm.lifecycle > 6):
                if firm.lifecycle > 6:
                # fire employees
                    firm.bankrupt = len(firm.employees_IDs)
                    
                    for employee_id in firm.employees_IDs:
                        employee = firm.model.schedule.agents[employee_id]
                        employee.employer_ID = None
                    firm.employees_IDs = [] 
 
                    
                    r = firm.region
                    average_regional_NW = self.model.datacollector.model_vars["Regional_average_NW"][int(self.model.schedule.time)]
                    regional_wage = self.salaries_cons
                   
                    

                     #updating

                        
                   
    
                    #average_regional_cons_prof =  self.model.datacollector.model_vars["Regional_profits_cons"][int(self.model.schedule.time)]
                   # current_average_price =  self.model.datacollector.model_vars["Cosumption_price_average"][int(self.model.schedule.time)]
                  #  average_regional_cons=  self.aggregate_cons 
                    
                    firm.net_worth = average_regional_NW[r] * 0.5 
                    firm.credit_rationed = False
                    firm.capital_vintage = []
                    
                    
                    ###---location of the firm  ---###
                    #cons_distance = (average_regional_cons[r] - average_regional_cons[1 - r]) /  max(average_regional_cons)
                    '''
                    cons_distance =   0 #average_regional_cons[3 + r]
                    
                    
                    mp = migration.firms_migration_probability( cons_distance, r, firm.model , w_1 = 0, w_2 = 1)
                    #print("mp is", mp
                    #firm.region_history.append([mp, r])
                    firm.region, firm.employees_IDs, firm.net_worth,  firm.wage = migration.firm_migrate(mp, firm.model, r, firm.unique_id, firm.employees_IDs, firm.net_worth, regional_wage[r], 0) 
                    '''
                    r = firm.region
                    
                    if self.aggregate_employment[r] == r or self.unemployment_rates[r] == 1:
                        firm.region, firm.employees_IDs, firm.net_worth,  firm.wage = migration.firm_migrate(1, firm.model, r, firm.unique_id, firm.employees_IDs, firm.net_worth, regional_wage[r], 0) 
                        
                    
                    firm.capital_amount =  self.model.datacollector.model_vars["Capital_Regional"][int(self.model.schedule.time)][r + 2]
                    '''
                    supplier_region =[ a  for a in self.model.firms1 if  a.region == r]
                    if supplier_region != []:
                        supplier = random.choice(supplier_region)
                    else:
                        supplier = random.choice(self.model.firms1)
                    '''
                    supplier = self.best_firm1[r]
                    #supplier_id = random.choice(self.model.ids_firms1)

                    
                   #for i in range(math.ceil(firm_capital_amount)):
                    for i in range(4):
                        firm.capital_vintage.append(Vintage(prod=round(supplier.productivity[0]), amount= firm.capital_amount//4) )
                    # a = [[[1,5],[1,0]], ]
                    #print
                    
                   # firm.price = current_average_price[r] + 0.00001
                    firm.markup = 0.3
                    firm.productivity= [supplier.productivity[1], supplier.productivity[1]]   #, supplier.productivity[1]]
                    #firm.productivity[1] = supplier.productivity[1]
                    
                    firm.normalized_price += 0.00001                    #to avoid Nan when calculating competitiveness
                    firm.competitiveness = [ round(self.average_normalized_comp[0]  , 5), round( self.average_normalized_comp[1] , 5)] 
                    
                    #firm.market_share = [self.average_normalized_market_share[0] * 0.75, self.average_normalized_market_share[1] * 0.75, self.average_normalized_market_share[2] * 0.75]
                    #firm.market_share_history = [ sum(firm.market_share), sum(firm.market_share)] 
                    firm.market_share = [0, 0, 0]
                    firm.past_demands = []
                    #for i in range(3):
                     #   firm.past_demands.append( sum(average_regional_cons) * firm.market_share_history[0])
                    firm.competitiveness = self.average_normalized_comp
                    firm.markup = 0.3
                    firm.real_demand = 0 # firm.past_demands[1]
                    firm.unfilled_demand = 0
                    firm.invetories = 0
                    firm.offers = []
                    firm.wage = regional_wage[r] #* 1.1
                    firm.order_reduced = 0
                    #firm.investment_cost = 0
                    firm.market_share_history = []
                    firm.past_demands = []
                    firm.production_made = 0
                    
                        #firm.market_share_history[ 0.9 * self.average_normalized_market_share, 0.9 * self.average_normalized_market_share]      # the firms gets a fraction (0.9) of the avergae
                    
                    
                    firm.lifecycle = -1
              
                   # print("new entri cons!", firm.unique_id)
        
        
        
    def wage_and_cons_unempl(self): 
        
        

       ##---- WAGES ---#
        salaries = []
        salaries  = self.average_wage_regions(self.model.firms_1_2, True)
        self.salaries_cons  = self.average_wage_regions(self.model.firms2)
        self.salaries_cap  = self.average_wage_regions(self.model.firms1)
        av_sal0 = salaries[0] / self.av_price_cons[0]
        av_sal1 = salaries[1]/ self.av_price_cons[1]
        RAE0 = salaries[2]
        RAE1 = salaries[3]
        salary_diffrence0, salary_diffrence1 = self.variable_difference(av_sal0, av_sal1, True)
        self.average_wages =  [ salaries[0], salaries[1], salary_diffrence0, salary_diffrence1]

        ## ---- EMPLOYMENT --#
        
        ARU0, ARU1 = self.aggregate_unemployment_regions(self.model.households)
        self.aggregate_unemployment = [ARU0, ARU1]
        self.aggregate_employment = [RAE0, RAE1]
    
        ## --- UNEMPLOYMENT ---##
        
        regional_wage = self.average_wages 
        regional_unemployment_subsidy = self.unemployment_subsidy
       
        ## --- POPULATION ---##
         
        self.regional_pop_hous =  [ARU0 + RAE0, ARU1 + RAE1 ]
        
        unemployment_rate_0 =  round( max( 1 , ARU0)/ max( 1,self.regional_pop_hous[0]) , 2)
        unemployment_rate_1 =  round( max( 1 , ARU1)/ max(1 ,self.regional_pop_hous[1]) , 2)
        unemployment_rate_total = (ARU0 + ARU1) / sum(self.regional_pop_hous)
        
        unemployment_diffrence0 , unemployment_diffrence1 = self.variable_difference(unemployment_rate_0, unemployment_rate_1, False)
                    
        self.unemployment_rates = [ round(unemployment_rate_0, 2) ,  round(unemployment_rate_1, 2), unemployment_diffrence0, unemployment_diffrence1, unemployment_rate_total  ]
        
         ## --- CONSUMPTION ---##
        
        C0 = (regional_wage[0] * RAE0)  + (ARU0 *  regional_unemployment_subsidy[0])
        C1 = (regional_wage[1] * RAE1)  + (ARU1 *  regional_unemployment_subsidy[1])
        
        exp_share = self.model.datacollector.model_vars[ "Regional_sum_market_share"][int(self.model.schedule.time)]
        self.export_demand = self.export_demand * ( 1 + self.fraction_exp) # ( C0 + C1 )  * self.fraction_exp 
        export_demand = self.export_demand
        #print( exp_share)
       # self.export_demand_list.append(self.export_demand)
        C0 +=  round( export_demand  * exp_share[3] , 3)
        C1 += round( export_demand *  exp_share[4] , 3)
        
        if self.model.S > 0: 
            if int(self.model.schedule.time) - 1 == self.model.shock_time:
        
               # print('cut cons')
                shock = self.model.S #np.random.beta(self.model.beta_a,self.model.beta_b)
                C0 = (1 - shock) * C0
        
       # cons_diffrence0, cons_diffrence1 = self.variable_difference(C0, C1, True)    
        self.aggregate_cons = [round(C0, 3), round(C1, 3), C0+C1,  export_demand] #,  old_exp_C * ( 1 + constant_g)]
        
        
        
    def variable_difference(self, variable0, variable1, higher_foreign_values_more_favorble):
        #variable_difference = abs(variable0 - variable1)
        #variable_difference_perc = 0
        variable_difference0 = 0
        variable_difference1 = 0
        
        if higher_foreign_values_more_favorble == True:
            
            if variable1 > variable0:
                variable_difference0 = ( variable0 - variable1) / (variable0 + 0.001)
            if variable0 > variable1:
                variable_difference1 = (variable1 - variable0) / (variable1 + 0.001)
        
        if higher_foreign_values_more_favorble == False:
            
            if variable1 < variable0:
                variable_difference0 = (variable1 - variable0) / (variable1 + 0.001)
            
            if variable0 < variable1:
                variable_difference1 = (variable0 - variable1) / (variable1 + 0.001)
        
        
        #if variable_difference > 0.1 :
           #  variable_difference_perc  = variable_difference / (variable0 + variable1) /2 
        '''
        if variable_difference > 0.01:
             variable_diffrence0  =  round( (variable0 - variable1) / max(variable0 , variable1), 2)
             variable_diffrence1  =  round( (variable1 - variable0) /max(variable0 , variable1), 2)
        '''
        
        return round( max( -0.4 ,variable_difference0 ) ,3),round( max( -0.4 ,variable_difference1), 3)
    
   
    def produtivity_calculation(self):
        '''
        AE0 = 0
        AE1 = 0
        firms = self.model.firms2
        for i in range(len(firms)):
            i = firms[i]
            if i.region == 0:
                AE0 += len(i.employees_IDs)
            if i.region == 1:
                AE1 += len(i.employees_IDs)
                
        AEs = [AE0, AE1]
        '''
        
            
            
        
        self.regional_av_prod = self.productivity_firms(self.model.firms_1_2, self.aggregate_employment)
        #self.cap_av_prod = self.productivity_firms(self.model.firms1)
   
    
   
    def prices_calculation(self):
        
        self.av_price_cons = self.price_average(self.model.firms2)
        self.av_price_cap = self.price_average(self.model.firms1)
        #prices_cons = self.av_price_cons
        #av_norm_price_reg_0 = prices_cons[0] /sum(prices_cons)
        #av_norm_price_reg_1 = prices_cons[1] /sum(prices_cons)
        #comp_0 = -1 * av_norm_price_reg_0
        #comp_1 = -1 * av_norm_price_reg_0 * ( 1 + self.model.transport_cost)
        
        
        
        
    def sales_firms(self):
   
        sales0 = 0
        sales1 = 0
        firms = self.model.firms2
        for i in range(len(firms)):
            if firms[i].region == 0:
                sales0 += (firms[i].sales - firms[i].total_costs)
            if firms[i].region == 1:
                sales1 += (firms[i].sales - firms[i].total_costs)
     #  sales_firms2.append(firms[i].sales)
        if sales0 > 0:
            sales0 = round( math.log(sales0), 3)
        else:
            sales0 = 0
        if sales1 > 0:
            sales1 = round( math.log(sales1), 3)
        else:
            sales1 = 0
        
       # sales_diff0, sales_diff1 = self.variable_difference(sales0, sales1, True)
        self.net_sales_cons_firms = [sales0, sales1] #, sales_diff0, sales_diff1]


    '''
    HERE SOME FUNCTIONS USED TO MAKE CALCULATION WITHIN THE STEPS
    '''
    
    def price_average(self, agents_class):
        
        price0 = 0
        price1 = 0

        agents = agents_class
        for i in range(len(agents)):
            a = agents[i]
        #if a.type == "Cons":
            price0 += a.price * a.market_share[0]
            price1 += a.price * a.market_share[1]
                #firms1 += a.quantity_mad
        return [ abs(round(price0, 5)) , abs( round(price1, 5))]#
    #print("Average cons price con MS, " , [ round(price0, 5) , round(price1, 5)])
        
    
    def average_wage_regions(self , agents_class,   aggregate = False):
        var_0 = 0
        den_0 = 0
        av_var_0 = 0
        var_1 = 0
        den_1 = 0
        av_var_1 = 0   
        agents = agents_class
        for i in range(len(agents)):
            agent= agents[i]

            if agent.region == 0:
                var_0 += agent.wage * len(agent.employees_IDs)
                den_0 += len(agent.employees_IDs)
                
            if agent.region == 1:
                var_1 += agent.wage * len(agent.employees_IDs)
                den_1 += len(agent.employees_IDs)
        
        if den_0 !=0:
            av_var_0 =  round( var_0 /den_0 , 2)
            
        if den_1 !=0:
            av_var_1 =  round( var_1 /den_1 , 2)
        
        return [av_var_0 , av_var_1] if  aggregate == False else [av_var_0, av_var_1, den_0, den_1]
                 
            
    def aggregate_unemployment_regions(self, agents_class):
        ARE0 = 0 
        ARE1 = 0
        households = agents_class #model.schedule.agents
        for j in range(len(households)):
            if households[j].employer_ID == None:
                if households[j].region == 0:
                    ARE0 += 1
                elif households[j].region == 1:
                    ARE1 += 1
        return ARE0, ARE1
                
                # remove clients from suppliers
    
    
    def productivity_firms(self, agents_class, AE):
        productivity0 = 0
        firms0 = AE[0]
        productivity1 = 0
        firms1 = AE[1]
        a = agents_class

        #if int(model.schedule.time) > 0:
        productivity_old =  self.model.datacollector.model_vars["Regional_average_productivity"][int(self.model.schedule.time) ]
        productivity0_old = max( 1 ,productivity_old[0])
        productivity1_old = max( 1, productivity_old[1])

    #model.schedule.agents
        for i in range(len(a)):
        #if type(a[i]) is CapitalGoodFirm:
            if a[i].region == 0:
                productivity0 +=  a[i].production_made #* firm.price# a[i].productivity[1] * len(a[i].employees_IDs) #a[i].market_share[0]
               # firms0 += 1 # len(a[i].employees_IDs)
            if a[i].region == 1:
                productivity1 += a[i].production_made # * firm.price #a[i].productivity[1]  * len(a[i].employees_IDs) #*  sum(a[i].market_share)
#                firms1 +=  1 #len(a[i].employees_IDs)
            
        if firms0 != 0:
            av_prod0 = round( productivity0/firms0 , 3)
   
        else:
            av_prod0 = productivity0_old
       
    
        if firms1 != 0:
            av_prod1 = round(productivity1/firms1 , 3)
        else:
            av_prod1 = productivity1_old
        
        prod_increase_0 = max( -0.5 , min( 0.5 , round( (av_prod0 - productivity0_old)/ productivity0_old, 3)))
        prod_increase_1 = max( -0.5, min( 0.5 , round( (av_prod1 - productivity1_old)/ productivity1_old , 3)))
        
        return [av_prod0, av_prod1,  prod_increase_0, prod_increase_1 ]
        
    
    def climate_damages(self):
         self.flooded = False
         if self.region == 0 and (int(self.model.schedule.time) == self.model.shock_time or int(self.model.schedule.time) == self.model.shock_time + 100):
             self.flooded == True 
             print('gov flood')
    '''
    def climate_damages(self):
        #self.flooded = False
        if self.region == 0 and int(self.model.schedule.time) == self.model.shock_time:
            firms = self.model.firms_1_2
            print('flooding all the firms')
            for i in range(len(firms)):
                if firms[i].region == 0:
                    firms[i].flooded = True
                    
        if self.region == 0 and int(self.model.schedule.time) == self.model.shock_time + 1:
            firms = self.model.firms_1_2
            print('remove flooding from all the firms')
            for i in range(len(firms)):
                if firms[i].flooded == True:
                    firms[i].flooded = False
    '''         
            
            
            
    
    '''
    
    STAGES 
    
    '''
    
    
    def stage0(self):
        #print('gov')
        if self.region == 0:
           self.minimum_wage()
           if int(self.model.schedule.time) == 10:
                self.model.transport_cost = self.model.transport_cost / 10
                self.model.transport_cost_RoW =  self.model.transport_cost_RoW / 10
                '''
                agents = self.model.firms2
               
                for i in range(len(agents)):
                    agent = agents[i]
                    agent.market_share[2] = 1/len(agents)
                '''
                print('we')
           
        
        pass
        #self.collect_taxes()
        #self.determine_subsidy()
        
    def stage1(self):
        self.open_vacancies_list()
        
        pass

    def stage2(self):
        if self.region == 0:
            self.price_and_unfilled_demand_cons()
            
        pass
        #self.collect_taxes()
        #self.determine_subsidy()

    def stage3(self):
        if self.region == 0:
           self.comp_norm()
           self.market_share_normalizer_cap()
           self.prices_calculation()
           self.wage_and_cons_unempl()
           self.produtivity_calculation()
           
          # self.determine_subsidy()
        

        #self.collect_taxes()
        #self.determine_subsidy()
    def stage4(self):
        self.market_share_normalizer_cons()


        

    def stage5(self):
        
        if  self.region == 0:
            self.sales_firms()
            self.entry_exit_cons()
            self.market_share_normalizer_cons()
            #if self.model.schedule.time > 75:
            self.entry_exit_cap()
            self.market_share_normalizer_cap()
        
        pass
        


        pass
        #self.collect_taxes()
        #self.determine_subsidy()

    def stage6(self):
       # self.climate_damages()
        pass
       # self.collect_taxes()
        
        #if self.model.schedule.time > 10 and self.model.schedule.time % 2 ==0:
