# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:46:57 2020

@author: TabernaA
"""
from mesa import Agent
import numpy as np
from sklearn import preprocessing
import math 
import random
from scipy.stats import beta
from model.classes.vintage import Vintage
import bisect

from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm


class Government (Agent):
    def __init__(self, unique_id, model, region):
        super().__init__(unique_id, model)
        
        
        self.type = "Gov"
        self.region = region
        self.fiscal_balance = 0
        self.fiscal_revenues = 0 
        self.unemployment_rate = 0.4
        self.unemployment_subsidy = 0
        self.tax_rate = 0.3
        self.unemployment_expenditure = 0
        self.market_shares_ids_cons_region0 = []
        self.market_shares_ids_cons_region1 = []
        self.market_shares_normalized = 0
        self.average_normalized_market_share = 0
        self.average_normalized_comp = [0,0]
        self.norm_price_unfilled_demand = 0
        #self.initial_minimum_wage = 0.01
        self.minimum_wage_region = 0.5
        self.local_ids = [a.unique_id for a in self.model.schedule.agents if a.region == self.region]

    def minimum_wage(self):
        if self.region == 0:
            average_salaries = average_regional_wage = self.model.datacollector.model_vars['Average_Salary'][int(self.model.schedule.time)]
            max_av_salary = max(average_salaries)
            self.minimum_wage_region = max_av_salary * (1 - 0.5)
            
        #r= self.region
        #current_average_productivty_my_region = self.model.datacollector.model_vars['Regiona_average_productivity'][int(self.model.schedule.time)][r]
       # previous_average_productivty_my_region = self.model.datacollector.model_vars["Regiona_average_productivity"][int(self.model.schedule.time) - 1 ][r]
        #if previous_average_productivity == 0:
         #   return
        #else:
         #   delta_unemployment = (current_unemployment_rate_my_region - previous_unemployment_rate_my_region) / previous_unemployment_rate_my_region
          #  self.minimum_wage_region[r] = self.minimum_wage_region[r] * (1 + 0.5 * delta_unemployment )
        

    def collect_taxes(self):
        self.fiscal_revenues = 0
        pos = (0, self.region)
        regional_population = self.model.grid.get_cell_list_contents([pos])
        for i in range(len(regional_population)):
            if (regional_population[i].type == "Cons" or regional_population[i].type == "Cap" and regional_population[i].profits > 0):
                self.fiscal_revenues += self.tax_rate * regional_population[i].profits
        
       # print( "I am gov of region ", self.region, " my fiscal revenus are", self.fiscal_revenues)
            
        
    def determine_subsidy(self):
        r = self.region
        regional_wage_my_region = self.model.datacollector.model_vars['Average_Salary'][int(self.model.schedule.time)][r]
        if self.model.schedule.time < 2:
            self.unemployment_subsidy = 0.3
        else:
            self.unemployment_subsidy = round(max( 0.0001, self.unemployment_rate * regional_wage_my_region), 3)
        regional_aggr_unemployment = self.model.datacollector.model_vars['Aggregate_Unemployment'][int(self.model.schedule.time)][r]
        self.unemployment_expenditure = regional_aggr_unemployment * self.unemployment_subsidy
        self.fiscal_balance += self.fiscal_revenues - self.unemployment_expenditure
        #print("my unemployment agg was", regional_aggr_employment, " my total expenses are ", self.unemployment_expenditure, "my balance is ", self.fiscal_balance)
    '''
    def normalizing_cons(self):
        agents = self.model.schedule._agents
        if self.region == 0:
            self.market_shares_ids_cons_region0 = []
            market_shares_region0 = []
            for i in range(len(agents)):
                if agents[i].type == "Cons":
                    self.market_shares_ids_cons_region0.append([agents[i].unique_id , agents[i].market_share[0]])
            for i in range(len(self.market_shares_ids_cons_region0)):
                market_shares_region0.append(self.market_shares_ids_cons_region0[i][1])
            print("market shares region 0 ",market_shares_region0)
            market_shares_region0_array = np.array(market_shares_region0)
            market_shares_region0_normalized = preprocessing.normalize([market_shares_region0_array])
            sum_shares = sum(market_shares_region0_normalized)
            print("the normalized sum is ", sum_shares)
            for i in range(len(self.market_shares_ids_cons_region0)):
                self.market_shares_ids_cons_region0[i][1] = market_shares_region0_normalized[0][i]
            print("market shares region 0 normalized ", self.market_shares_ids_cons_region0, "summed is equal to ", sum(self.market_shares_ids_cons_region0[1]))
                

        if self.region == 1:
            self.market_shares_ids_cons_region1 = []
            market_shares_region1 = []
            for i in range(len(agents)):
                if agents[i].type == "Cons":
                    self.market_shares_ids_cons_region1.append([agents[i].unique_id, agents[i].market_share[1] ])
''' 
    def market_share_normalizer_cons(self):
        if self.region == 0: 
            MS0 = [a.market_share[0] for a in self.model.schedule.agents if ( a.type == "Cons")]
            MS1 = [a.market_share[1] for a in self.model.schedule.agents if (a.type == "Cons")]
            ids = [a for a in self.model.ids_firms2]
            
            #ids = [a.unique_id for a in self.model.schedule.agents if (a.type == "Cons")]
            norm0 =  [ round(float(i)/(sum(MS0) + 0.0001) , 5) for i in MS0]
            norm1 =  [round(float(i)/(sum(MS1) + 0.0001), 5) for i in MS1] 
            self.average_normalized_market_share = round ((sum(norm0) + sum(norm1)) / len(ids) , 5)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids, norm_ms))
            self.market_shares_normalized = norm_ms_dict
            #print(self.market_shares_normalized)
            
            
            
    def market_share_normalizer_cap(self):
        if self.region == 0: 
            PM0 = [a.production_made for a in self.model.schedule.agents if ( a.type == "Cap")]
            #MS1 = [a.production_made for a in self.model.schedule.agents if (a.type == "Cons")]
            ids = [a for a in self.model.ids_firms1]
            #ids = [a.unique_id for a in self.model.schedule._agents if (a.type == "Cap")]
            norm0 = [float(i)/(sum(PM0) + 0.0001) for i in PM0]
            #norm1 = [float(i)/(sum(MS1) + 0.0001) for i in MS1]
            #self.average_normalized_market_share = (sum(norm0) + sum(norm1)) / len(ids)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            #norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids, norm0))
            self.market_shares_normalized_cap = norm_ms_dict
            #print(self.market_shares_normalized)
    
    def comp_norm(self):
        if self.region == 0: 
        
            COMP_all_0 = [ (1 + a.competitiveness[0])  for a in self.model.schedule.agents if (a.type == "Cons")]
            COMP_all_1 = [ (1 +  a.competitiveness[1]) for a in self.model.schedule.agents if (a.type == "Cons")]
            ids_all = [a.unique_id for a in self.model.schedule.agents if (a.type == "Cons")]
            
            #norm0 = [(float(i)-min(COMP_all_0))/(max(COMP_all_0 )-min(COMP_all_0) ) for i in COMP_all_0]
            #norm1 = [(float(i)-min(COMP_all_1))/(max(COMP_all_1 )-min(COMP_all_1) ) for i in COMP_all_1]
            norm0 = [ round(float(i)/(sum(COMP_all_0) + 0.0001), 5) for i in COMP_all_0]
            norm1 = [ round(float(i)/(sum(COMP_all_1) + 0.0001), 5) for i in COMP_all_1]
            norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids_all, norm_ms))

            self.comp_normalized = norm_ms_dict

        
            #C0 =  [ (1 + a.competitiveness[0]) * a.market_share[0] for a in self.model.schedule.agents if ( a.type == "Cons")]
            #C1 =  [ (1 + a.competitiveness[1]) * a.market_share[1] for a in self.model.schedule.agents if (a.type == "Cons")]
            #norm_region_0 = [round(float(i)/(sum(C0) + 0.0001), 5) for i in C0]
            #norm_region_1 = [round(float(i)/(sum(C1) + 0.0001), 5) for i in C1]
            #self.average_normalized_comp = [  round(sum(norm_region_0)/ (len(norm_region_0) + 0.0001), 6), round(sum(norm_region_1)/ (len(norm_region_0) + 0.0001),6)]
            C0 =  [ a.market_share[0] for a in self.model.schedule.agents if ( a.type == "Cons")]
            C1 =  [ a.market_share[1] for a in self.model.schedule.agents if (a.type == "Cons")]
            norm_region_0 = [ norm0[i] * C0[i] for i in range(len(C0))]
            norm_region_1 = [ norm1[i] * C1[i] for i in range(len(C1))]
            self.average_normalized_comp = [  round(sum(norm_region_0), 6), round(sum(norm_region_1),6)]

            #print(self.average_normalized_comp)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}

    
    
    def price_and_unfilled_demand_cons(self):
        if self.region == 0: 
            MS0 = [a.price for a in self.model.schedule.agents if (a.type == "Cons")] 
            #print(self.comp_normalized)rice for a in self.model.schedule.agents if ( a.type == "Cons")]
            MS1 = [a.unfilled_demand for a in self.model.schedule.agents if (a.type == "Cons")]
            ids = [a.unique_id for a in self.model.schedule.agents if (a.type == "Cons")]
            norm0 = [round(float(i)/(sum(MS0) + 0.0001) , 6) for i in MS0]
            norm1 = [round(float(i)/(sum(MS1) + 0.0001) , 6)for i in MS1]
            #self.average_normalized_market_share = (sum(norm0) + sum(norm1)) / len(ids)
            #print("the average normalized market share is ", self.average_normalized_market_share)
            # Make a dictionary of the form
            # {unique_id : (normalized ms 0, normalized ms 1)}
            norm_ms = zip(norm0, norm1)
            norm_ms_dict = dict(zip(ids, norm_ms))
            self.norm_price_unfilled_demand = norm_ms_dict


        

      
            #print(self.market_shares_normalized)
                
    def entry_exit_variable(self, o = 0.50, x1=-0.15, x2=0.15):
        # exit firms
        if self.region == 0: 
            all_firms = [a for a in self.model.schedule.agents if (a.type == "Cap" or a.type == "Cons") ]
            for firm in all_firms:
                if firm.lifecycle > 10:
                    if  sum(firm.market_share) < 0.00001 or firm.net_worth < 0:
                        print(firm.unique_id, "lifecyle ", firm.lifecycle, "time", self.model.schedule.time)
                        self.exit_firm(firm)
            for i in range(1):
                region = i
        # entry firms
        # sectoral liquidity-to-debt ratios for this and the previous period
                #LTD_cur = self.sectoral_liquidity_to_debt_ratio(int(self.model.schedule.time), region)
                #LTD_prev = self.sectoral_liquidity_to_debt_ratio(int(self.model.schedule.time)-1, region )
                PR_cur =  self.sectoral_profit(int(self.model.schedule.time), region)
                PR_prev = self.sectoral_profit(int(self.model.schedule.time)-1, region )
                
        # financial attractiveness of each sector
                DP_cap = min( x2 , PR_cur[0] - PR_prev[0])
                DP_cons = min( x2, PR_cur[1] - PR_prev[1])
                print("region ", region, "prvious prof cap, cons  ", PR_prev, "current prof  cap, cons", PR_cur, "delta profits cap, cons", DP_cap, DP_cons )
        # determine number of entrants in each sector
                num_firms1 = len([a for a in self.model.schedule.agents if a.type=="Cap" and a.region == region])
                num_firms2 = len([a for a in self.model.schedule.agents if a.type=="Cons" and a.region == region])
                if DP_cap <= 0:
                    entrants_cap = 0
                else:
                    entrants_cap = math.floor(num_firms1 * ((1-o)*DP_cap + o*random.uniform(x1, x2)))
                if DP_cons <= 0:
                    entrants_cons= 0
                else:
                    entrants_cons = math.floor(num_firms2 * ((1-o)*DP_cons + o*random.uniform(x1, x2)))
                print("entrant cons", entrants_cons, "entrant cap", entrants_cap, "region ", region)
        # entry of firms from each sector
                for j in range(entrants_cap):
                    self.enter_firm("Cap", region)

                for j in range( entrants_cons):
                    self.enter_firm("Cons", region)
            
            
                self.model.num_agents += entrants_cap + entrants_cons
    

    def enter_firm(self, sector, region):

        ids = list(self.model.schedule._agents.keys())
        max_id = max(ids)
        u_id = max_id + 1
        r = region
        time = int(self.model.schedule.time)
        average_regional_NW  = self.model.datacollector.model_vars['Sectoral_liquid_assets'][time][r]
        regional_wage = self.model.datacollector.model_vars['Average_Salary'][time][r]
        print(sector, r)
        if sector == "Cap":
            firm = CapitalGoodFirm(u_id, self.model)
            self.model.ids_firms1.append(u_id)

            current_average_price =  self.model.datacollector.model_vars["Capital_price_average"][time][r]
            
            firm.net_worth = average_regional_NW[0] * 0.9
            firm.wage = regional_wage * (1 + 0.1)
            prod0 = [a.productivity[0] for a in self.model.schedule.agents if (a.type == "Cap")]
            prod1 = [a.productivity[1] for a in self.model.schedule.agents if (a.type == "Cap")]
                        #print(prod0)
            firm.productivity[0] = max(prod0) * (1 - 0.15 + beta.rvs(3,3)*(0.15 + 0.15)) 
            firm.productivity[1] = max(prod1) * (1 - 0.15 + beta.rvs(3,3)*(0.15 + 0.15)) 
            firm.price = current_average_price
                        #max_prod1 = max(prod1)
            print("new entry prod id", firm.unique_id)
        elif sector == "Cons":
            firm = ConsumptionGoodFirm(u_id, self.model)
            self.model.ids_firms2.append(u_id)
            
            average_regional_NW = average_regional_NW[1]
            average_regional_capital =  self.model.datacollector.model_vars["Capital_Regional"][time][r]
            average_regional_cons_firm =  self.model.datacollector.model_vars["Population_Regional_Cons_Firms"][time][r]
            average_regional_cons=  self.model.datacollector.model_vars["CONSUMPTION"][time][r]
            current_average_price =  self.model.datacollector.model_vars["Cosumption_price_average"][time][r]
            
            
            firm_capital_amount = 0.9 * (average_regional_capital / average_regional_cons_firm)
            firm.capital_vintage = []
            supplier_id = random.choice(self.model.ids_firms1)
            print(supplier_id)
            supplier = self.model.schedule.agents[supplier_id]
            for i in range(math.ceil(firm_capital_amount)):
                firm.capital_vintage.append(Vintage(prod=round(supplier.productivity[0]), amount=1) )
                    
            firm.net_worth = average_regional_NW * 0.9
            firm.credit_rationed = False
            firm.price = current_average_price 
            firm.markup = 0.3
            firm.productivity = supplier.productivity
            firm.wage = regional_wage
            firm.normalized_price += 0.001                    #to avoid Nan when calculating competitiveness
            firm.competitiveness = [ round(self.average_normalized_comp[0] * 0.9 , 5), round( self.average_normalized_comp[1] * 0.9, 5)] 
            #if self.average_normalized_market_share < 0.0001:
            #firm.market_share = [0.01, 0.01]
            #firm.market_share = [0.01, 0.01, 0.1]
            #else:
            firm.market_share = [self.average_normalized_market_share * 0.9, self.average_normalized_market_share * 0.9]
            firm.market_share_history = [ sum(firm.market_share), sum(firm.market_share), sum(firm.market_share)]
           # print(firm.market_share_history)
            firm.past_demands = [ average_regional_cons * self.average_normalized_market_share,  average_regional_cons * self.average_normalized_market_share,  average_regional_cons * self.average_normalized_market_share  ]
            firm.competitiveness = self.average_normalized_comp
                        
            firm.real_demand = firm.past_demands[-1]
            firm.unfilled_demand = 0
            firm.invetories = 0
            firm.investment_cost = 0
            firm.debt = 0
            #firm.market_share_history[ 0.9 * self.average_normalized_market_share, 0.9 * self.average_normalized_market_share]      # the firms gets a fraction (0.9) of the avergae
            print("new entri cons!", firm.unique_id)
         
        y = r
        self.model.grid.place_agent(firm, (0,y))
        self.model.schedule.add(firm)
        firm.region = y
        if y == 0:
            bisect.insort(firm.model.ids_region0, firm.unique_id)
        elif y == 1:
            bisect.insort(firm.model.ids_region1, firm.unique_id)
        firm.lifecycle = 0
                           

    

                        

                        
                       


                    
    
    def exit_firm(self, firm):
        # fire employees
        for employee_id in firm.employees_IDs:
            employee = self.model.schedule.agents[employee_id]
            employee.employer_ID = None

        # remove offers from clients
        if firm.type == "Cap" and firm.net_worth < 0:
            for client_id in firm.client_IDs:
                #print("Cap firm", firm.unique_id,"exiting and removing client", client_id)
                
                client = self.model.schedule.agents[client_id]
                
                if firm.brochure_regional in client.offers:
                    #print("removing regional brochure")
                    client.offers.remove(firm.brochure_regional)
                elif firm.brochure_export in client.offers:
                    #print("removing export brochure")
                    client.offers.remove(firm.brochure_export)
                #else:
                    #print("client had no brochure to remove")
                client.supplier_id = None
                self.model.ids_firms1.remove(firm.unique_id)
                
            
        
        if firm.type == "Cons":
            for offer in firm.offers:
                supplier_id = offer[2]
                supplier = self.model.schedule.agents[supplier_id]
                if firm.unique_id in supplier.client_IDs:
                    supplier.client_IDs.remove(firm.unique_id)
                    #print(firm.unique_id, firm.type)
            self.model.ids_firms2.remove(firm.unique_id)
        self.model.num_agents -= 1
        self.model.grid._remove_agent(firm.pos, firm)
        self.model.schedule.remove(firm)
        if firm.region == 0:
            self.model.ids_region0.remove(firm.unique_id)
        elif firm.region == 1:
            self.model.ids_region1.remove(firm.unique_id)
        else:
            print("something wrong with regions gov")
        print("[ENTRY/EXIT]Firm", firm.unique_id,  "type ", firm.type ,"exited!")

        # remove firm 

    '''   
    def sectoral_liquidity_to_debt_ratio(self, time, region):
        
        if time == int(self.model.schedule.time):
            
            LA_cap = [a.net_worth for a in self.model.schedule.agents if a.type == "Cap" and a.region == region]
            LA_cons = [a.net_worth for a in self.model.schedule.agents if a.type == "Cons" and a.region == region]
            liquid_assets = [sum(LA_cap), sum(LA_cons)]

            debt_cons = [a.debt for a in self.model.schedule.agents if a.type == "Cons" and a.region == region]
            debt = [0, sum(debt_cons)]
        else:
            liquid_assets = self.model.datacollector.model_vars['Sectoral_liquid_assets'][time]
            liquid_assets = liquid_assets[region]
            debt = self.model.datacollector.model_vars['Sectoral_debt'][time]
            debt = debt[region]

        # account for debt and liquid assets being <= 0
        # very ugly, need to think of a better way
        if debt[0] > 0 and liquid_assets[0] > 0:
            MC_cap = math.log(liquid_assets[0]) - math.log(debt[0])
        elif debt[0] <= 0 and liquid_assets[0] <= 0:
            MC_cap = 0
        elif debt[0] <= 0:
            MC_cap = math.log(liquid_assets[0])
        elif liquid_assets[0] <= 0:
            MC_cap = math.log(-math.log(debt[0]))

        if debt[1] > 0 and liquid_assets[1] > 0:
            MC_cons = math.log(liquid_assets[1]) - math.log(debt[1])
        elif debt[1] <= 0 and liquid_assets[1] <= 0:
            MC_cons = 0
        elif debt[1] <= 0:
            MC_cons = math.log(liquid_assets[1])
        elif liquid_assets[1] <= 0:
            MC_cons = math.log(-math.log(debt[1]))

        
        return [MC_cap, MC_cons] 
   ''' 

    def sectoral_profit(self, time, region):
        
        if time == int(self.model.schedule.time):
            
            PR_cap = [a.profits for a in self.model.schedule.agents if a.type == "Cap" and a.region == region]
            PR_cons = [a.profits for a in self.model.schedule.agents if a.type == "Cons" and a.region == region]
            profits_cons  = sum(PR_cons)
            profits_cap = sum(PR_cap)

            #debt_cons = [a.debt for a in self.model.schedule.agents if a.type == "Cons" and a.region == region]
            #debt = [0, sum(debt_cons)]
        else:
            profits_cons  = self.model.datacollector.model_vars['Regional_profits_cons'][time][region]
            
            profits_cap =  self.model.datacollector.model_vars['Regional_profits_cap'][time][region]


        # account for debt and liquid assets being <= 0
        # very ugly, need to think of a better way
        #print(profits_cons)
        if profits_cons <= 0:
            profits_cons = 0
        else:
            profits_cons = math.log(profits_cons)
        if profits_cap <= 0:
            profits_cap = 0
        else:
            profits_cap = math.log(profits_cap)
            


        
        return [profits_cap, profits_cons] 
    
    
    
    
    
    
    
    
    
    
    
    
    def entry_exit(self):
        # exit firms
        all_firms_cons = [a for a in self.model.schedule.agents if a.type == "Cons"]
        all_firms_cap = [a for a in self.model.schedule.agents if a.type == "Cap"]
        all_firms = [a for a in self.model.schedule.agents if a.type == "Cap" or a.type == "Cons"]
        for firm in all_firms:
            if firm.lifecycle > 10:
                if  sum(firm.market_share) < 0.00001 or firm.net_worth < 0:
                # fire employees
                    for employee_id in firm.employees_IDs:
                        employee = firm.model.schedule.agents[employee_id]
                        employee.employer_ID = None
                        firm.employees_IDs = []
                    
                    
                    r = firm.region
                    average_regional_NW = self.model.datacollector.model_vars["Regional_average_NW"][int(self.model.schedule.time)][r]
                    regional_wage = self.model.datacollector.model_vars['Average_Salary'][int(self.model.schedule.time)][r]
                    current_average_price =  self.model.datacollector.model_vars["Cosumption_price_average"][int(self.model.schedule.time)][r]
                    
                    if firm.type == "Cons":
                     #updating

                        
                        average_regional_capital =  self.model.datacollector.model_vars["Capital_Regional"][int(self.model.schedule.time)][r]
                        average_regional_cons_firm =  self.model.datacollector.model_vars["Population_Regional_Cons_Firms"][int(self.model.schedule.time)][r]
                        
                        average_regional_cons=  self.model.datacollector.model_vars["CONSUMPTION"][int(self.model.schedule.time)][r]
                        
                        
                        firm_capital_amount = 0.9 * (average_regional_capital / average_regional_cons_firm)
                        firm.capital_vintage = []
                        supplier_id = random.choice(self.model.ids_firms1)
                        supplier = self.model.schedule.agents[supplier_id]
                        for i in range(math.ceil(firm_capital_amount)):
                           firm.capital_vintage.append(Vintage(prod=round(supplier.productivity[0]), amount=1) )
                    
                        firm.net_worth = average_regional_NW * 0.8
                        firm.credit_rationed = False
                        firm.price = current_average_price + 0.00001
                        firm.markup = 0.3
                        firm.productivity = supplier.productivity
                        firm.wage = regional_wage
                        firm.normalized_price += 0.0001                    #to avoid Nan when calculating competitiveness
                        firm.competitiveness = [ round(self.average_normalized_comp[0]  , 5), round( self.average_normalized_comp[1] , 5)] 
                        if self.average_normalized_market_share < 0.0001:
                            firm.market_share = [0.01, 0.01]
                            firm.market_share = [0.01, 0.01, 0.1]
                        else:
                            firm.market_share = [self.average_normalized_market_share * 0.7, self.average_normalized_market_share * 0.7]
                            firm.market_share_history = [ sum(firm.market_share), sum(firm.market_share)] 
                        firm.past_demands = [ average_regional_cons * self.average_normalized_market_share,  average_regional_cons * self.average_normalized_market_share,  average_regional_cons * self.average_normalized_market_share  ]
                        firm.competitiveness = self.average_normalized_comp
                        firm.markup = 0.3
                        firm.real_demand =  round ((average_regional_cons * self.average_normalized_market_share) / firm.price , 4)
                        firm.unfilled_demand = 0
                        firm.invetories = 0
                        firm.investment_cost = 0
                        #firm.market_share_history[ 0.9 * self.average_normalized_market_share, 0.9 * self.average_normalized_market_share]      # the firms gets a fraction (0.9) of the avergae
                        print("new entri cons!", firm.unique_id)
                        
                       
                    if firm.type == "Cap" and firm.net_worth  < 0:
                        firm.net_worth = average_regional_NW * 0.9
                        firm.wage = regional_wage    
                        prod0 = [a.productivity[0] for a in self.model.schedule.agents if (a.type == "Cap")]
                        prod1 = [a.productivity[1] for a in self.model.schedule.agents if (a.type == "Cap")]
                        #print(prod0)
                        firm.productivity[0] = max(prod0) * (1 - 0.15 + beta.rvs(3,3)*(0.15 + 0.15)) 
                        firm.productivity[1] = max(prod1) * (1 - 0.15 + beta.rvs(3,3)*(0.15 + 0.15)) 
                        firm.price = current_average_price
                        #max_prod1 = max(prod1)
                        print("new entry prod")
                    firm.lifecycle = 0
                            
                '''
                
                    for offer in firm.offers:
                        supplier_id = offer[2]
                        supplier = self.model.schedule._agents[supplier_id]
                        supplier.client_IDs.remove(firm.unique_id) 
        
            
                  
                
                # remove offers from clients
                    if firm.type == "Cap":
                        for client_id in firm.client_IDs:
                        #print("Cap firm", firm.unique_id,"exiting and removing client", client_id)
                        client = self.model.schedule._agents[client_id]
                        if firm.brochure_regional in client.offers:
                            #print("removing regional brochure")
                            client.offers.remove(firm.brochure_regional)
                        elif firm.brochure_export in client.offers:
                            #print("removing export brochure")
                            client.offers.remove(firm.brochure_export)
                        #else:
                            #print("client had no brochure to remove")
                        client.supplier_id = None

                # remove clients from suppliers
                '''
        
                
            
    def step(self):
        print("my region  is ", self.region)
        if(self.model.schedule.time > 1):
           self.collect_taxes()
           self.determine_subsidy()
        
        
        
    '''
    ---------------------------------------------------------------------------------------------
                                      Stages for staged activation
                                  all stages together make up one step
    ---------------------------------------------------------------------------------------------
    '''

    def stage0(self):
        pass

    def stage1(self):
        self.minimum_wage()
        pass
        #self.collect_taxes()
        #self.determine_subsidy()

    def stage2(self):
        self.price_and_unfilled_demand_cons()
        pass
        #self.collect_taxes()
        #self.determine_subsidy()

    def stage3(self):
        self.comp_norm()
        self.market_share_normalizer_cap()

        #self.collect_taxes()
        #self.determine_subsidy()
    def stage4(self):
        self.market_share_normalizer_cons()


        

    def stage5(self):
        if self.model.schedule.time > 0:
            self.entry_exit()
            self.market_share_normalizer_cons()


        pass
        #self.collect_taxes()
        #self.determine_subsidy()

    def stage6(self):
        self.collect_taxes()
        self.determine_subsidy()
        #if self.model.schedule.time > 10 and self.model.schedule.time % 2 ==0:
