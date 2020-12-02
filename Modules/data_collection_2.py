# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:40:14 2020
@author: TabernaA
"""
# A file for storing data collection functions
# model/modules/data_collection.py
from scipy.stats import beta 
import numpy as np
from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.classes.household import Household

def gdp_cap(model):
    GDP0 = 0
    GDP1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cap":
            if firm.region == 0:
               GDP0 += firm.production_made * firm.price
            elif firm.region == 1:
                GDP1 += firm.production_made * firm.price
    return[ round(GDP0, 3), round(GDP1, 3), GDP0 + GDP1]



def RD_CCA_investment(model):
    RD_CCA = 0 
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cap" or firm.type == "Cons"  :
            if firm.region == 0:
               RD_CCA += firm.CCA_RD_budget
    return round(RD_CCA)




def RD_coefficient_average(model):
    RD0= 0 
    RD1= 0 
    firms = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cap" or firm.type == "Cons"  :
            if firm.region == 0:
               RD0 += firm.CCA_resilience[1]
               RD1 += firm.CCA_resilience[1]
               firms += 1
               
    return[RD0/ firms , RD1 / firms]

def gdp_cons(model):
    GDP0 = 0
    GDP1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons":
            if firm.region == 0:
               GDP0 += firm.price * firm.production_made
            elif firm.region == 1:
                GDP1 += firm.price * firm.production_made
    return[ round(GDP0, 3), round(GDP1, 3), GDP0 + GDP1]

def RD_total(model):
    RD0 = 0
    RD1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cap":
            if firm.region == 0:
               RD0 += firm.RD_budget
            elif firm.region == 1:
               RD1 += firm.RD_budget
    return[ round(RD0, 5), round(RD1, 5), RD0 + RD1]



    

def gdp(model):
    GDP0 = 0
    GDP1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons" or firm.type == "Cap":
            if firm.region == 0:
               GDP0 += firm.price * firm.production_made
            elif firm.region == 1:
                GDP1 += firm.price * firm.production_made
    return[ round(GDP0, 3), round(GDP1, 3), GDP0 + GDP1]
                
def investment(model):
    I0 = 0
    I1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons":
            if firm.region == 0:
                I0 += firm.investment_cost
            elif firm.region == 1:
                I1 += firm.investment_cost
    return[ round(I0, 3), round(I1, 3), I0 + I1]

def inventories(model):
    INV0 = 0
    INV1 = 0
    inventories_evolution = [[0,0],[0,0]]
    agents = model.schedule._agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons":
            if firm.region == 0:
                INV0 += firm.inventories * firm.price
            elif firm.region == 1:
                INV0 += firm.inventories * firm.price
    inventories_evolution.append([INV0,INV1])
    delta_inventories0 = inventories_evolution[-2][0] - inventories_evolution[-1][0]
    delta_inventories1 = inventories_evolution[-2][1] - inventories_evolution[-1][1]
    return[ delta_inventories0, delta_inventories1 , delta_inventories0 +  delta_inventories1  ]





def consumption(model):
    regional_wage = model.datacollector.model_vars['Average_Salary'][int(model.schedule.time)]
    aggr_employment =model.datacollector.model_vars['Aggregate_Employment'][int(model.schedule.time)]
    aggr_unemployment =model.datacollector.model_vars["Aggregate_Unemployment"][int(model.schedule.time)] 
    regional_unemployment_subsidy =model.datacollector.model_vars['Regional_unemployment_subsidy'][int(model.schedule.time)]
    C0 = regional_wage[0] * aggr_employment[0]  + aggr_unemployment[0] *  regional_unemployment_subsidy[0]
    C1 = regional_wage[1] * aggr_employment[1]  + aggr_unemployment[1] *  regional_unemployment_subsidy[1]
    #print("Aggregare CONS  regions 0,1:", [C0, C1])
    return [round(C0, 3), round(C1, 3), C0+C1]


def consumption_labor_check(model):
    HE0 = 0
    HE1 = 0
    CAE0 = 0 
    CAE1 = 0
    COE0 = 0
    COE1 = 0
    
    agents = model.schedule.agents
    
    for i in range(len(agents)):
        if agents[i].type == "Household":
            if agents[i].employer_ID != None:
                if agents[i].region == 0:
                   HE0 += 1
                elif agents[i].region == 1:
                   HE1 += 1
        if agents[i].type == "Cons" :
            if agents[i].region == 0:
                 COE0 += len(agents[i].employees_IDs)
            elif agents[i].region == 1:
                 COE1 += len(agents[i].employees_IDs)
        if agents[i].type == "Cap":
            if agents[i].region == 0:
                CAE0 += len(agents[i].employees_IDs)
            elif agents[i].region == 1:
                CAE1 += len(agents[i].employees_IDs)
            
    #print("Aggregare employment looking at H regions 0,1:", [HE0, HE1, HE1 + HE0 ])
    #print("Aggregare employment looking at Cons regions 0,1:", [COE0, COE1, COE1 + COE0 ])
    #print("Aggregare employment looking at Cap regions 0,1:", [CAE0, CAE1, CAE1 + CAE0 ])
    return [[HE0, HE1, HE1 + HE0 ] ,  [COE0, COE1, COE1 + COE0 ] , [CAE0, CAE1, CAE1 + CAE0 ] ]

def price_average_cons(model):
    price0 = 0
    price1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule._agents[i]
        if a.type == "Cons":
            if a.region == 0:
                price0 += a.price * a.production_made
                firms0 += a.production_made
            elif a.region == 1:
                price1 += a.price * a.production_made
                firms1 += a.production_made 
    #print("Average cons price ", [price0 / firms0, price1 / firms1])
    return [ round(price0 / firms0, 4), round(price1 / firms1, 4)]      


def regional_average_profits_cons(model):
    profit0 = 0
    profit1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            if a.region == 0:
                profit0 += a.profits #* a.production_made
                firms0 += 1
            elif a.region == 1:
                profit1 += a.profits # * a.production_made
                firms1 += 1
    #print(" Average profits are  ", [profit0 / firms0, profit1 / firms1])
    return [round(profit0 / firms0, 4), round(profit1 / firms1, 4)]  
def regional_average_profits_cap(model):
    profit0 = 0
    profit1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cap":
            if a.region == 0:
                profit0 += a.profits #* a.production_made
                firms0 += 1
            elif a.region == 1:
                profit1 += a.profits # * a.production_made
                firms1 += 1
    #print(" Average profits are  ", [profit0 / firms0, profit1 / firms1])
    return [round(profit0 / firms0, 4), round(profit1 / firms1, 4)]  


def regional_profits_cons(model):
    profit0 = 0
    profit1 = 0
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            if a.region == 0:
                profit0 += a.profits #* a.production_made
            elif a.region == 1:
                profit1 += a.profits # * a.production_made
    #print("Total profits cons are  ", [profit0, profit1])
    return [round(profit0, 4), round(profit1, 4)]  

def regional_profits_cap(model):
    profit0 = 0
    profit1 = 0
    for i in range(len(model.schedule._agents)):
        a = model.schedule.agents[i]
        if a.type == "Cap":
            if a.region == 0:
                profit0 += a.profits #* a.production_made
            elif a.region == 1:
                profit1 += a.profits # * a.production_made
   # print(" Total profits cap are  ", [profit0, profit1])
    return [round(profit0, 4), round(profit1, 4)]  




def regional_average_nw(model):
    NW0 = 0
    NW1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            if a.region == 0:
                NW0 += a.net_worth # * a.production_made
                firms0 += 1
            elif a.region == 1:
                NW1 += a.net_worth # * a.production_made
                firms1 += 1
    #print(" Average NW are  ", [NW0 / firms0, NW1 / firms1])
    return [round(NW0 / firms0, 2), round(NW1 / firms1, 2)]  
    


def sectoral_aggregate_liquid_assets(model):
    LA_cap0 = [a.net_worth for a in model.schedule.agents if a.type == "Cap" and a.region == 0]
    LA_cap1 = [a.net_worth for a in model.schedule.agents if a.type == "Cap" and a.region == 1]
    LA_cons0 = [a.net_worth for a in model.schedule.agents if a.type == "Cons" and a.region == 0]
    LA_cons1 = [a.net_worth for a in model.schedule.agents if a.type == "Cons" and a.region == 1]
    return [[sum(LA_cap0), sum(LA_cons0)],[sum(LA_cap1), sum(LA_cons1)]]

def sectoral_aggregate_debt(model):
    debt_cap = 0
    debt_cons0 = [a.debt for a in model.schedule.agents if a.type == "Cons" and a.region == 0]
    debt_cons1 = [a.debt for a in model.schedule.agents if a.type == "Cons" and a.region == 1]
    return [[debt_cap, sum(debt_cons0)], [debt_cap, sum(debt_cons1)]]

def cons_ids_region(model):
    region0_IDs = [ a.unique_id for a in model.schedule.agents if a.type =="Cons" and a.region == 0 ]
    region1_IDs = [ a.unique_id for a in model.schedule.agents if a.type == "Cons" and a.region == 1 ]
    '''
    last0 = max(region0_IDs)
    last1 = max(region1_IDs)
    if len(region0_IDs) > 0:
        region0_IDs.remove(last0)
    if len(region1_IDs):
        region1_IDs.remove(last1)
    '''
    return[region0_IDs, region1_IDs]

def firm_region(model):
    region0_cap = [ [a, a.unique_id]  for a in model.schedule.agents if a.type == "Cap" and a.region == 0 ]
    region1_cap = [ [a, a.unique_id] for a in model.schedule.agents if  a.type == "Cap"  and a.region == 1 ]
    region0_cons = [ [a, a.unique_id] for a in model.schedule.agents if a.type == "Cons" and a.region == 0 ]
    region1_cons = [ [a, a.unique_id] for a in model.schedule.agents if  a.type == "Cons"  and a.region == 1 ]
    '''
    last0 = max(region0_IDs)
    last1 = max(region1_IDs)
    if len(region0) >1:
        for j in region0:
            if j[1] == last0:
                region0.remove(j)
    if len(region1) > 0:
        for m in region1:
            if m[1] == last1:
                region1.remove(m)
    '''
    return[[region0_cap, region0_cons],[region1_cap, region1_cons]]