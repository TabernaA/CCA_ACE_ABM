# A file for storing data collection functions
# model/modules/data_collection.py
from scipy.stats import beta 
#import numpy as np
#from model.classes.capital_good_firm import CapitalGoodFirm
#from model.classes.consumption_good_firm import ConsumptionGoodFirm
#from model.classes.household import Household






'''
Productivity
'''

def productivity_firms_average(model):
    gov = model.governments[0]
    average_prod = gov.regional_av_prod
    
    return average_prod
        


def productivity_consumption_firms_average(model):
    gov = model.governments[0]
    average_prod = gov.regional_av_prod
    
    return average_prod



def productivity_capital_firms_average(model):
    gov = model.governments[0]
    average_prod = gov.cap_av_prod
    
    return average_prod


def investment_units(model):
    replacment_investment_units0 = 0
    expansion_investment_units0 = 0
    replacment_investment_units1 = 0
    expansion_investment_units1 = 0
    agents = model.firms2 #model.schedule.agents
    for i in range(len(agents)):
        #if (agents[i].type == "Cons"):
        if agents[i].region == 0:
            replacment_investment_units0 += agents[i].replacement_investment_units
            expansion_investment_units0 += agents[i].expansion_investment_units         
        if agents[i].region == 1:
            replacment_investment_units1+= agents[i].replacement_investment_units
            expansion_investment_units1 += agents[i].expansion_investment_units
                
    print(replacment_investment_units0 , expansion_investment_units0 , replacment_investment_units1 ,expansion_investment_units1)
    return [ round(replacment_investment_units0, 3) , round(expansion_investment_units0 , 3), round(replacment_investment_units1, 3) ,round(expansion_investment_units1, 3)]







def climate_shock_generator(model, a=1,b=100):
    s = beta.rvs(1,100)
    print("Climate shock is",s)
    return s


def regional_unemployment_rate(model):
    gov = model.governments[0]
    unemployment_rate = gov.unemployment_rates
    '''
    unemployment_rate_0 =  unemployment_rate[0]
    unemployment_rate_1 =  unemployment_rate[1]
   
    if int(model.schedule.time) > 0:
        unemployment_rate_0_old = model.datacollector.model_vars['Unemployment_Regional'][int(model.schedule.time) - 1][0]
        unemployment_rate_1_old = model.datacollector.model_vars['Unemployment_Regional'][int(model.schedule.time) - 1][1]
    else: 
        unemployment_rate_0_old = 1
        unemployment_rate_1_old = 1
    
    if (unemployment_rate_0 or unemployment_rate_0_old)  < 0.01:
        delta_unemployment_0 = 0
    else:
            
        delta_unemployment_0 =  max( -0.025, min( 0.025 ,( unemployment_rate_0 - unemployment_rate_0_old) / max(unemployment_rate_0, unemployment_rate_0_old)))
        
    if (unemployment_rate_1 or unemployment_rate_1_old)  < 0.01:
        delta_unemployment_1 = 0
    else:
            
        delta_unemployment_1 =  max( -0.025, min( 0.025 ,( unemployment_rate_1 - unemployment_rate_1_old) / max(unemployment_rate_1, unemployment_rate_1_old)))
     
    unemployment_diffrence0 = unemployment_rate[2]
    unemployment_diffrence1 = unemployment_rate[3]
    '''
    
    return   unemployment_rate 




def regional_aggregate_market_share(model):
    RAMS0 = 0
    RAMS1 = 0
    RAMS2 = 0
    RAMS2_0 = 0
    RAMS2_1 = 0
    agents = model.firms2 #model.schedule.agents
    for i in range(len(agents)):
        #if agents[i].type == "Cons":
        RAMS0 += agents[i].market_share[0]
        RAMS1 += agents[i].market_share[1]
        RAMS2 += agents[i].market_share[2]
        if agents[i].region == 0:
           RAMS2_0 += agents[i].market_share[2]
        elif agents[i].region == 1: 
           RAMS2_1 += agents[i].market_share[2]
    #print("Aggregate market share region 0,1:", [RAMS0, RAMS1])
    return [round(RAMS0, 4), round(RAMS1,4), round(RAMS2,4), round(RAMS2_0,4), round(RAMS2_1,4)]


def demand_export_rate(model):
    RAMS0 = 0 
    RAMS1 = 0
    old_exp_C = 1
    if model.schedule.time > 1:
        old_exp_C = model.datacollector.model_vars['CONSUMPTION'][int(model.schedule.time) ][2]
    agents = model.firms2 #model.schedule.agents
    for i in range(len(agents)):
        if agents[i].region == 0:
            RAMS0 += agents[i].regional_demand[2]
        elif agents[i].region == 1:
            RAMS1 += agents[i].regional_demand[2]
        #if agents[i].type == "Cons":
    return [round(RAMS0/old_exp_C , 5), round(RAMS1/ old_exp_C,5)]

'''
Normalized market shares for each region


def market_share_normalized(model):
    MS0 = [a.market_share[0] for a in model.schedule.agents if ( a.type == "Cons")]
    MS1 = [a.market_share[1] for a in model.schedule.agents if (a.type == "Cons")]

    ids = [a.unique_id for a in model.schedule.agents if (a.type == "Cons")]
    
    norm0 = [float(i)/(sum(MS0) + 0.0001) for i in MS0]
    norm1 = [float(i)/(sum(MS1) + 0.0001) for i in MS1]


    # Make a dictionary of the form
    # {unique_id : (normalized ms 0, normalized ms 1)}
    norm_ms = zip(norm0, norm1)
    norm_ms_dict = dict(zip(ids, norm_ms))

    return norm_ms_dict
'''

def regional_costs(model):
    RC0 = 0
    RC1 = 0
    agents = model.firms2 #model.schedule.agents
    for i in range(len(agents)):
        #if agents[i].type == "Cons":
        if agents[i].region == 0:
            RC0 += agents[i].cost
        elif agents[i].region == 1:
            RC1 += agents[i].cost 
    #print("Aggregate market share region 0,1:", [RAMS0, RAMS1])
    return [round(RC0, 4), round(RC1,4)]


def regional_minimum_wage(model):
    RMW = 0.0001
    '''
    agents = model.schedule.agents
    #for i in range(len(agents)):
        #if agents[i].type == "Gov":
            if agents[i].region == 0:
    '''
    gov = model.governments[0]
    RMW = gov.minimum_wage_region
    #print(" Min wage  0,1:", [round(RMW, 4)])
    return RMW

def regional_unemployment_subsidy(model):


    agent = model.governments[0] #model.schedule.agents
    
    #print("Unemployment subsidy region 0,1:", [round(RUS0, 4), round(RUS1, 4)])
    return  agent.unemployment_subsidy



def regional_aggregate_employment(model):
    gov = model.governments[0]
    aggr_empl = gov.aggregate_employment
    return aggr_empl


def regional_aggregate_unemployment(model):
    gov = model.governments[0]
    aggr_unempl = gov.aggregate_unemployment
    return aggr_unempl


#------------------SALARIES -----------------------------#
'''
# returns average salary for both regions in the form [region0, region1]
def regional_average_salary(model):
    salaries0 = []
    salaries1 = []
    agents = model.schedule._agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons" or firm.type == "Cap":
            if firm.region == 0:
                for id in firm.employees_IDs:
                    salaries0.append(firm.wage)
            elif firm.region == 1:
                for id in firm.employees_IDs:
                    salaries1.append(firm.wage )
    if salaries0 != [] and salaries1 != []:
        #print("Avg salary regions 0,1:", [np.average(salaries0),np.average(salaries1)])
        return [np.average(salaries0),np.average(salaries1)]
    elif salaries0 == [] and salaries1 == []:
        print("Could not compute average salary")
        return [0,0]
    elif salaries0 == []:
        #print("Avg salary regions 0,1:", [0, np.average(salaries1)])
        return [0, np.average(salaries1)]
    elif salaries1 == []:
        #print("Avg salary regions 0,1:", [np.average(salaries0), 0])
        return [np.average(salaries0), 0]
'''
def regional_average_salary(model):
    
    gov = model.governments[0]

    return gov.average_wages
# returns average salary for capital-good firms for both regions in the form [0, 1]
# edit this to use the method above
def regional_average_salary_cap(model):
    gov = model.governments[0]

    return gov.salaries_cap


# average salary for consumption-good firms for each region
# edit this to be the same as two above
# maybe overload regional_average_salary and pass Cap/Cons 
# (self.type) into the function to remove redundancy
def regional_average_salary_cons(model):
    gov = model.governments[0]

    return gov.salaries_cons




def regional_average_competitiveness(model):
    '''
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Gov" and agents[i].region == 0:
            avg_comp = agents[i].average_normalized_comp
    '''
    gov = model.governments[0]
    avg_comp = gov.average_normalized_comp
    #print("regional avg competitiveness 0,1:", [avg_comp[0], avg_comp[1] ])
    return [round(avg_comp[0], 6), round(avg_comp[1], 6), round(avg_comp[2], 6)]

'''
Population
'''
'''
def regional_population_total(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule._agents)):
        a = model.schedule._agents[i]
        if a.region == 0:
            r0 += 1
        elif a.region == 1:
            r1 += 1
    return [r0, r1]

'''
def regional_population_households(model):
    agent = model.governments[0] 
    house_pop = agent.regional_pop_hous

    return house_pop 


def regional_population_households_region_0(model):
    agent = model.governments[0] 
    house_pop = agent.regional_pop_hous[0]

    return house_pop 


def regional_population_cons_region_0(model):
    r0 = 0
    agents = model.firms2 
    for i in range(len(agents)):
        a = agents[i]
        if a.region == 0:
            r0 += 1
        
    #print("Cap population region 0,1 is ", [r0, r1]) 
    return r0



def regional_population_cons(model):
    r0 = 0
    r1 = 0
    agents = model.firms2 
    for i in range(len(agents)):
        a = agents[i]
        if a.region == 0:
            r0 += 1
        elif a.region == 1:
            r1 += 1
    #print("Cap population region 0,1 is ", [r0, r1]) 
    return [r0, r1]

'''
# should be redundant
def regional_population_cons_region_0(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule._agents)):
        a = model.schedule._agents[i]
        if a.region == 0 and a.type == "Cons":
            r0 += 1
       
    return r0
'''

def regional_population_cap(model):
    r0 = 0
    r1 = 0
    agents = model.firms1
    for i in range(len(agents)):
        a = agents[i]
        if a.region == 0:
            r0 += 1
        elif a.region == 1:
            r1 += 1
    #print("Cap population region 0,1 is ", [r0, r1])
    return [r0, r1]



def regional_balance(model):
    d0 = 0
    d1 = 0
    governments = model.governments
    for i in range(len(governments)):
        a = governments[i]
        if  a.region == 0:
            d0 = a.fiscal_balance
        elif a.region == 1:
            d1 = a.fiscal_balance
    return [d0, d1]


def regional_capital(model):
    r0 = 0
    r1 = 0
    fraction = 0.3
    agents = model.firms2 
    for i in range(len(agents)):
        a = agents[i]
        #if a.type == "Cons":
        capital_stock = sum(i.amount for i in a.capital_vintage)
        if a.region == 0:
            r0 += capital_stock
        elif a.region == 1:
            r1 += capital_stock
    if r0 == 0:
        r0 = r1
    if r1 ==0:
        r1 = r0
    average_regional_cons_firm= model.datacollector.model_vars["Population_Regional_Cons_Firms"][int(model.schedule.time)]
    firm_capital_amount0 = round( fraction * r0 / max( 1, average_regional_cons_firm[0]) , 2)
    firm_capital_amount1 =round( fraction * r1 / max( 1, average_regional_cons_firm[1]) , 2)

    return [ round(r0, 2), round(r1, 2),  firm_capital_amount0,  firm_capital_amount1]


#-----------------PRICE------#
def price_average_cons(model):
    gov = model.governments[0]
    cons_prices = gov.av_price_cons

    return cons_prices


def quantity_ordered(model):
    
    firms0 = 0.0001
    firms1 = 0.0001
    agents = model.firms2
    for i in range(len(agents)):
        a = agents[i]
        #if a.type == "Cons":
        if a.region == 0:
            firms0 += a.quantity_ordered
        if a.region ==1:
            firms1 += a.quantity_ordered
    demand_0 = 0
    demand_1 = 0
    cap_agents = model.firms1
    for j in range(len(cap_agents)):
        c = cap_agents[j]
        if c.region == 0:
            demand_0 += sum(c.real_demand_cap)
        if c.region == 1:
            demand_1 += sum(c.real_demand_cap)

            
                #firms1 += a.quantity_mad
    #print("Average cons price con MS, " , [ round(price0, 5) , round(price1, 5)])
    return [ abs(round(firms0, 5)) , abs( round(firms1, 5)),  demand_0, demand_1, firms0 + firms1 - demand_0 - demand_1]#/ firms0, price1 / firms1]


def price_average_cap(model):
    gov = model.governments[0]
    cap_prices = gov.av_price_cap

    return cap_prices
