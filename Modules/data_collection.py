# A file for storing data collection functions
# model/modules/data_collection.py
from scipy.stats import beta 
import numpy as np
from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.classes.household import Household

'''
Aggregate productivity

def productivity_a1(model):
    productivity = 0
    for i in range(len(model.schedule.agents)):
        if (type(model.schedule.agents[i]) is CapitalGoodFirm):
            productivity += model.schedule.agents[i].productivity[0]
    return productivity

def productivity_b1(model):
    productivity = 0
    for i in range(len(model.schedule.agents)):
        if (type(model.schedule.agents[i]) is CapitalGoodFirm):
            productivity += model.schedule.agents[i].productivity[1]
    return productivity

Capital good firms


def productivity_capital_firms_region_0_average(model):
    productivity = 0
    firms = 0.00001
    for i in range(len(model.schedule.agents)):
        if (type(model.schedule.agents[i]) is CapitalGoodFirm and model.schedule.agents[i].region == 0 ):
            productivity += model.schedule.agents[i].productivity[0]
            firms += 1
            
    return productivity / firms

def productivity_capital_firms_region_1_average(model):
    productivity = 0
    firms = 0.00001
    for i in range(len(model.schedule.agents)):
        if (type(model.schedule.agents[i]) is CapitalGoodFirm and model.schedule.agents[i].region == 1 ):
            productivity += model.schedule.agents[i].productivity[0]
            firms += 1
    return productivity / firms


'''

#Consumption good firms 

def productivity_consumption_firms_average(model):
    productivity0 = 0
    firms0 = 0.00001
    productivity1 = 0
    firms1 = 0.00001
    a = model.schedule.agents
    for i in range(len(a)):
        if type(a[i]) is ConsumptionGoodFirm:
            if a[i].region == 0:
                productivity0 += a[i].productivity[1] * sum(a[i].market_share) #en(a[i].employees_IDs) #a[i].market_share[0]
               # firms0 += len(a[i].employees_IDs)
            if a[i].region == 1:
                productivity1 += a[i].productivity[1] *  sum(a[i].market_share)
               # firms1 +=   len(a[i].employees_IDs)
                
    return [productivity0, productivity1]
def productivity_capital_firms_average(model):
    productivity0 = 0
    firms0 = 0.00001
    productivity1 = 0
    firms1 = 0.00001
    a = model.schedule.agents
    for i in range(len(a)):
        if type(a[i]) is CapitalGoodFirm:
            if a[i].region == 0:
                productivity0 += a[i].productivity[1] * sum(a[i].market_share) #en(a[i].employees_IDs) #a[i].market_share[0]
               # firms0 += len(a[i].employees_IDs)
            if a[i].region == 1:
                productivity1 += a[i].productivity[1] *  sum(a[i].market_share)
               # firms1 +=   len(a[i].employees_IDs)
                
    return [productivity0, productivity1]

def productivity_consumption_firms_region_1_average(model):
    productivity = 0
    firms = 0.00001
    for i in range(len(model.schedule._agents)):
        agent = model.schedule._agents[i]
        if (agent.type == "Cons" and agent.region == 1 ):
            productivity += agent.productivity[1]
            firms += 1
            
    return productivity / firms


def regional_unemployment_rate_region10(model):
    replacment_investment_units0 = 0
    expansion_investment_units0 = 0
    replacment_investment_units1 = 0
    expansion_investment_units1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        if (agents[i].type == "Cons"):
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



# more general function: returns result for both regions
def regional_unemployment_rate(model):
    unemployed_households0 = 0.00001
    unemployed_households1 = 0.00001
    households_region0 = 0.00001
    households_region1 = 0.00001
    households = model.ids_households
    agents = model.schedule._agents
    for i in households:
        if agents[i].region == 0:
            households_region0 += 1
            if agents[i].employer_ID == None:
                unemployed_households0 += 1
        elif agents[i].region == 1:
            households_region1 += 1
            if agents[i].employer_ID == None:
                unemployed_households1 += 1

    print("Unemployment rate regions 0,1:", [round(unemployed_households0 / households_region0, 5), round(unemployed_households1 / households_region1, 5)])
    return [ round(unemployed_households0 / households_region0, 5), round(unemployed_households1 / households_region1, 5)]


# see if this one can be removed
def regional_unemployment_rate_region0(model):
    unemployed_households0 = 0.00001
    unemployed_households1 = 0.00001
    households_region0 = 0.00001
    households_region1 = 0.00001
    households = model.ids_households
    agents = model.schedule.agents
    for i in households:
        if agents[i].region == 0:
            households_region0 += 1
            if agents[i].employer_ID == None:
                unemployed_households0 += 1


    #print("Unemployment rate regions 0,1:", [unemployed_households0 / households_region0, unemployed_households1 / households_region1])
    return unemployed_households0 / households_region0 



def regional_aggregate_market_share(model):
    RAMS0 = 0
    RAMS1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Cons":
            RAMS0 += agents[i].market_share[0]
            RAMS1 += agents[i].market_share[1]
    #print("Aggregate market share region 0,1:", [RAMS0, RAMS1])
    return [round(RAMS0, 4), round(RAMS1,4)]


'''
Normalized market shares for each region
'''

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



def regional_minimum_wage(model):
    RMW = 0.0001
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Gov":
            if agents[i].region == 0:
                RMW = agents[i].minimum_wage_region
    #print(" Min wage  0,1:", [round(RMW, 4)])
    return round(RMW, 4)

def regional_unemployment_subsidy(model):
    RUS0 = 0.00001
    RUS1 = 0.00001
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Gov":
            if agents[i].region == 0:
                RUS0 = agents[i].unemployment_subsidy
            elif agents[i].region == 1:
                RUS1 = agents[i].unemployment_subsidy
    #print("Unemployment subsidy region 0,1:", [round(RUS0, 4), round(RUS1, 4)])
    return [round(RUS0, 4), round(RUS1, 4)]



def regional_aggregate_employment(model):
    ARE0 = 0.00001
    ARE1 = 0.00001
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Cons" or agents[i].type == "Cap":
            if agents[i].region == 0:
                ARE0 += len(agents[i].employees_IDs)
            elif agents[i].region == 1:
                ARE1 += len(agents[i].employees_IDs)
    #print("Aggregare employment regions 0,1:", [ARE0, ARE1])
    return [ARE0, ARE1]


def regional_aggregate_unemployment(model):
    ARE0 = 0.00001
    ARE1 = 0.00001
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Household" and agents[i].employer_ID == None:
            if agents[i].region == 0:
                ARE0 += 1
            elif agents[i].region == 1:
                ARE1 += 1
    #print("Aggregare unemployment regions 0,1:", [ARE0, ARE1])
    return [ARE0, ARE1]


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
    salaries0 = 0
    firms0 = 0
    salaries1 = 0
    firms1 = 0
    agents = model.schedule.agents
    for i in range(len(agents)):
        firm = agents[i]
        if firm.type == "Cons" or firm.type == "Cap":
            if firm.region == 0:
                salaries0 += firm.wage * len(agents[i].employees_IDs)
                firms0 += 1 * len(agents[i].employees_IDs)
            elif firm.region == 1:
                salaries1 += firm.wage * len(agents[i].employees_IDs)
                firms1 += 1 * len(agents[i].employees_IDs)

   # print("Agents in region 0:", model.ids_region0)
    #print("Agents in region 1:", model.ids_region1)

    if salaries0 != 0 and salaries1 != 0:
        print("Avg  salary regions 0,1:", [round(salaries0 / firms0 , 5), round(salaries1 / firms1, 5)])
        return [round(salaries0 / firms0, 5), round(salaries1 / firms1, 5)]
    elif salaries0 == 0:
        #print("Avg  salary regions 0,1:", [0, round(salaries1 / (firms1 + 0.001), 5)])
        return [0, round((salaries1 +0.001 )/ ( firms1 + 0.001), 5)]
    elif salaries1 == 0:
        #print("Avg csalary regions 0,1:", [ round(salaries0 / firms0,5),  0])
        return [ round(salaries0 / firms0, 5), 0]
    else:
        #print("Could not compute average cap salary")
        return [0,0]

# returns average salary for capital-good firms for both regions in the form [0, 1]
# edit this to use the method above
def regional_average_salary_cap(model):
    salaries0 = 0
    firms0 = 0
    salaries1 = 0
    firms1 = 0
    agents = model.schedule.agents
    for i in range(0,len(agents)):
        if agents[i].type == "Cap":
            if agents[i].region == 0:
                salaries0 += agents[i].wage * len(agents[i].employees_IDs)
                firms0 += 1 * len(agents[i].employees_IDs)
            elif agents[i].region == 1:
                salaries1 += agents[i].wage * len(agents[i].employees_IDs)
                firms1 += 1 * len(agents[i].employees_IDs)

   # print("Agents in region 0:", model.ids_region0)
    #print("Agents in region 1:", model.ids_region1)

    if salaries0 != 0 and salaries1 != 0:
        print("Avg cap salary regions 0,1:", [salaries0 / firms0, salaries1 / firms1])
        return [ round(salaries0 / firms0, 5), round( salaries1 / firms1, 5)]
    elif salaries0 == 0:
        #print("Avg cap salary regions 0,1:", [0, salaries1 / (firms1 + 0.001)])
        return [0, round((salaries1 +0.001) / ( firms1 + 0.001), 5)]
    elif salaries1 == 0:
        #print("Avg cap salary regions 0,1:", [salaries0 / firms0, 0])
        return [salaries0 / firms0, 0]
    else:
        #print("Could not compute average cap salary")
        return [0,0]


# average salary for consumption-good firms for each region
# edit this to be the same as two above
# maybe overload regional_average_salary and pass Cap/Cons 
# (self.type) into the function to remove redundancy
def regional_average_salary_cons(model):
    salaries0 = 0
    firms0 = 0
    salaries1 = 0
    firms1 = 0
    agents = model.schedule.agents
    for i in range(0,len(agents)):
        if agents[i].type == "Cons":
            if agents[i].region == 0:
                salaries0 += agents[i].wage * len(agents[i].employees_IDs)
                firms0 += 1 * len(agents[i].employees_IDs)
            elif agents[i].region == 1:
                salaries1 += agents[i].wage * len(agents[i].employees_IDs)
                firms1 += 1 * len(agents[i].employees_IDs)

   # print("Agents in region 0:", model.ids_region0)
    #print("Agents in region 1:", model.ids_region1)

    if salaries0 != 0 and salaries1 != 0:
        #print("Avg Cons salary regions 0,1:", [round(salaries0 / firms0, 5),round( salaries1 / firms1, 5)])
        return [ round(salaries0 / firms0 , 3), round(salaries1 / firms1, 3)]
    elif salaries0 == 0:
        #print("Avg Cons salary regions 0,1:", [0,round( (salaries1 + 0.001) / (firms1 + 0.001), 5)])
        return [0, round((salaries1 + 0.001) / (firms1 + 0.001), 5)]
    elif salaries1 == 0:
        #print("Avg Cons salary regions 0,1:", [salaries0 / firms0, 0])
        return [round(salaries0 / firms0, 5), 0]
    else:
        #print("Could not compute average cap salary")
        return [0,0]



def regional_average_competitiveness(model):
    
    agents = model.schedule.agents
    for i in range(len(agents)):
        if agents[i].type == "Gov" and agents[i].region == 0:
            avg_comp = agents[i].average_normalized_comp
    #print("regional avg competitiveness 0,1:", [avg_comp[0], avg_comp[1] ])
    return [round(avg_comp[0], 6), round(avg_comp[1], 6)]

'''
Population
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


def regional_population_households(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule._agents)):
        a = model.schedule.agents[i]
        if a.region == 0 and a.type == "Household":
            r0 += 1
        elif a.region == 1 and a.type == "Household":
            r1 += 1
    return [r0,r1]


# should be redundant
def regional_population_households_region_0(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule._agents)):
        a = model.schedule._agents[i]
        if a.region == 0 and a.type == "Household":
            r0 += 1
    return r0


def regional_population_cons(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.region == 0 and a.type == "Cons":
            r0 += 1
        elif a.region == 1 and a.type == "Cons":
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
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.region == 0 and a.type == "Cap":
            r0 += 1
        elif a.region == 1 and a.type == "Cap":
            r1 += 1
    #print("Cap population region 0,1 is ", [r0, r1])
    return [r0, r1]



def regional_balance(model):
    d0 = 0
    d1 = 0
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Gov" and a.region == 0:
            d0 = a.fiscal_balance
        elif a.region == 1 and a.type == "Gov":
            d1 = a.fiscal_balance
    return [d0, d1]


def regional_capital(model):
    r0 = 0
    r1 = 0
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            capital_stock = sum(i.amount for i in a.capital_vintage)
            if a.region == 0:
                r0 += capital_stock
            elif a.region == 1:
                r1 += capital_stock

    return [r0, r1]


#-----------------PRICE------#
def price_average(model):
    price0 = 0
    price1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            price0 += a.price * a.market_share[0]
            price1 += a.price * a.market_share[1]
                #firms1 += a.quantity_mad
    #print("Average cons price con MS, " , [ round(price0, 5) , round(price1, 5)])
    return [ abs(round(price0, 5)) , abs( round(price1, 5))]#/ firms0, price1 / firms1]


def quantity_ordered(model):
    
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cons":
            if a.region == 0:
                firms0 += a.quantity_ordered
            if a.region ==1:
                firms1 += a.quantity_ordered
            
                #firms1 += a.quantity_mad
    #print("Average cons price con MS, " , [ round(price0, 5) , round(price1, 5)])
    return [ abs(round(firms0, 5)) , abs( round(firms1, 5))]#/ firms0, price1 / firms1]


def price_average_cap(model):
    price0 = 0
    price1 = 0
    firms0 = 0.0001
    firms1 = 0.0001
    for i in range(len(model.schedule.agents)):
        a = model.schedule.agents[i]
        if a.type == "Cap":
            price0 += a.price * a.market_share[0]
            price1 += a.price * a.market_share[1]
                #firms1 += a.quantity_mad
    #print("Average cons price con MS, " , [ round(price0, 5) , round(price1, 5)])
    return [ abs(round(price0, 5)) , abs( round(price1, 5))]#/ firms0, price1 / firms1]