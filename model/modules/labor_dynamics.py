'''
model/modules/labor_dynamics.py

Functions for Labor Dynamics
labor_search()   : for households
labor_demand()   : for firms
hire_and_fire()  : for firms

'''
import random
import math
#import numpy as np


''' Labor Search:
Each unemployed household searches through the available suitable_employers
in both sectors and picks the employer offering the highest wage.

PARAMETERS
employer : reference to my employer
model    : reference to the model
pos      : reference to my position (self.pos)
'''
def labor_search(unique_id, employer_ID, model,  region):
    #if employer_ID == None:
    
        #suitable_employers_IDs = []
        # get the list of all the firms living in the region 
        gov  = model.governments[region]  
        suitable_employers =  gov.open_vacancies
        suitable_employers_IDs = [i.unique_id for i in suitable_employers]

        # find rhe ones with open vacancies in my region
        '''
        for i in range(len(region_employers)):
            if ( region_employers[i][0].open_vacancies == True):
                suitable_employers_IDs.append(region_employers[i][1])
        #print("SUITABLE EMPLOYERS IDS", suitable_employers_IDs)
        '''
        if len(suitable_employers_IDs) > 0:
            # due to bounded rationality, choose from a subset of all suitable employers
            possible_employers_IDs = random.sample(suitable_employers_IDs, math.ceil(len(suitable_employers_IDs)/5))

            # choose the employer who offers the highest wage
            if len(possible_employers_IDs) > 0:
                possible_employers_wages = []
                for id in possible_employers_IDs:
                    possible_employer = model.schedule.agents[id]
                    possible_employers_wages.append(possible_employer.wage)

                best_i = possible_employers_wages.index(max(possible_employers_wages))
                employer = model.schedule.agents[possible_employers_IDs[best_i]]
                employer.employees_IDs.append(unique_id)
                employer_ID = possible_employers_IDs[best_i]

                if employer.desired_employees == len(employer.employees_IDs):
                      employer.open_vacancies = False
                      suitable_employers.remove(employer)
                      


        return employer_ID

####----------Same procedure as labor search, but only for capital firms, used if we want to separate the labor market matching between the sectors (i.e. capital good go first since they pay more)

def labor_search_cap(unique_id, employer_ID, model, region):
        #suitable_employers_IDs = []
        # get the list of all the firms living in the region 
        gov  = model.governments[region]  
        suitable_employers =  gov.open_vacancies_cap
        suitable_employers_IDs = [i.unique_id for i in suitable_employers]

        # find rhe ones with open vacancies in my region
        '''
        for i in range(len(region_employers)):
            if ( region_employers[i][0].open_vacancies == True):
                suitable_employers_IDs.append(region_employers[i][1])
        #print("SUITABLE EMPLOYERS IDS", suitable_employers_IDs)
        '''
        if len(suitable_employers_IDs) > 0:
            # due to bounded rationality, choose from a subset of all suitable employers
            possible_employers_IDs = random.sample(suitable_employers_IDs, math.ceil(len(suitable_employers_IDs)/5))

            # choose the employer who offers the highest wage
            if len(possible_employers_IDs) > 0:
                possible_employers_wages = []
                for id in possible_employers_IDs:
                    possible_employer = model.schedule.agents[id]
                    possible_employers_wages.append(possible_employer.wage)

                best_i = possible_employers_wages.index(max(possible_employers_wages))
                employer = model.schedule.agents[possible_employers_IDs[best_i]]
                employer.employees_IDs.append(unique_id)
                employer_ID = possible_employers_IDs[best_i]

                if employer.desired_employees == len(employer.employees_IDs):
                      employer.open_vacancies = False
                      suitable_employers.remove(employer)
                      


        return employer_ID




####----------Same procedure as labor search, but only for consumption firms,

def labor_search_cons(unique_id, employer_ID, model, region):
        #suitable_employers_IDs = []
        # get the list of all the firms living in the region 
        gov  = model.governments[region]  
        suitable_employers =  gov.open_vacancies_cons
        suitable_employers_IDs = [i.unique_id for i in suitable_employers]

        # find rhe ones with open vacancies in my region
        '''
        for i in range(len(region_employers)):
            if ( region_employers[i][0].open_vacancies == True):
                suitable_employers_IDs.append(region_employers[i][1])
        #print("SUITABLE EMPLOYERS IDS", suitable_employers_IDs)
        '''
        if len(suitable_employers_IDs) > 0:
            # due to bounded rationality, choose from a subset of all suitable employers
            possible_employers_IDs = random.sample(suitable_employers_IDs, math.ceil(len(suitable_employers_IDs)/10))

            # choose the employer who offers the highest wage
            if len(possible_employers_IDs) > 0:
                possible_employers_wages = []
                for id in possible_employers_IDs:
                    possible_employer = model.schedule.agents[id]
                    possible_employers_wages.append(possible_employer.wage)

                best_i = possible_employers_wages.index(max(possible_employers_wages))
                employer = model.schedule.agents[possible_employers_IDs[best_i]]
                employer.employees_IDs.append(unique_id)
                employer_ID = possible_employers_IDs[best_i]

                if employer.desired_employees == len(employer.employees_IDs):
                      employer.open_vacancies = False
                      suitable_employers.remove(employer)
                      


        return employer_ID

'''
Firms determination of labor demand
'''
def labor_demand(capital_vintage, feasible_production):
    ## Find the most productive machines for the quantity I want to produce   ##
    ## Fewest machines needed to satisfy feasible_production ##
    Q = 0               # cumulative quantity
    machines_used = []

    # go over the machine vintage backwards, so starting with the most productive machines
    #num_machines = len(capital_vintage) - 1
    '''
    i = len(capital_vintage) - 1 
    while i >= 0:
        if (Q < feasible_production):
            machines_used.append(capital_vintage[i])
            Q += capital_vintage[i].amount
            #print(Q)
            i -= 1
    '''
    for i in range(len(capital_vintage) - 1, -1, -1):
        if Q < feasible_production:
            machines_used.append(capital_vintage[i])
            
            Q += capital_vintage[i].amount
           # print("Q", Q, "Feasible prod", feasible_production, "amount", capital_vintage[i].amount, "prod ", capital_vintage[i].productivity ),
            
    if Q > feasible_production:
        machines_used[-1].amount -= int(Q - feasible_production )
        #print("- 1 amount ", capital_vintage[-1].amount, "minus ", int(Q - feasible_production ))
        #for j in range(len(machines_used)):
            #print(machines_used[j].amount, machines_used[j].productivity)
    
    
    
   


    ## weighted average productivity of the machines I want to use ##
    average_productivity =  math.ceil(sum( [v.amount * v.productivity for v in machines_used] )  / sum([a.amount for a in machines_used] ))
    #print( "the totla is ", sum( [v.amount * v.productivity for v in machines_used]), "total amount is", sum([v.amount for v in machines_used] ))
   # print("average productivity is ", average_productivity, "my feasible production is ", feasible_production)
    ##for i in machines_used:
       # print("amount", i.amount, "productivity", i.productivity)
    # How much labor is needed to satisfy feasible production, given my average productivity
    #round(feasible_production , 5)
    labor_demand = max( 0, round(feasible_production / average_productivity , 2))
    #print("labor demand is ", labor_demand)
    
    return labor_demand, average_productivity




'''
hire and fire workers
'''
def hire_and_fire(labor_demand, employees_IDs, open_vacancies, model, id):
    # Number of desired employees 
    desired_employees =  round(labor_demand)
    #print("I desire", desired_employees, "I have ", len(employeeDesirs_IDs))
    #Depending on how many employees has vs.how many it wants, the firms post new vancancies or fire
    if desired_employees == len(employees_IDs):
        open_vacancies = False

    elif desired_employees > len(employees_IDs):
        open_vacancies = True
            
    elif desired_employees < len(employees_IDs):
        open_vacancies = False
        firing_employees = abs( desired_employees - len(employees_IDs))
        #print("firing employees", firing_employees)
        for i in range( int(firing_employees)):
            j = employees_IDs[0]
            employee = model.schedule.agents[employees_IDs[0]]
            employee.employer_ID = None
            del employees_IDs[0]
        


    return desired_employees, employees_IDs, open_vacancies
