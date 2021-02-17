# model/classes/household.py
# A MESA Agent class for households / consumers

from mesa import Agent

#from model.classes.consumption_good_firm import ConsumptionGoodFirm
#from model.modules import labor_dynamics as ld
#from model.modules import migration as migration
from scipy.stats import bernoulli
import math
import bisect
import random


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.employer_ID = None
        self.type = "Household"
        self.lifecycle = 0
        self.migration_pr = self.model.pr_migration_h
        self.migration_start = 80
        
    def migration(self):
        #if bernoulli.rvs(self.migration_pr) == 1:# and self.model.schedule.time > self.model.start_migration:
            
            '''
            if self.employer_ID != None:
                employer = self.model.schedule.agents[self.employer_ID]
                mp, self.migration_pr = migration.households_migration_probability(self.region, self.model, employer.wage, self.migration_pr)
            else:
                unemployment_subsidy = self.model.datacollector.model_vars["Regional_unemployment_subsidy"][int(self.model.schedule.time)]
            '''
            mp = 0 #migration.households_migration_probability(self.region, self.model)
            region = self.region 
            model = self.model
            w_1 = 1
            #w_2 = 0.5
    ##--retrieve relevant parameters --#
           # prob_migration = 0
            gov = model.governments[0]
            unemployment_diff = gov.unemployment_rates[2 + region]
            wage_diff=  gov.average_wages[2 + region]
            if  wage_diff < 0  and unemployment_diff <= 0:
                 mp = 1-math.exp(w_1* wage_diff ) #+ w_2 * unemployment_diff )
        # migrate
            #print("my mp is (H) ", mp)
            if mp > 0:
                unique_id = self.unique_id
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
                    self.region = 1-region

            


    '''
    ---------------------------------------------------------------------------------------------
                                      Stages for staged activation
                                  all stages together make up one step
    ---------------------------------------------------------------------------------------------
    '''

    def stage0(self):
        
        if self.lifecycle > 15: # and  self.model.schedule.time > self.migration_start:
          #  if self.employer_ID == None:
                #if self.employer_ID % 10 != 0:
               self.migration()
    
        pass

    def stage1(self):
        pass

    def stage2(self):
        # first search for employment from capital-good sector
        if self.employer_ID == None:
          
            unique_id = self.unique_id 
           # employer_ID = 0
            model = self.model
            region = self.region
        #suitable_employers_IDs = []
        # get the list of all the firms living in the region 
            gov  = model.governments[region]  
            suitable_employers =  gov.open_vacancies_cap
            suitable_employers_IDs = [i.unique_id for i in suitable_employers]

        # find rhe ones with open vacancies in my region

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
                    self.employer_ID = possible_employers_IDs[best_i]

                    if employer.desired_employees == len(employer.employees_IDs):
                         employer.open_vacancies = False
                         suitable_employers.remove(employer)
                      



        
        #if unsuccessful search for employment in consumption-good sector
        
        if self.employer_ID == None:
            unique_id = self.unique_id 
           # employer_ID = 0
            model = self.model
            region = self.region
        #suitable_employers_IDs = []
        # get the list of all the firms living in the region 
            gov  = model.governments[region]  
            suitable_employers =  gov.open_vacancies_cons
            suitable_employers_IDs = [i.unique_id for i in suitable_employers]

        # find rhe ones with open vacancies in my region

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
                    self.employer_ID = possible_employers_IDs[best_i]

                    if employer.desired_employees == len(employer.employees_IDs):
                         employer.open_vacancies = False
                         suitable_employers.remove(employer)
        
        # if unsuccessful both times --> unemployed
        #print("Household", self.unique_id,"has employer:", self.employer_ID)
        

    def stage3(self):
        pass

    # migration
    def stage4(self):
        pass
    def stage5(self):
        '''
        if self.model.schedule.time > 50 and self.employer_ID == None:
           # if self.employer_ID is not None:
                #if self.employer_ID % 10 != 0:
            self.migration()
        '''
        pass        
        
       

    def stage6(self):
        '''
        if self.model.schedule.time > 50 and self.employer_ID == None:
           # if self.employer_ID is not None:
                #if self.employer_ID % 10 != 0:
            self.migration()
        '''
       

            #else:
             #   self.migration()
        self.lifecycle += 1
        pass