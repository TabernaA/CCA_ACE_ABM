# model/classes/household.py
# A MESA Agent class for households / consumers

from mesa import Agent

#from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.modules import labor_dynamics as ld
from model.modules import migration as migration
from scipy.stats import bernoulli
#import random 


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.employer_ID = None
        self.type = "Household"
        self.lifecycle = 0
       # self.migration_pr = self.model.pr_migration_h
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
            mp = migration.households_migration_probability(self.region, self.model)

        # migrate
            #print("my mp is (H) ", mp)
            if mp > 0:
                self.region = migration.household_migrate(mp, self.model, self.region, self.unique_id)
            #self.lifecyle = 0
            


    '''
    ---------------------------------------------------------------------------------------------
                                      Stages for staged activation
                                  all stages together make up one step
    ---------------------------------------------------------------------------------------------
    '''

    def stage0(self):
        
        if self.lifecycle > 16: # and  self.model.schedule.time > self.migration_start:
          #  if self.employer_ID == None:
                #if self.employer_ID % 10 != 0:
               self.migration()
    
        pass

    def stage1(self):
        pass

    def stage2(self):
        # first search for employment from capital-good sector
        if self.employer_ID == None:
            self.employer_ID = ld.labor_search_cap(self.unique_id, self.employer_ID, self.model, self.region)
        
        #if unsuccessful search for employment in consumption-good sector
        
        if self.employer_ID == None:
           self.employer_ID = ld.labor_search_cons(self.unique_id, self.employer_ID, self.model, self.region)
        
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