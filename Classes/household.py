# model/classes/household.py
# A MESA Agent class for households / consumers

from mesa import Agent

from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.modules import labor_dynamics as ld
from model.modules import migration as migration
import random 


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.employer_ID = None
        self.type = "Household"
        
    def migration(self):
        if random.uniform(0,1) < 0.20 and self.model.schedule.time > self.model.start_migration:
            
            if self.employer_ID != None:
                employer = self.model.schedule._agents[self.employer_ID]
                mp = migration.households_migration_probability(self.region, self.model, employer.wage,)
            else:
                unemployment_subsidy = self.model.datacollector.model_vars["Regional_unemployment_subsidy"][int(self.model.schedule.time)]
                mp = migration.households_migration_probability(self.region, self.model, unemployment_subsidy[1 - self.region])

        # migrate
            #print("my mp is (H) ", mp)
            self.region = migration.household_migrate(mp, self.model, self.region, self.unique_id)



    def step(self):

        # search for employer
      
        self.employer_ID = ld.labor_search(self.unique_id, self.employer_ID, self.model, self.region)
        self.employer_ID = ld.labor_search_cons(self.unique_id, self.employer_ID, self.model, self.pos)
        #print(self.employer_ID)
        # calculate migration probability
        if self.model.schedule.time > 0 and self.model.schedule.time % 4 == 0:
            if self.employer_ID != None:
                employer = self.model.schedule._agents[self.employer_ID]
                mp = migration.households_migration_probability(self.region, self.model, employer.wage)
            else:
                unemployment_subsidy = self.model.datacollector.model_vars["Regional_unemployment_subsidy"][int(self.model.schedule.time)]
                mp = migration.households_migration_probability(self.region, self.model, unemployment_subsidy[1 - self.region])
            
        else:   
            mp = 0
        
        
        self.region = migration.household_migrate(mp, self.model, self.region, self.unique_id)
         
    '''
    ---------------------------------------------------------------------------------------------
                                      Stages for staged activation
                                  all stages together make up one step
    ---------------------------------------------------------------------------------------------
    '''

    def stage0(self):
        pass

    def stage1(self):
        pass

    def stage2(self):
        # first search for employment from capital-good sector
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
        pass        
        
       

    def stage6(self):
        
       # if self.model.schedule.time > 50:
           # if self.employer_ID is not None:
                #if self.employer_ID % 10 != 0:
        #self.migration()
            #else:
             #   self.migration()
        
        pass