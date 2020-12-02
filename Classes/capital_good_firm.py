'''

model/classes/capital_good-firm
A MESA Agent class for capital good firms (sector 1)

''' 

from mesa import Agent, Model

from model.modules import research_and_development as rd
from model.modules import goods_market as gm
from model.modules import labor_dynamics as ld
from model.modules import migration as migration

import numpy as np
import math
import random
import bisect


class CapitalGoodFirm(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # general #
        self.type = "Cap"
        self.previous_productivity = [self.model.initial_productivity,self.model.initial_productivity]
        self.productivity = [self.model.initial_productivity, self.model.initial_productivity]      #(A,B) pair
        self.cost = 0                  # production cost (recalculated later)
        self.price = 0                 # unit price (recalculated later)
        self.sales = 0                 # previous sales, initially 0
        self.net_worth= self.model.initial_net_worth               # money for spending
        self.lifecycle = 1
        #self.flooded = False
        
        # labor market #
        self.open_vacancies = False
        self.employees_IDs = []
        self.wage = self.model.initial_wages

        # capital goods market #
        
        trade_cost = self.model.transport_cost   #temporary
        self.competitiveness = [1,1]
        self.market_share = [1/self.model.num_firms1, 1/self.model.num_firms1]  
        self.market_share_history = []# [region 0, region 1]
        self.regional_orders = []      # orders placed by consumption-good firms
        self.export_orders = []
        self.brochure_regional =[self.productivity[0], self.price, self.unique_id]
        self.brochure_export = [self.productivity[0], self.price * trade_cost, self.unique_id]
        self.profits = 0 
        self.real_demand_cap = 0
        self.production_made = 0 
        self.client_IDs = []
        self.lifecycle = 0
        self.RD_budget = 0
        self.productivity_list = []
    
        # climate change #
        self.CCA_resilience = [1,1] 
        self.CCA_RD_budget = 0        # placeholder value for now


    ''' 
    Calculate the unit cost of production
    '''
    def calculateProductionCost(self):
        '''
        if self.flooded == True:
            damages = min( self.model.S / self.CCA_resilience[0], self.model.S)
            self.productivity[1] =  self.productivity[1]
        '''
        self.cost = self.wage / self.productivity[1]


    ''' 
    Calculate the unit price
    '''
    def calculatePrice(self, markup=0.1):
        self.price = 1+markup * self.cost
        #print("my cap price is ", self.price)


    ''' 
    Research and development: Productivity
    '''
    def RD(self):
        # step 1: RD budget
        self.RD_budget, self.IN, self.IM = rd.calculateRDBudget(self.sales, self.net_worth)
        '''
        if self.flooded == True:
            self.RD_budget = 0 
        '''
        # step 2: innovate
        #print("old prod", self.productivity)
        in_productivity = rd.innovate(self.IN, self.productivity)
        #print("new prod", in_productivity)
        # step 3: imitate
        #competitors = [ a.unique_id for a in self.model.schedule.agents if ( a.type == "Cap" and a.lifecycle > 1 )]
        im_productivity = rd.imitate(self.IM,
                                     self.model.ids_firms1,
                                     self.model.schedule._agents,
                                     self.productivity,
                                     self.region)
        #print("In ", self.IN, "IM ", self.IM)
        # step 4: choose best technology to adopt
        self.previous_productivity = self.productivity
        self.productivity_list.append([ self.productivity[0], in_productivity[0], im_productivity[0]])
        self.productivity[0] = round(max(self.productivity[0], in_productivity[0], im_productivity[0]), 3)
        self.productivity[1] = round(max(self.productivity[1], in_productivity[0], im_productivity[1]) , 3)
        
        #self.model.schedule.time == (self.model.shock_time + 2):                             ## Recovering lab productivity after disaster 
         #   self.productivity[1] = max( self.productivity[0], self.model.productivity[1])


    '''
    Research and development: Climate Change Adaptation
    '''
    def CCA_RD(self):
        # step 1: CCA RD budget
        self.CCA_RD_budget, self.CCA_IN, self.CCA_IM = rd.calculateRDBudgetCCA(self.sales, self.net_worth)

        # step 2: innovate
        in_resilience = rd.innovate_CCA(self.CCA_IN, self.CCA_resilience)

        # step 3: imitate
        im_resilience = rd.imitate_CCA(self.CCA_IM,
                                       self.model.ids_firms1,
                                       self.model.schedule._agents,
                                       self.CCA_resilience,
                                       self.region)

        # step 4: choose the best resilience coefficient to adopt
        self.CCA_resilience[0] = max(self.CCA_resilience[0], in_resilience[0], im_resilience[0])
        self.CCA_resilience[1] = max(self.CCA_resilience[1], in_resilience[1], im_resilience[1])
        
    
            



    ''' 
    Method stumps to be filled in later
    '''
    def calculateLaborProductivity(self):
        return 2

    def wage_calculation(self):
    #initial value
        r = self.region
        average_regional_wage = self.model.datacollector.model_vars['Average_Salary'][int(self.model.schedule.time)][r]
        minimum_wage = self.model.datacollector.model_vars["Minimum_wage"][int(self.model.schedule.time) ]
           # if average_regional_wage == 0:
                #self.wage = max(1 , self.wage)
            #else:
        self.wage = max( minimum_wage, average_regional_wage * (1 + 0.01))
            

            


    '''
    NOT USED AT THE MOMENT, different procedure for wage calculation 
 
    def wage_determination(self):
        if (self.model.schedule.time < 2):
            return #random.randint(5,15)        #initial value
        else:
            r = self.region
            current_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time)][r]
            previous_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time) - 1 ][r]
            current_average_productivty_my_region = self.model.datacollector.model_vars['Capital_firms_av_prod'][int(self.model.schedule.time)][r]
            previous_average_productivty_my_region = self.model.datacollector.model_vars['Capital_firms_av_prod'][int(self.model.schedule.time) - 1 ][r]

         

            if previous_average_productivty_my_region < 1 :
                delta_productivity_average = 2
            else:
                delta_productivity_average = (current_average_productivty_my_region - previous_average_productivty_my_region) / previous_average_productivty_my_region
                    
            current_productivty  = self.productivity[1]
            delta_unemployment = (current_unemployment_rate_my_region - previous_unemployment_rate_my_region) / previous_unemployment_rate_my_region
            delta_my_productivity = (current_productivty - self.previous_productivity[1]) / self.previous_productivity[1]
            #print("I am cap firm ",self.unique_id, "my wage was ", self.wage, "")
            #print( "my region is ", self.region ," delta unemployment  ", delta_unemployment, " deltaregiona productivity ", delta_productivity_average)
            self.wage = self.wage * (1 + 0.50 * delta_my_productivity + 0.50 * delta_productivity_average + 0 *delta_unemployment )
            

    '''


    '''
    Advertise products to consumption-good firms
    '''
    def advertise(self):
        trade_cost = self.model.transport_cost
        self.regional_orders = []
        self.export_orders = []

        # A brochure of what the firm offers: [productivity, price]
        self.brochure_regional =[self.productivity[0], self.price, self.unique_id]
        self.brochure_export = [self.productivity[0], self.price *(1 + trade_cost), self.unique_id]   #trade cost for buyers of the other region 

        # choose potential clients (pc) to advertise to, store their ids in a list self.agents_by_type[agent_type][agent.unique_id]
        r = self.region
        client_IDs = self.model.datacollector.model_vars['Cons_regional_IDs'][int(self.model.schedule.time)]
        client_regional_IDs = client_IDs[r]
        client_export_IDs = client_IDs[1 - r]
        
        
        

        #print(client_regional_IDs, client_export_IDs)
        #pc_IDs = self.client_IDs
        
        ##---pick some new random client from both regions (more likely from the same region ) --##
        if len(self.client_IDs) > 0:
            new_clients = min(1, len(self.client_IDs)//4)
            for i in range(new_clients):
                if random.uniform(0,1) < 0.7 and len(client_regional_IDs) > 0:
                    new_regional_client = random.sample(client_regional_IDs, 1)
                    if len(new_regional_client) > 1:  
                        new_regional_client += random.sample(client_regional_IDs, 1)
                    if new_regional_client not in self.client_IDs:
                            self.client_IDs += new_regional_client
                        
                
                    #print("my new regional clients is ", new_regional_client)
                else:
                    if len(client_export_IDs) > 0:
                        new_export_client = random.sample(client_export_IDs, 1)
                        if len(new_export_client)> 1:
                            new_export_client += random.sample(client_export_IDs, 1)
                        if new_export_client not in self.client_IDs:
                            self.client_IDs += new_export_client
                   # print("my new export clients are ", new_export_client)
            #print(" Hi I am firms", self.unique_id, "my new clients are ", self.client_IDs)
               
            
            #print("so my initial clients are ", self.client_IDs )
       
        ##----pick clients, first step or new entry firms --##
        if len(self.client_IDs) == 0:
            initial_clients = round(self.model.num_firms2 / 40)    
            for i in range(initial_clients):
                if random.uniform(0,1) < 0.7 and len(client_regional_IDs) > 0:
                    regional_client = random.sample(client_regional_IDs, 1)
                    if len(regional_client) > 1:  
                        regional_client += random.sample(client_regional_IDs, 1)
                    self.client_IDs += regional_client
                    #print("my regional clients are ", regional_client)
                else:
                    if len(client_export_IDs) > 0:
                        export_client = random.sample(client_export_IDs, 1)
                        if len(export_client)> 1:
                            export_client += random.sample(client_export_IDs, 1)
                        self.client_IDs += export_client
                   # print("my export clients are ", export_client)
            ##print(" Hi I am firms", self.unique_id, "my number of initial clients are ", initial_clients)
               
            
            #print("so my initial clients are ", self.client_IDs )
                
            
            # bounded rationality: randomly selec
        #t half of the firms

    
        ##-- send brochure to my chosen firms --##
        if len(self.client_IDs) > 0:
            for firm_id in self.client_IDs:
                #if firm_id in self.model.ids_firms2:
                    #print(firm_id)
                client = self.model.schedule.agents[firm_id]
                if client.region == self.region and client.type =="Cons":
                    client.offers.append(self.brochure_regional)        #regional client 
                elif client.region == 1 - self.region and client.type =="Cons":
                    client.offers.append(self.brochure_export)          #export client

        # note: return two lists of ids, one for local and one for export?
        return self.client_IDs





    '''
    Open vacancies or fire employees based on demand in this period
    '''
    def hire_and_fire_cap(self):
        self.real_demand_cap = 0
        if self.regional_orders != []:
            demand_int = 0
            for i in range(len(self.regional_orders)):
                demand_int += self.regional_orders[i][0]
            #print("regiona_orders", self.regional_orders, "demand int ", demand_int )
            self.real_demand_cap = demand_int
        if self.export_orders != []:
            demand_exp = 0
            for i in range(len(self.export_orders)):
                demand_exp += self.export_orders[i][0]
            #print("export_orders", self.export_orders, "demand int ", demand_exp)
            self.real_demand_cap += demand_exp

        #self.real_demand_cap = demand_int + demand_exp

        #self.regional_orders = []
        #self.export_orders = []

        self.labor_demand = self.real_demand_cap / self.productivity[1]  #+ self.RD_budget / self.wage 
        self.desired_employees, self.employees_IDs, self.open_vacancies = ld.hire_and_fire(self.labor_demand, 
                                                                                        self.employees_IDs, 
                                                                                           self.open_vacancies, 
                                                                                           self.model,
                                                                                           self.unique_id)
        #print("I am cap firm ", self.unique_id, "I want", self.desired_employees, "I have ", len(self.employees_IDs), "because my real demand was" , self.real_demand_cap)

    '''
    Firms check wheter they were able to satisy all their orders of they have to cancell some
    Accounting profits, sales, and costs
    '''
    def accounting_orders(self):
        
       # if self.flooded == True:
        #    self.productivity[1] = (1 - self.model.S) * self.productivity[1]
        
        self.production_made = max( 0 , round(  (len(self.employees_IDs) -  self.RD_budget  /self.wage )* self.productivity[1] , 2))
        #if self.model.schedule.time == (self.model.shock_time):  
         #   damages = min( self.model.S / self.CCA_resilience[1], self.model.S)
          #  self.production_made = (1 - damages) * self.production_made
        self.orders_filled = min(self.real_demand_cap, self.production_made)
        if self.orders_filled < self.real_demand_cap:
            total_orders = self.regional_orders + self.export_orders  # first come, first served (first the one in the region)
            #print("I  am cap firm ", self.unique_id, "I have to cancel some orders because I could produe ", self.production_made_cap," and my demand was ", self.real_demand_cap, "total orders ", total_orders)
            amount_to_cancel = self.real_demand_cap - self.orders_filled
            total_orders.reverse()
            #print("total orders timeline", total_orders)
            orders_canceled = []
            #print(total_orders)
            for i in range(len(total_orders)):
                if sum(j[0] for j in orders_canceled) < amount_to_cancel:
                    orders_canceled.append(total_orders[0])
                    del total_orders[0]
            #print(orders_canceled)
            for order in range(len(orders_canceled)):
    
                buyer = self.model.schedule.agents[orders_canceled[order][1]]
                buyer.order_canceled = True

                #print(buyer)
                     
               
                #print(buyer)
                       
                
            #print("orders cancelled", orders_canceled, "total orders ", total_orders )
                
            
            
         ##---acounting---##   
        self.sales = self.orders_filled * self.price
        self.total_costs_cap = self.cost * self.orders_filled 
        self.profits = self.sales - self.total_costs_cap - self.RD_budget - self.CCA_RD_budget
        self.net_worth += self.profits 
            
    '''
    Update my market share so that it's normalized
    '''
    def market_share_normalized(self):
        '''
        agents = self.model.schedule.agents
        for i in range(len(agents)):
            if agents[i].type == "Gov" and agents[i].region == 0:
                self.market_share[0] = agents[i].market_shares_normalized_cap[self.unique_id]
        '''
        gov = self.model.governments[0]
        self.market_share[0] = gov.market_shares_normalized_cap[self.unique_id]
        
                

    def climate_damages(self):
        self.flooded == False
        if self.region == 0 and self.model.schedule.time == self.model.shock_time:
            self.flooded == True
            #climate_shock = self.model.datacollector.model_vars['Climate_shock'][int(self.model.schedule.time)] +(self.model.s /10)
           # print("I am firm ", self.unique_id, "my capital vintage pre shock is ", self.capital_vintage)

           # print ("capital vintage after shocks ", self.capital_vintage)



    '''
    Migration
    '''
    def migrate(self):
        ##-- stochastic entry barrier to migration process --##
         if random.uniform(0,1) < 0.05 or self.lifecycle < 3:
        #if self.unique_id % 10 != 0:

            r = self.region

            if self.regional_orders != []:
                demand_int = 0
                for i in range(len(self.regional_orders)):
                    demand_int += self.regional_orders[i][0]
            else: 
                demand_int = 0
            if self.export_orders != []:
                demand_exp = 0
                for i in range(len(self.export_orders)):
                    demand_exp += self.export_orders[i][0]
            else: 
                demand_exp = 0
         
            if self.lifecycle == 0:
                demand_distance =  self.model.datacollector.model_vars["Regional_profits_cap"][int(self.model.schedule.time)]
                demand_distance = (demand_distance[r] - demand_distance[1 - r] )/ demand_distance[r]
            else:
                demand_distance = (demand_int - demand_exp) / (demand_int + 0.00001)
            unemployment_subsidy = self.model.datacollector.model_vars["Regional_unemployment_subsidy"][int(self.model.schedule.time)]
            mp = migration.cap_firms_migration_probability(demand_distance, self.region, self.wage, self.profits, self.model, unemployment_subsidy)
        
            self.region, self.employees_IDs, self.net_worth = migration.firm_migrate(mp, self.model, self.region, self.unique_id, self.employees_IDs, self.net_worth, self.wage)



    '''
    Firm exit and entry
    
    def entry_exit(self):
        # if my market share everywhere is zero
        if self.market_share[0] + self.market_share[1] == 0:
            print("Firm", self.unique_id,"exiting...")

            # remove myself
            if self.region == 0:
                self.model.ids_region0.remove(self.unique_id)
            elif self.region == 1:
                self.model.ids_region1.remove(self.unique_id)
            self.model.ids_firms1.remove(self.unique_id)
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

            # replace myself with a new firm, it takes over my unique_id
            i = self.unique_id
            a = CapitalGoodFirm(i, self.model)
            self.model.schedule.add(a)
            self.model.ids_firms1.append(i)

            # place the new firm in the grid
            y = self.random.randrange(2)
            self.model.grid.place_agent(a, (0,y))
            a.region = y
            if y == 0:
                self.model.ids_region0.append(i)
            elif y == 1:
                self.model.ids_region1.append(i)

            print("Capital good firm", self.unique_id, "left the market")
                        
            return True
        return False
    '''
    

    def stage0(self):
       # if self.region == 0:
        #    self.CCA_RD()
        if self.lifecycle > 0:
            #print(self.region, self.productivity)
            if self.sales > 0:
                self.RD()
            self.calculateProductionCost()
            self.calculatePrice()
            self.advertise()

    def stage1(self):
            self.wage_calculation()
        #self.wage_determination()
            self.hire_and_fire_cap()

    # wait for households to do labor search
    def stage2(self):
        pass

    def stage3(self):
        if self.lifecycle > 0:
            self.accounting_orders()
        #self.market_share_normalized()

    def stage4(self):
        if self.model.schedule.time >2:
            self.market_share_normalized()
        pass
        #self.climate_damages()
        #pass
    def stage5(self):
        pass

    def stage6(self):
        
        #if self.model.schedule.time > self.model.start_migration:
         #   self.migrate()
            
        '''           
        if self.model.S > 0:
            self.climate_damages()
            if (self.model.schedule.time >= self.model.shock_time and self.model.schedule.time < (self.model.shock_time + 10)) :
                if random.uniform(0,1) < 0.05:
                    self.migrate()
        '''
        #if self.lifecycle > 0:
            
        self.lifecycle += 1
        #pass