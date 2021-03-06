'''

model/classes/capital_good-firm
A MESA Agent class for capital good firms (sector 1)

''' 
seed_value = 12345678
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed=seed_value)
from mesa import Agent

from model.modules import research_and_development as rd
#from model.modules import goods_market as gm
from model.modules import labor_dynamics as ld
from model.modules import migration as migration
#from scipy.stats import bernoulli
import math
\
import bisect
#import numpy as np

from scipy.stats import bernoulli
from scipy.stats import beta


class CapitalGoodFirm(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # general #
        self.type = "Cap"
       # self.previous_productivity = [self.model.initial_productivity, self.model.initial_productivity ]
        self.productivity = [self.model.initial_productivity, self.model.initial_productivity]      #(A,B) pair
        self.cost = 0                  # production cost (recalculated later)
        self.price = 0                 # unit price (recalculated later)
        self.sales = 0                 # previous sales, initially 0
        self.net_worth= self.model.initial_net_worth               # money for spending
        self.lifecycle = 1
        self.flooded = False
        
        # labor market #
        self.open_vacancies = False
        self.employees_IDs = []
        self.wage = self.model.initial_wages
        self.labor_demand = 0
        # capital goods market #
        
        trade_cost = self.model.transport_cost   #temporary
        self.competitiveness = [1,1]
        self.market_share = [1/self.model.num_firms1, 1/self.model.num_firms1]  
        self.market_share_history = []# [region 0, region 1]
        self.regional_orders = []      # orders placed by consumption-good firms
        self.export_orders = []
        #self.demands = [0,0]         
        self.brochure_regional =[self.productivity[0], self.price, self.unique_id]
        self.brochure_export = [self.productivity[0], self.price * ( 1 + trade_cost), self.unique_id]
        self.profits = 0 
        self.real_demand_cap = [0,0]         # the first element refers to regional demand, the second to the demand coming from the other region 
        self.production_made = 0 
        self.client_IDs = []
        self.lifecycle = 0
        self.RD_budget = 0
        self.IM = 0
        self.IN = 0
        self.productivity_list = []
        self.bankrupt = None
        
        #Migration tracking
        self.distances_mig = []
        self.region_history = []
       # self.migration_pr =  self.model.pr_migration_f
        self.pre_shock_prod = 0
        
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
        if self.flooded == True:
            shock = self.model.S
            self.pre_shock_prod = self.productivity[1]
           # print(self.productivity[1])
            #damages = min( self.model.S / self.CCA_resilience[0], self.model.S)
            self.productivity[1] =  self.productivity[1] * ( 1 - shock)
            #print('Flood I lost the productivity now', self.productivity[1])
        self.cost = self.wage / self.productivity[1]


    ''' 
    Calculate the unit price
    '''
    def calculatePrice(self, markup=0.04):
        self.price = 1+markup * self.cost
        #print("my cap price is ", self.price)


    ''' 
    Research and development: Productivity
    '''
    def RD(self):
        # step 1: RD budget
        if self.sales > 0:
            self.RD_budget, self.IN, self.IM = rd.calculateRDBudget(self.sales, self.net_worth)
        '''
        if self.flooded == True:
            self.RD_budget = 0 
        '''
        IN = self.IN
        prod = self.productivity 
        Z=0.3
        a=3
        b=3
        x_low=-0.15
        x_up=0.15
        in_productivity = [0,0]

    # Bernoulli draw to determine success (1) or failure (0)
        p = 1-math.exp(-Z*IN)
    #p1 = 1-math.exp(-Z*IN/2)
    #print( "P ", p , "b", b)
    
        if bernoulli.rvs(p) == 1: # new production productivity (B) from innovation
           a = (1 + x_low + beta.rvs(a,b)*(x_up-x_low)) 
           in_productivity[1] = prod[0] * a

       # if bernoulli.rvs(p) == 1:
        # new machine productivity (A) from innovation
           a_1 = (1 + x_low + beta.rvs(a,b)*(x_up-x_low))
           in_productivity[0] = prod[1] * a_1
        #print(a)
        



        #print("new prod", in_productivity)
        # step 3: imitate
        #competitors = [ a.unique_id for a in self.model.schedule.agents if ( a.type == "Cap" and a.lifecycle > 1 )]
    
        IM = self.IM
        firm_ids = self.model.ids_firms1
        agents = self.model.firms1
        reg = self.region
       # Z=0.3
        e=2
        im_productivity = [0,0]
       # print(Z, IM, IN)
    # Bernoulli draw to determine success (1) or failure (0)
        p = 1-math.exp(-Z*IM)
        if bernoulli.rvs(p) == 1:
        # store imitation probabilities and the corresponding firms
            imiProb = []
            imiProbID = []

        #for all capital-good firms, compute inverse Euclidean distances
            for id in firm_ids:
               firm = agents[id]
               distance = math.sqrt(pow(prod[0] - firm.productivity[0], 2) + \
                           pow(prod[0] - firm.productivity[0], 2))
               if distance == 0:
                   imiProb.append(0)
               else:
                # increase distance if the firm is in another region
                    if firm.region != reg:
                        imiProb.append(1/e*distance)
                    else:
                        imiProb.append(1/distance)
               imiProbID.append(firm.unique_id)

        # cumulative probability
            _sum = sum(imiProb)

            if (_sum > 0):
               acc = 0
               for i in range(len(imiProb)):
                   acc += imiProb[i] / _sum
                   imiProb[i] = acc

            # randomly pick a firm to imitate (index j)
               rnd = random.uniform(0,1)
               j = bisect.bisect_right(imiProb, rnd)

            # copy that firm's technology
               if j < len(imiProb):
                   firm = agents[imiProbID[j]]
                   im_productivity[0] = firm.productivity[0]
                   im_productivity[1] = firm.productivity[1]
        # step 4: choose best technology to adopt
        if self.pre_shock_prod != 0:                           ## Recovering lab productivity after disaster 
            self.productivity[1] = self.pre_shock_prod
            self.pre_shock_prod = 0 

            #print('restored now', self.productivity[1])
        self.previous_productivity = self.productivity
       # print(' previous prod', self.previous_productivity)
       # if self.flooded == True:
            # print(' previous prod', self.previous_productivity)
        self.productivity_list.append([ self.productivity[0], in_productivity[0], im_productivity[0]])
        self.productivity[0] = round(max(self.productivity[0], in_productivity[0], im_productivity[0], 1), 3)
        self.productivity[1] = round(max(self.productivity[1], in_productivity[1], im_productivity[1], 1), 3)
        


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
                                       self.model.firms1,
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
            

            


    
   # NOT USED AT THE MOMENT, different procedure for wage calculation 
 
    def wage_determination(self):


            r = self.region
            gov = self.model.governments[0]
            minimum_wage = gov.minimum_wage_region[r] 
           
           # minimum_wage = self.model.datacollector.model_vars["Minimum_wage"][int(self.model.schedule.time) ][r]
           
            top_wage = self.model.datacollector.model_vars['Top_wage'][int(self.model.schedule.time)][r]
            self.wage = max( minimum_wage, top_wage)
            '''
           
            #current_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time)][r]
            #previous_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time) - 1 ][r]
            #current_average_productivty_my_region = self.model.datacollector.model_vars['Capital_firms_av_prod'][int(self.model.schedule.time)][r]
            #previous_average_productivty_my_region = self.model.datacollector.model_vars['Capital_firms_av_prod'][int(self.model.schedule.time) - 1 ][r]
            
            #delta_unemployment = (current_unemployment_rate_my_region - previous_unemployment_rate_my_region) / previous_unemployment_rate_my_region
            current_productivty  = self.productivity[1]
            delta_my_productivity = (current_productivty - self.previous_productivity[1]) / self.previous_productivity[1]
            #print("I am cap firm ",self.unique_id, "my wage was ", self.wage, "")
            delta_productivity_average = gov.regional_av_prod[r + 2]
            minimum_wage = gov.minimum_wage_region[r] 
 
        

            #print( "my region is ", self.region ," delta unemployment  ", delta_unemployment, " deltaregiona productivity ", delta_productivity_average)
            self.wage = max(minimum_wage , round(self.wage * (1 + 0.25  * delta_my_productivity + 0.75 * delta_productivity_average), 3))
            '''



    '''
    Advertise products to consumption-good firms
    '''
    def advertise(self):
        trade_cost = self.model.transport_cost
        #print(trade_cost)
        self.regional_orders = []
        self.export_orders = []

        # A brochure of what the firm offers: [productivity, price]
        self.brochure_regional =[self.productivity[0], self.price, self.unique_id]
        self.brochure_export = [self.productivity[0], self.price *(1 + trade_cost), self.unique_id]   #trade cost for buyers of the other region 

        # choose potential clients (pc) to advertise to, store their ids in a list self.agents_by_type[agent_type][agent.unique_id]
        r = self.region
        client_IDs = self.model.datacollector.model_vars['Cons_regional_IDs'][int(self.model.schedule.time)]
        client_regional_IDs = client_IDs[r]
        len_regional = len(client_regional_IDs)
        client_export_IDs = client_IDs[1 - r]
        len_exp = len(client_export_IDs)
        new_export_clients = []
        new_regional_clients = []
        
        
        

        #print(client_regional_IDs, client_export_IDs)
        #pc_IDs = self.client_IDs
        
        ##---pick some new random client from both regions (more likely from the same region ) --##
        if len(self.client_IDs) > 0:
            new_clients = min(1, len(self.client_IDs)//5)
            number_regional_client = min( len_regional , round(new_clients * 0.75))
            number_external_client =  min( len_exp , new_clients - number_regional_client)
            if number_regional_client > 0:
                new_regional_clients = random.sample(client_regional_IDs, number_regional_client)

            if number_external_client > 0:
                new_export_clients = random.sample(client_export_IDs, number_external_client)
                
            all_clients = new_regional_clients + new_export_clients
            new_clients =  set(all_clients) - set(self.client_IDs)
            self.client_IDs += list(new_clients)
            
 

                   # print("my new export clients are ", new_export_client)
            #print(" Hi I am firms", self.unique_id, "my new clients are ", self.client_IDs)
               
            
            #print("so my initial clients are ", self.client_IDs )
       
        ##----pick clients, first step or new entry firms --##
        if len(self.client_IDs) == 0:
            initial_clients = round(self.model.num_firms2 / 40)
            number_regional_client = min( len_regional , round(initial_clients * 0.75))
            number_external_client =  min( len_exp , initial_clients - number_regional_client)
            
            if number_regional_client > 0:
                new_regional_clients = random.sample(client_regional_IDs, number_regional_client)

            if number_external_client > 0:
                new_export_clients = random.sample(client_export_IDs, number_external_client)
                
            all_clients = new_regional_clients + new_export_clients
            self.client_IDs = all_clients
            
        ##-- send brochure to my chosen firms --##
    
       #export client
       
        for firm_id in self.client_IDs:
                #if firm_id in self.model.ids_firms2:
                    #print(firm_id)
                client = self.model.schedule.agents[firm_id]
                if client.region == self.region:
                    client.offers.append(self.brochure_regional)        #regional client 
                elif client.region == 1 - r :
                    client.offers.append(self.brochure_export)  

        # note: return two lists of ids, one for local and one for export?
        return self.client_IDs





    '''
    Open vacancies or fire employees based on demand in this period
    '''
    def hire_and_fire_cap(self):
        self.real_demand_cap = [0,0]
        if self.regional_orders != []:
            demand_int = 0
            for i in range(len(self.regional_orders)):
                demand_int += self.regional_orders[i][0]
            #print("regiona_orders", self.regional_orders, "demand int ", demand_int )
            self.real_demand_cap[0] = demand_int
        if self.export_orders != []:
            demand_exp = 0
            for i in range(len(self.export_orders)):
                demand_exp += self.export_orders[i][0]
            #print("export_orders", self.export_orders, "demand int ", demand_exp)
            self.real_demand_cap[1] = demand_exp

        #self.real_demand_cap = demand_int + demand_exp

        #self.regional_orders = []
        #self.export_orders = []

        self.labor_demand = sum(self.real_demand_cap) / self.productivity[1] # + self.RD_budget / self.wage 
        self.desired_employees, self.employees_IDs, self.open_vacancies = ld.hire_and_fire(self.labor_demand, 
                                                                                           self.employees_IDs, 
                                                                                           self.open_vacancies, 
                                                                                           self.model,
                                                                                           self.unique_id)
        #print("I am cap firm ", self.unique_id, "I want", self.desired_employees, "I have ", len(self.employees_IDs), "because my real demand was" , sum(self.real_demand_cap))

    '''
    Firms check wheter they were able to satisy all their orders of they have to cancell some
    Accounting profits, sales, and costs
    '''
    def accounting_orders(self):
        
       # if self.flooded == True:
        #    self.productivity[1] = (1 - self.model.S) * self.productivity[1]
        
        self.production_made = max( 0 , round(  len(self.employees_IDs)  * self.productivity[1] , 2))   #   self.RD_budget  /self.wage )   
        if self.flooded == True:
            shock = self.model.S
            self.production_made = ( 1 - shock) * self.production_made
        #if self.model.schedule.time == (self.model.shock_time):  
         #   damages = min( self.model.S / self.CCA_resilience[1], self.model.S)
          #  self.production_made = (1 - damages) * self.production_made
        total_orders = sum(self.real_demand_cap)
        self.orders_filled = min( total_orders, self.production_made)
        if self.orders_filled < total_orders:
            orders = self.regional_orders + self.export_orders  # first come, first served (first the one in the region)
            #print("I  am cap firm ", self.unique_id, "I have to cancel some orders because I could produe ", self.production_made_cap," and my demand was ", self.real_demand_cap, "total orders ", total_orders)
            amount_to_cancel = total_orders - self.orders_filled
            orders.reverse()
            #print("total orders timeline", total_orders)
            #orders_canceled = []
            #print(total_orders)
            while amount_to_cancel > 0 and len(orders) > 0:
                ##-- ensure that the correct amount gets canceled from each order --##
                c = min(orders[0][0], amount_to_cancel)
                orders[0][0] -= c #this is rounded to 2 decimals because so is self.production_made
                buyer = self.model.schedule.agents[orders[0][1]]

                ##-- order is canceled --##
                if orders[0][0] <= 0:
                    buyer.order_canceled = True
                    del orders[0]
                ##-- order is partially canceled --##
                else:
                    buyer.order_reduced = c
                amount_to_cancel -= c


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
        self.market_share[0] = round( gov.market_shares_normalized_cap[self.unique_id][0],  6)
        self.market_share[1] = round(gov.market_shares_normalized_cap[self.unique_id][1], 6)
        
                

    def climate_damages(self):
        self.flooded = False
        if self.region == 0 and int(self.model.schedule.time) == self.model.shock_time:
            #print('flood')
            self.flooded = True
            #climate_shock = self.model.datacollector.model_vars['Climate_shock'][int(self.model.schedule.time)] +(self.model.s /10)
           # print("I am firm ", self.unique_id, "my capital vintage pre shock is ", self.capital_vintage)

           # print ("capital vintage after shocks ", self.capital_vintage)



    '''
    Migration
    '''
    def migrate(self):
        ##-- stochastic entry barrier to migration process --##
       
        if bernoulli.rvs(0.25) == 1:
        #if self.unique_id % 10 != 0:
           
           # unemployment_subsidy = self.model.datacollector.model_vars["Regional_unemployment_subsidy"][int(self.model.schedule.time)]
            demand_distance = 0 
            demand = self.real_demand_cap
            if demand[1] >= demand[0]:
               demand_distance = ( demand[0] - demand[1]) / (demand[0] + 0.001)
            
               mp = migration.firms_migration_probability( demand_distance, self.region,  self.model)
               if mp> 0:
                   self.region, self.employees_IDs, self.net_worth, self.wage = migration.firm_migrate(mp, self.model, self.region, self.unique_id, self.employees_IDs, self.net_worth, self.wage, 0)


    

    def stage0(self):
        #print('cap')
       # if self.region == 0:
        #    self.CCA_RD()
        if self.lifecycle > 16:
         #   if self.model.schedule.time > 80:
                self.migrate()
            #print(self.region, self.productivity)
            #if self.sales > 0:
           
        if self.model.S > 0:
            self.climate_damages()
          
        self.RD()
        self.calculateProductionCost()
        self.calculatePrice()
        self.advertise()

    def stage1(self):
        #print('cap')
            #self.wage_calculation()
        self.wage_determination()
        self.hire_and_fire_cap()

    # wait for households to do labor search
    def stage2(self):
        pass

    def stage3(self):
        if self.lifecycle > 0:
            self.accounting_orders()
        #self.market_share_normalized()

    def stage4(self):
      #  if self.model.schedule.time >2:
            #self.market_share_normalized()
        pass
        #self.climate_damages()
        #pass
    def stage5(self):
        pass

    def stage6(self):
        self.market_share_normalized()
        
        #if self.model.schedule.time > self.model.start_migration:
         #  self.migrate()
            


        #if self.lifecycle > 0:
            
        self.lifecycle += 1
        #pass