# model/classes/consumption_good-firm
# A MESA Agent class for consumption good firms (sector 2)

from mesa import Agent

from model.modules import research_and_development as rd
from model.modules import labor_dynamics as ld
from model.modules import goods_market as gm
from model.modules import migration as migration
from model.classes.vintage import Vintage

import random
import statistics
import math


class ConsumptionGoodFirm(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "Cons"
        self.capital_vintage = []
        initial_productivity = []
        for i in range(self.model.initial_number_of_machines):
            initial_productivity.append(self.model.initial_amount_capital)
        for i in range(len(initial_productivity)):
            self.capital_vintage.append(Vintage(initial_productivity[i], amount = self.model.initial_amount_capital))   # each element is a vintage, witinh each machine [0] indicates the amount and [1] the productivity
        self.net_worth = self.model.initial_net_worth            # the amount of money
        self.credit_rationed = False
        self.debt = 0 
        self.lifecycle = 1

        # production #
        self.past_demands = [1,1]                 # a list that will contain all past demand values
        self.feasible_production = 0                  # production considering capital constraint
        self.price = 1
        self.normalized_price = 0
        self.markup = 0.3
        self.cost = 0
        self.total_costs = 0
        self.production_made = 1                      # the production the firm manage to do given its employees 
        self.inventories = 0 
        self.productivity = [self.model.initial_productivity, self.model.initial_productivity]                   #[0] is old and [1] is current average
        self.order_canceled = False
       
        
        
        # investment #
        self.expansion_investment_units = 0
        self.replacement_investment_units = 0    
        self.scrapping_machines = 0 
        self.investment_cost = 0
        #self.total_investment_units = 0 
        self.supplier_id = None
        self.quantity_ordered = 0 

        # labor market #
        self.labor_demand = 0
        self.employees_IDs = []
        self.open_vacancies = False
        self.desired_employees = 0
        self.wage = self.model.initial_wages

        # goods market #
        self.competitiveness = [1,1]    #[0] = region 0, [1] = region 1
        self.market_share = [ 1/self.model.num_firms2 , 1/self.model.num_firms2]  
        self.market_share_history = []# [region 0, region 1]
        self.global_market_share = 0
        self.regional_demand = [0,0]    # demand for my product in [region 0, region 1]
        self.demand = 0                 # total demand
        self.real_demand = 1 
        self.sales = 100
        self.filled_demand = 0
        self.unfilled_demand = 0        # In case firm does not manage to fill all demand
        self.profits = 0  
        self.offers = []                # offers from capital-good firms who advertise to me

        # climate change adaptation #
        self.CCA_resilience = [1, 1 ]     # placeholder value for now
        self.CCA_RD_budget = 0 
        #self.flooded = False

    def CCA_RD(self):
        # step 1: CCA RD budget
        self.CCA_RD_budget, self.CCA_IN, self.CCA_IM = rd.calculateRDBudgetCCA(self.sales, self.net_worth)

        # step 2: innovate
        in_resilience = rd.innovate_CCA(self.CCA_IN, self.CCA_resilience)

        # step 3: imitate
        im_resilience = rd.imitate_CCA(self.CCA_IM,
                                       self.model.ids_firms2,
                                       self.model.schedule._agents,
                                       self.CCA_resilience,
                                       self.region)

        # step 4: choose the best resilience coefficient to adopt
        self.CCA_resilience[0] = max(self.CCA_resilience[0], in_resilience[0], im_resilience[0])
        self.CCA_resilience[1] = max(self.CCA_resilience[1], in_resilience[1], im_resilience[1])




    '''
    Determine how much to invest into new capital based on desired production

    Output: replacement_investment
            expansion_investment
            total_investment

            feasible production
    
    
    '''
    
    
    
    '''
    çapital invetment, choose supplier, make order
    '''
    def capital_investments(self):

        capital_stock =  round(sum(i.amount for i in self.capital_vintage) , 3)       # sum the quantity of each machine, the element [0] in the capital_vintage list
        self.past_demands.append(self.real_demand)                                    # add past demand to the list of old demands
        if len(self.past_demands) > 2:
            local_past_demands = self.past_demands[-3:]                               #save only the last 3
        else:
            local_past_demands = self.past_demands

        #print("ConsumptionGoodFirm:", self.unique_id,"past demands 3:",local_past_demands, "inventories", self.inventories, "capital_stock", capital_stock)

        expected_production = round(statistics.mean(local_past_demands), 3)    # expected production is the mean of the last 3    demands     
        '''
        if self.flooded == True:
         #expected_production = (1 - self.model.s) * expected_production
        '''
        desired_level_inventories = 0.10 * expected_production                #set desired levelof inventories 
        
        if self.model.schedule.time < 5:
            self.inventories = desired_level_inventories          # this is just to the beginning to let the model start smoothly 
        desired_production = max( 0 , expected_production + desired_level_inventories - self.inventories)
        
        ## Calculate how much I can vs I want produce ###
        self.feasible_production = min(desired_production, (capital_stock / self.model.capital_output_ratio))                     #constrain productivity with the capital output ration
        if (self.feasible_production < desired_production ):                                                                      #If I did not have enough capital stock to produce what I wanted
            self.expansion_investment_units = round(((desired_production * self.model.capital_output_ratio) - capital_stock)  )   #I will invest to buy more capital
        else:
            self.expansion_investment_units = 0
        self.feasible_production = round(self.feasible_production, 4)

        self.replacement_investment_units = round(self.calc_replacement_investment() )     
        #print("capital_stock ", capital_stock, "expected production", expected_production, "desired production ", desired_production, "feasible production " , self.feasible_production)
        #print("replacement investment_", self.replacement_investment_units, "expasion investment", self.expansion_investment_units, "capital stock", capital_stock)
        '''
        if self.flooded == True:
            self.expansion_investment_units = 0 
            self.replacement_investment_units = 0

        '''

    def calc_replacement_investment(self):
        # pick supplier as benchmark for cost and productivity
        ratios = [prod/price for [prod,price,u_id] in self.offers]
        if len(ratios) != 0:
            best_i = ratios.index(max(ratios))
            new_machine = self.offers[best_i]  # [productivity, price, seller_id]
        else:
            # pick a random capital-good firm
            supplier_id = random.choice(self.model.ids_firms1)
            supplier = self.model.schedule.agents[supplier_id]
            if supplier.region == self.region:
                new_machine = supplier.brochure_regional
            else:
                new_machine = supplier.brochure_export

        replacement_investment = 0
        for vintage in self.capital_vintage:
            

            # unit cost advantage of new machines
            UCA = self.wage / vintage.productivity - self.wage / new_machine[0]                    #payback rule

            if UCA > 0 and (new_machine[1]/UCA <= 3):  # don't consider if productivity is equal, prevent division by zero
                replacement_investment += vintage.amount
        return replacement_investment
    
    
    
    
    '''
    Choosing the supplier and placing the order of machines 
    '''
    def choose_supplier_and_place_order(self):
        #choose based on highest productivity / price ratio
        #print(self.investment_cost)
        self.investment_cost = 0
        self.quantity_ordered = 0
        self.debt = 0
        ratios = [prod/price for [prod,price,u_id] in self.offers]
        #print( "cons ", self.unique_id, len(ratios), ratios)
        ##--if I don't have any offer, will pick a random supplier--##
        if len(ratios) == 0:
            
            self.supplier_id = random.choice(self.model.ids_firms1)
            supplier_price = self.model.schedule.agents[self.supplier_id].price
            self.model.schedule.agents[self.supplier_id].client_IDs.append(self.unique_id)
            ##--if I have any offer--##
        ##--else the most productive among my offers --##        
        else:
            best_i = ratios.index(max(ratios))
            #print("best_i", best_i)
            # beside productivity and price, advertisers include their unique_id in the brochure
            self.supplier_id = self.offers[best_i][2]  
            supplier_price = self.offers[best_i][1] 
            #print("supplier price ", supplier_price)
        
        supplier = self.model.schedule.agents[self.supplier_id]                                                # record the supplier
        total_number_machines_wanted = self.expansion_investment_units + self.replacement_investment_units
        #print( "expnasion investment units ", self.expansion_investment_units) #count how many machines I want to buy 
        
        ##-- convert units I need into costs --##
        total_investment_expenditure = min( (self.net_worth / 10), total_number_machines_wanted * supplier_price )
        
        ##--only if I am not credit constrained --## 
        if self.net_worth > 0:  
            #Can I finance it with my own funds?
            if total_investment_expenditure <= self.net_worth:
                # if I can I will buy it all and record the investment cost 
                self.quantity_ordered =  total_number_machines_wanted    
                self.investment_cost = total_investment_expenditure
                #print("my investment cost", self.investment_cost, "total investment expenditure", total_investment_expenditure, "total number machines wanted", total_number_machines_wanted)
            
            # Can't pay it all by myself
            else:
                # if I am credit rationed and can't do new debt 
                if self.credit_rationed == True: 
                    # how much I can afford?
                    maximum_quantity_affordable = round(self.net_worth / supplier_price, 2)
                    self.quantity_ordered = maximum_quantity_affordable
                    # expansions investments come first than replacement investments 
                    if maximum_quantity_affordable <= self.expansion_investment_units:
                        self.scrapping_machines = 0
                    elif maximum_quantity_affordable > self.expansion_investment_units:
                        replacement_machines = maximum_quantity_affordable - self.expansion_investment_units
                        self.scrapping_machines = replacement_machines
                    else:
                        print("something wrong with scrapping machines rationed")
                    #self.investment_cost = self.quantity_ordered * supplier_price
                    #if self.investment_cost <0:
                        #print("my investment cost L 210", self.investment_cost, "quantity ordered", self.quantity_ordered,"supplier price",  supplier_price)
                    
                
                #if I cannot finance with my own funds but I am not credit rationed, I do new debt
                elif self.credit_rationed == False:
                    
                    #maximum amount of debt I can do
                    maximum_debt = self.sales * self.model.debt_sales_ratio
                    
                    # max I can afford with debt
                    if total_investment_expenditure <= self.net_worth + maximum_debt:                                                     # Can I finance it with my own funds?
                        self.quantity_ordered =  total_number_machines_wanted    # if I can I will buy it all
                        #self.investment_cost = self.net_worth
                        self.debt = self.net_worth -total_investment_expenditure 
                    #same prodecure as before, see how much I can afford, but here also considering the debt   
                    elif total_investment_expenditure > self.net_worth + maximum_debt:
                        maximum_quantity_affordable = round((self.net_worth + maximum_debt )/ supplier_price, 2)
                        self.quantity_ordered = maximum_quantity_affordable
                        if maximum_quantity_affordable <= self.expansion_investment_units:
                            self.scrapping_machines = 0
                        elif maximum_quantity_affordable > self.expansion_investment_units:
                             replacement_machines = maximum_quantity_affordable - self.expansion_investment_units
                        #self.quantity_ordered += replacemente_machines
                             self.scrapping_machines = replacement_machines
                        else:
                            print("something wrong with scrapping machines NOT rationed")
                    else:
                        print("something wrong in  Not rationed")
                    
                    # check if maximum amount of debt was reached
                    if self.debt >= maximum_debt:
                        self.credit_rationed = True 
                    

                   # if self.investment_cost <0:
                        #print("my investment cost L236", self.investment_cost, "quantity ordered", self.quantity_ordered,"supplier price",  supplier_price)
                        
                      
                    
                else:
                    print("someting wrong with credit rationing ")
                
                # my final investment cost 
                self.investment_cost = self.quantity_ordered * supplier_price
                    
                
                
                
        
        r = self.region
        self.offers = []
        
        ##-- if was able to order something, I add  mi order to the my supplier's list (regional/export depending if it is in my region or not)
        if self.quantity_ordered > 0:
            if supplier.region == r:
                #print(supplier.unique_id, self.quantity_ordered)
                supplier.regional_orders.append([self.quantity_ordered, self.unique_id]) #total_investment is the amount of capital I want to order
            elif supplier.region == 1 - r:
                #print(supplier.unique_id, self.quantity_ordered)
                supplier.export_orders.append([self.quantity_ordered, self.unique_id]) 
            else:
                print("something is wrong with regiona supplier") 

    '''
    Labor demand, hire and fire 
    '''
    def hire_and_fire_cons(self):
        
        '''
        if self.flooded == True:
            damages = min( self.model.S , self.model.S / self.CCA_resilience[0])
            self.productivity[1] = (1 - damages) * self.productivity[1]
        '''
        #keep track of my old productivity, before update it 
        self.productivity[0] = self.productivity[1]
        
        ##-- determine labor demand and then hiring  the functions are in modules ---> labor dynamics ---#
        self.labor_demand, self.productivity[1] = ld.labor_demand(self.capital_vintage, self.feasible_production)
        
        #self.labor_demand += math.floor(self.CCA_RD_budget / self.wage )
        
        self.desired_employees, self.employees_IDs, self.open_vacancies = ld.hire_and_fire(self.labor_demand,                                                                          self.employees_IDs, 
                                                                                           self.open_vacancies, 
                                                                                           self.model, 
                                                                                           self.unique_id)
        '''
        if self.model.schedule.time < 10:
            self.productivity[0]=1
            self.productivity[1]=1
            self.desired_employees = 100
        '''
        
       # print("Hi I am cons firm", self.unique_id, "my prod is ", self.productivity[1]," I want", self.desired_employees, "I have ", len(self.employees_IDs), self.open_vacancies)
    
    
    '''
    costs, price and competitiveness
    '''
    def compete_and_sell(self):
        r = self.region
        
        #self.labor_productivity[0] = ( 1 - self.model.S) * self.labor_productivity[0]
        
        ##-- check my costs and price, the fucntions are in modoules --> goods market --#
        self.cost = round( gm.calc_prod_cost(self.wage, self.productivity[1]), 6)
        if self.model.schedule.time < 10:    #keep markup fixed for the first rounds
            self.markup = 0.3
        else:
            #print( "market share now  ", sum(self.market_share) , "old ", self.market_share_history[-1])
            self.markup = round( self.markup * ( 1 + 0.05 *((self.market_share_history[-1] - self.market_share_history[-2])/ self.market_share_history[-2])) , 6)
            
        #print( "old price ", self.price)
        self.price = round( gm.calc_price(self.cost, self.markup), 5) +  0.000001
        #print( "cons ", self.unique_id,"cost", self.cost, "markup", self.markup, "wage", self.wage , "prod", self.productivity[1], "price", self.price)

        ##-- my competitiveness in both regions --#
        trade_cost = self.model.transport_cost
        self.competitiveness = gm.calc_competitiveness(self.normalized_price, r, trade_cost, self.unfilled_demand)
        
        #print("conf firm", self.unique_id, "normalized price", self.normalized_price, "normalized unfilled demand ", self.unfilled_demand, self.lifecycle, "old market_shares", self.market_share)
        #print("so my competitivenesss is ", self.competitiveness )
        #print("ConsGoodsFirm", self.unique_id, "competiveness is ", self.competitiveness)
    
    
    
    '''
    The firm retrives normalized competiveness from the government, that does it at central level and calculate its market share
    '''
    def market_share_calculation(self):
        
        agents = self.model.schedule.agents
        for i in range(len(agents)):
            if agents[i].type == "Gov" and agents[i].region == 0:
                self.competitiveness[0] = agents[i].comp_normalized[self.unique_id][0]
                self.competitiveness[1] =  agents[i].comp_normalized[self.unique_id][1]
                avg_comp = agents[i].average_normalized_comp
        #print( "cons fir ", self.unique_id, "av comp is ", avg_comp, "my norm comp is ", self.competitiveness)
        # my new market share in both regions
        #print("ConsGodsFirm",self.unique_id, "market shares  normalized old ", self.market_share)
        capital_stock = sum(i.amount for i in self.capital_vintage)
        self.market_share = gm.calc_market_share_cons(self.lifecycle, self.market_share, self.competitiveness, avg_comp, capital_stock)
        #print("market shares not normalized" ,self.market_share)
        # my global market share (currently not used anywhere)
        self.global_market_share = gm.calc_global_market_share(self.market_share)
        
    
    
    
    
    '''
    Calculating individual demand, compared to production made and accounting costs, sales and profits
    '''
    def accounting(self):
        
        ##--get the overall consumption--##
        average_regional_cons=  self.model.datacollector.model_vars["CONSUMPTION"][int(self.model.schedule.time)]
        '''
        if self.flooded == True:
            damages = min( self.model.S , self.model.S / self.CCA_resilience[1])
            average_regional_cons =[  average_regional_cons[0] * ( 1 - self.model.S),  average_regional_cons[1] ]
           
            self.inventories = (1 - damages) * self.inventories
        '''
        
        ##-- check what was my demand in both region --#
        self.regional_demand = [round( average_regional_cons[0] * self.market_share[0], 5) , round( average_regional_cons[1] * self.market_share[1], 5)]
        self.monetary_demand = round(sum(self.regional_demand) , 6)
        
        ##-- convert monetary in real demand --##
        self.real_demand = round( self.monetary_demand / self.price, 6)
        
        
        #print( "Cons firm ", self.unique_id, "lifecycle", self.lifecycle,  "market share ", self.market_share, "regional demand ", self.regional_demand ,"cons firm ", self.unique_id, "real demand ", self.real_demand, "monetary demand", self.monetary_demand, "price", self.price, "av comp", )
        

        ##--labor constraint (was I able to fulfill all my demand or not?) --##
        self.production_made = len(self.employees_IDs)  * self.productivity[1]   # - (math.floor(self.CCA_RD_budget / self.wage ) )) 
        
        #print( "Hi there, cons firm", self.unique_id, "I got", len(self.employees_IDs), "I produced ", self.production_made)
        #print("production made ", self.production_made, "number of employees", len(self.employees_IDs), "productivity", self.productivity[1], "lifecycle", self.lifecycle)           # how much production I managed to do? (important to be staged after labor market) 
        
        ##-- demand I filled, check if I have unfilled demand --##
        self.demand_filled = min(self.real_demand, (self.production_made + self.inventories))   #did I satisfy my demand with my production + inventories or not?
        #print("demand filled:", self.demand_filled)
        if (self.demand_filled ==  (self.production_made + self.inventories)):                  #If I did not fill all the demand with my current production and inventories 
            self.unfilled_demand = self.real_demand - self.production_made - self.inventories   #calculate unfilled demand (to add in competiveness)
            self.inventories = 0                                                                #sold them all
        elif (self.demand_filled == self.real_demand):                                          #If all the demand was filles 
            self.inventories += round((self.production_made - self.real_demand) )               #adjust the inventories 
            self.unfilled_demand = 0                                                            #no unfilled demand  
        else: 
            print("something wrong with demand filled cons ")                                                       
        
        ##--Accounting --#

        self.total_costs = (self.production_made * self.cost - self.CCA_RD_budget )
        self.sales = self.demand_filled * self.price                                            # revenues 
        
        ##-- In case the supplier was not able to fulfill my order and cancelled it ---##
        if self.order_canceled == True:
            self.scrapping_machines = 0 
            self.debt -= self.investment_cost
            self.investment_cost = 0
            self.quantity_ordered = 0
            self.order_canceled = False
        self.profits = round( self.sales - self.total_costs - self.debt * ( 1 + self.model.interest_rate ) , 3)
        
        ##-- if profits are positive, pay taxes --##
        if self.profits > 0:
            self.profits = (1 - 0.3) * self.profits

        # add earnings to net worth
        # from the total income and the market share derive my earnings
        self.net_worth += self.profits - self.investment_cost
        #self.debt = 0
        self.credit_rationed = False
        #print("production made ", self.production_made, "price", self.price, "sales ", self.sales, "real demand ", self.real_demand, "unfilled demand", self.unfilled_demand, "market share ", self.market_share, "profits", self.profits, "net worth", self.net_worth)
        #print("net worth:", self.net_worth)






    '''
    Update firm capital stock, with new machines that are delivered
    '''
    def update_capital(self):
        
        ##-- If I did an order --##
        if self.supplier_id != None and self.quantity_ordered > 0 :
            #print(" hey I am updating my capital")
            self.investment_cost = 0 
            supplier = self.model.schedule.agents[self.supplier_id]
            #roductivity[0]), amount=quantity_ordered)
            #--I add a new element in the vintage, with the amount I bought and the productivity of the supplier --##
            new_machine = Vintage(prod=round(supplier.productivity[0], 3), amount= round(self.quantity_ordered , 2)) #[quantity_ordered, round(supplier.productivity[0])]
            self.capital_vintage.append(new_machine)
            #---replace according to the replacement investment--#
            amount_to_replace = self.scrapping_machines
            for vintage in self.capital_vintage:
                if amount_to_replace > 0:
                    vintage.amount - amount_to_replace
                    amount_to_replace -= vintage.amount
                    if vintage.amount <= 0:
                        self.capital_vintage.remove(vintage)
                #--remove the machines that are too old --#
                vintage.age += 1
                if vintage.age >= vintage.lifetime:
                    self.capital_vintage.remove(vintage)
            self.scrapping_machines = 0         
            #self.capital_vintage = self.capital_vintage[-10:] 
            #print("capital wintage", self.capital_vintage)
            #self.supplier_id = None
        '''
        if self.flooded == True:
           for vintage in self.capital_vintage:
               damages = min( self.model.S, self.model.S / self.CCA_resilience[1])
               vintage.amount = math.floor((1 - damages)* vintage.amount)
        '''
        




    '''
    Wage calculation, can be indexed to difference factors 
    '''
    def wage_determination(self):
        ##--let the model start smoothly--##
        if (self.model.schedule.time < 2):
            return #random.randint(5,15)        #initial value
        else:
            
            ##---retrieve all the data from the datacollector, that I need to calculate the wage (for my region) --##
            r = self.region 
            current_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time)][r]
            previous_unemployment_rate_my_region = self.model.datacollector.model_vars['Unemployment_Regional'][int(self.model.schedule.time) - 1 ][r]
            #print( self.model.datacollector.model_vars['Consumption_firms_av_prod_region_0'][0][int(self.model.schedule.time)])
            current_average_productivty_my_region = self.model.datacollector.model_vars['Regional_average_productivity'][int(self.model.schedule.time)][r]
            previous_average_productivty_my_region = self.model.datacollector.model_vars["Regional_average_productivity"][int(self.model.schedule.time) - 1 ][r]
            minimum_wage = self.model.datacollector.model_vars["Minimum_wage"][int(self.model.schedule.time) ]

            
            #delta_productivity_average = (current_average_productivty_my_region - previous_average_productivty_my_region) / previous_average_productivty_my_region

                
            
            current_average_price =  self.model.datacollector.model_vars["Cosumption_price_average"][int(self.model.schedule.time)][r]
            previous_average_price = self.model.datacollector.model_vars["Cosumption_price_average"][int(self.model.schedule.time) - 1][r]
            
            ##--impose come constraints in case in was zero, to avoid too big jumps --##
            if previous_average_price == 0:
                delta_price_average = 1
            else:
            
                delta_price_average =  (current_average_price - previous_average_price) / previous_average_price
            previous_productivity = self.productivity[0]
            current_productivty  = self.productivity[1]
            
            if previous_unemployment_rate_my_region == 0:
                delta_unemployment = 0
            else:
            
                delta_unemployment = ( current_unemployment_rate_my_region - previous_unemployment_rate_my_region + 0.0001) / (previous_unemployment_rate_my_region + 0.0001)
            
            
            #print("current av prod ", current_average_productivty_my_region,  "previous average prod ", previous_average_productivty_my_region)
            delta_my_productivity = (current_productivty - previous_productivity) / previous_productivity 
            
            if previous_average_productivty_my_region == 0:
                delta_productivity_average = delta_my_productivity
           # elif self.flooded == True:
               # delta_productivity_average = 0
            else:
            
                delta_productivity_average = round ((current_average_productivty_my_region - previous_average_productivty_my_region) /  previous_average_productivty_my_region , 4)
            
            #print("I am cons firm ", self.unique_id,"my wage was ", self.wage, "my region is ", self.region ," delta unemployment  ", delta_unemployment, " deltaregiona productivity ", delta_productivity_average, "delta my productivity ", delta_my_productivity, "productivity ", self.productivity)
          
            ##-- wage is both upper and lower bounded, below by the governmental minimum wage, upper by the myopic zero profits level --##
            self.wage = max(minimum_wage , min( self.price * self.productivity[1], round(self.wage * (1 + 0.5 * delta_my_productivity + 0.5* delta_productivity_average - 0.0* delta_unemployment + 0.0 * delta_price_average), 5)))
            #print("my wage now is ", self.wage)
            #if self.model.schedule.time < 10:
                #self.wage = 1
          
            
            
            
            
            
    '''    
    def hire_and_fire_cons(self):
        self.productivity[0] = self.productivity[1]
        self.labor_demand, self.productivity[1] = ld.labor_demand(self.capital_vintage, self.feasible_production)
        
        #print("my labor demand is ", self.labor_demand, "my productivity is ", self.productivity[1])
        self.desired_employees, self.employees_IDs, self.open_vacancies = ld.hire_and_fire(self.labor_demand, 
                                                                                           self.employees_IDs, 
                                                                                           self.open_vacancies, 
                                                                                           self.model, 
                                                                                           self.unique_id)
        #print("Hi I am cons firm", self.unique_id, " I want", self.desired_employees, "I have ", len(self.employees_IDs))
    ''' 
    
    
    
    
    
    '''
    The firm retrives normalized price and unfilled demand (used to calculate competitiveness) from the government, that does it at central level
    '''
    def price_demand_normalized(self):
        agents = self.model.schedule.agents
        for i in range(len(agents)):
            if agents[i].type == "Gov" and agents[i].region == 0:
                self.normalized_price = round(agents[i].norm_price_unfilled_demand[self.unique_id][0], 6)  + 0.00001
                self.unfilled_demand = round(agents[i].norm_price_unfilled_demand[self.unique_id][1], 6)
        #print("Cons firm", self.unique_id, "region ", self.region,"normalized price", self.normalized_price, "normalized unfilled demand", self.unfilled_demand )


    '''
    The firm retrives normalized market shares from the government, that does it at central level
    '''
    def market_share_normalized(self):
        agents = self.model.schedule.agents
        for i in range(len(list(agents))):
        #for i in range(len(agents)):
            if agents[i].type == "Gov" and agents[i].region == 0:
                self.market_share[0] = round(agents[i].market_shares_normalized[self.unique_id][0], 6)
                self.market_share[1] = round(agents[i].market_shares_normalized[self.unique_id][1], 6)
        
        self.market_share_history.append(round(sum(self.market_share), 6))
        #print( "new market share ", self.market_share, "cons", self.unique_id, "region", self.region ) #, self.market_share_history)
    

        


    '''
    Damage due to natural disasters
    '''
    def climate_damages(self):
        self.flooded == False
        if self.region == 0 and self.model.schedule.time == self.model.shock_time:
            self.flooded = True
            #if self.model.S > 0:
            #climate_shock = self.model.datacollector.model_vars['Climate_shock'][int(self.model.schedule.time)] +(self.model.s /10)
           # print("I am firm ", self.unique_id, "my capital vintage pre shock is ", self.capital_vintage)
               
           # print ("capital vintage after shocks ", self.capital_vintage)




    '''
    Migration
    '''
    def migrate(self):
        
        ##----stocastic barrier to migration --##
        if ( random.uniform(0,1) < 0.05 or self.lifecycle < 3):
            demand = self.regional_demand
            ##--calculation of migration probability and process, modules ---> migration --##
            mp = migration.firms_migration_probability( demand, self.region, self.wage, self.profits, self.model)
            #print("mp is", mp
            self.region, self.employees_IDs, self.net_worth = migration.firm_migrate(mp, self.model, self.region, self.unique_id, self.employees_IDs, self.net_worth, self.wage)
        
        #if self.lifecycle < 10:# and self.unique_id % 10 != 0:
            #demand =  self.model.datacollector.model_vars["Regional_profits_cons"][int(self.model.schedule.time)]
        

                
                
        




    
    '''
    Firm exit and entry

    def entry_exit(self):
        if self.lifecycle >= 1:
            # if my market share everywhere is zero
            if self.market_share[0] + self.market_share[1] < 0.0001:
                print("Firm", self.unique_id,"exiting...")

                # remove myself
                if self.region == 0:
                    self.model.ids_region0.remove(self.unique_id)
                elif self.region == 1:
                    self.model.ids_region1.remove(self.unique_id)
                self.model.ids_firms2.remove(self.unique_id)
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)

                # replace myself with a new firm, it takes over my unique_id
                i = self.unique_id
                a = ConsumptionGoodFirm(i, self.model)
                self.model.schedule.add(a)
                self.model.ids_firms2.append(i)

                # place the new firm in the grid
                y = self.random.randrange(2)
                self.model.grid.place_agent(a, (0,y))
                a.region = y
                if y == 0:
                    self.model.ids_region0.append(i)
                elif y == 1:
                    self.model.ids_region1.append(i)

                print("Consumption good firm", self.unique_id, "left the market")

                return True
            return False
    '''



    def stage0(self):
       # if self.region == 0: 
        #   self.CCA_RD()
        if self.lifecycle > 0:
            self.update_capital()
            self.capital_investments()
            if self.replacement_investment_units > 0 or self.expansion_investment_units > 0:
                self.choose_supplier_and_place_order()

    def stage1(self):
        if self.lifecycle > 0:
            self.wage_determination()
            self.hire_and_fire_cons()

    # wait for households to perform labor search
    def stage2(self):
        pass

    def stage3(self):
        if self.lifecycle > 0:
            self.price_demand_normalized()
            self.compete_and_sell()

    def stage4(self):
        if self.lifecycle > 0:
            self.market_share_calculation()
        
    def stage5(self):
        if self.lifecycle> 0:
            self.market_share_normalized()
            self.accounting()
            
        #self.market_share_normalized()
        

    def stage6(self):
        
        #if self.model.schedule.time > self.model.start_migration:
         #   self.migrate()
        '''
        if  self.model.S > 0:
            self.climate_damages()
            if (self.model.schedule.time >= self.model.shock_time and self.model.schedule.time <= ( self.model.shock_time + 10)):
                if random.uniform(0,1) < 0.05:
                    self.migrate()
        
        #self.market_share_normalized()
        
        #if self.unique_id % 10 != 0:
        #if self.lifecycle > 0:
        '''   
        self.lifecycle += 1