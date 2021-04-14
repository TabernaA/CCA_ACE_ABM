# model/classes/model.py
# A MESA Model class, KSModel
seed_value = 12345678

import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed=seed_value)
from mesa import  Model
#from mesa.time import StagedActivation
#from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
#from model.modules.datacollection import DataCollector
#from mesa.batchrunner import BatchRunner
#import random
from model.classes.schedule import StagedActivationByType
from model.classes.capital_good_firm import CapitalGoodFirm
from model.classes.consumption_good_firm import ConsumptionGoodFirm
from model.classes.household import Household
from model.classes.government import Government
from model.modules.data_collection import *
from model.modules.data_collection_2 import *

#   def __init__(self, F1 = 5, F2= 10, H= 10, B= 1, T= 0.02, S = 0.1, width=1, height=2):
class KSModel(Model):
    def __init__(self, F1= 50, F2= 250, H= 3500, B= 1, T= 0.1, seed = seed_value, S=1, width=1, height=2):
        self.num_firms1 = F1
        self.num_firms2 = F2
        self.num_households = H
        self.num_agents = F1 + F2 + H
        self.s = 0
        self.running = True

        # lists to keep track of the unique_ids of each agent type. For quick lookup
        self.ids_firms1 = []
        self.firms1 = []
        self.ids_firms2 = []
        self.firms2 = []
        self.firms_1_2 = []
        self.ids_households = []
        self.households = []
        self.ids_region0 = []
        self.ids_region1 = []
        self.governments = []
        self.list_firms = []
        random.seed(seed)
        np.random.seed(int(seed))
        self.reset_randomizer(seed)
       # self.pr_migration_f = 0.1
       # self.pr_migration_h = 0.15
        


        # stages of activation
        stage_list = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
        #stage_list = ["step"]
        self.schedule = StagedActivationByType(self, stage_list)
        #self.schedule = StagedActivation(self, stage_list)         # schedule._agents is an OrderedDict() ; type: Dict[unique_id, Agent]
        self.initial_amount_capital = 2
        #self.running = True
        #self.grid = MultiGrid(width, height, True)
        self.initial_productivity = 1
        self.initial_number_of_machines = 3
        self.initial_wages = 1
        #self.capital_depreciation_rate = 0.02
        self.capital_output_ratio = B
        self.initial_net_worth = 100
       # self.start_migration = 30
        self.debt_sales_ratio = 2
        self.interest_rate = 0.01
        self.S = S /10
        self.shock_time = 75
        self.beta_a = 3
        self.beta_b = 27


        # transport cost for [region0, region1]
        self.transport_cost = T 
        self.transport_cost_RoW =  2 * T


        for i in range(self.num_agents):
            # add capital-good firms
            if i <= self.num_firms1 // 2:
                a = CapitalGoodFirm(i, self)
                self.schedule.add(a)
                self.ids_firms1.append(i)    # keep track of the unique_id
                self.firms1.append(a)
                a.region = 0
                self.ids_region0.append(i)
            elif i > self.num_firms1 // 2 and i <= self.num_firms1:
                a = CapitalGoodFirm(i, self)
                self.schedule.add(a)
                self.ids_firms1.append(i)    # keep track of the unique_id
                self.firms1.append(a)
                a.region = 1
                self.ids_region1.append(i)
            # add consumption-good firms
            elif i > self.num_firms1 and i <= self.num_firms1 + self.num_firms2 // 2:
                a = ConsumptionGoodFirm(i, self)
                self.schedule.add(a)
                self.ids_firms2.append(i)
                self.firms2.append(a)
                a.region = 0
                self.ids_region0.append(i)
            elif i > self.num_firms1 + self.num_firms2 // 2 and i <= self.num_firms1 + self.num_firms2:
                a = ConsumptionGoodFirm(i, self)
                self.schedule.add(a)
                self.ids_firms2.append(i)
                self.firms2.append(a)
                a.region = 1
                self.ids_region1.append(i)
            elif i >  self.num_firms1 + self.num_firms2 and i <=  self.num_firms1 + self.num_firms2 + self.num_households //2:
                a = Household(i, self)
                self.schedule.add(a)
                self.ids_households.append(i)
                self.households.append(a)
                a.region = 0
                self.ids_region0.append(i)
            elif i > self.num_firms1 + self.num_firms2 + self.num_households //2 and i <= self.num_agents:
                a = Household(i, self)
                self.schedule.add(a)
                self.ids_households.append(i)
                self.households.append(a)
                a.region = 1
                self.ids_region1.append(i)
                

            # add households
            else:
                print( 'something wrong in initialization')
        
        
        #print(self.firms_1_2)
        for i in range(2):  
            u_id = len(self.schedule.agents)                  
            a = Government(u_id, self, i)   #passing region=i into the constructor
            self.schedule.add(a)
            self.governments.append(a)
           # self.grid.place_agent(a, (0,i))
            self.num_agents += 1
            #a.region = i
        self.firms_1_2 = self.firms1 + self.firms2 

        '''
        # create two regional governments
        
        for i in range(2):                    
            a = Government(self.num_agents + i, self)
            self.schedule.add(a)
            self.grid.place_agent(a, (0,i))
        '''


        # data collection
        self.datacollector = DataCollector(

            model_reporters = {
                #"Productivity_A1" : productivity_a1, # machine productivity in sector 1
                #"Productivity_B1" : productivity_b1,  # labor productivity in sector 1
                #"Climate_shock" : climate_shock_generator,
             
                # "Unemployment_Region0" : regional_unemployment_rate_region0,
                "Regional_Costs" : regional_costs, 
                "Average_Salary" : regional_average_salary,
                "Average_Salary_Capital" : regional_average_salary_cap,
                "Average_Salary_Cons" : regional_average_salary_cons,
                "Competitiveness_Regional" : regional_average_competitiveness,
                "Aggregate_Employment" : regional_aggregate_employment,
                "Aggregate_Unemployment" : regional_aggregate_unemployment,
                "Unemployment_Regional" : regional_unemployment_rate,
                "Regional_unemployment_subsidy": regional_unemployment_subsidy,
                #"Population_Regional" : regional_population_total,
                "Population_Regional_Households" : regional_population_households,
               # "Population_Region_0_Households" :regional_population_households_region_0,
                #"Population_Region_0_Cons_Firms":regional_population_cons_region_0,
                #"Population_Region0_Households" : regional_population_households_region_0,
                "Population_Regional_Cons_Firms" : regional_population_cons, 
                #"Population_Region0_Cons_Firms" : regional_population_cons_region_0,
                "Population_Regional_Cap_Firms" : regional_population_cap,
                "Capital_Regional" : regional_capital,
               # "Investmen_units" : investment_units,
                #"Investment_units" : investment_units,
               # "Capital_firms_av_prod" : productivity_capital_firms_average,
                #"Capital_firms_av_prod_region_1" : productivity_capital_firms_region_1_average,
                "Regional_average_productivity" : productivity_firms_average,
              #  "Consumption_firms_av_prod" : productivity_consumption_firms_average,
                "Cosumption_price_average" : price_average_cons,
                 "Capital_price_average" : price_average_cap,
                "GDP": gdp,
                #"RD_CCA_INVESTMENT" : RD_CCA_investment,
                #"Average_CCA_coeff" :  RD_coefficient_average,
                #'Sectoral_debt': sectoral_aggregate_debt, 
                # 'Sectoral_liquid_assets': sectoral_aggregate_liquid_assets,
                "GDP_cons": gdp_cons,
                "GDP_cap": gdp_cap,
                'Real GDP coastal': real_gdp_cons_reg_0,
                'Real GDP internal': real_gdp_cons_reg_1,
                'Unemployment rate coastal':  regional_unemployment_rate_coastal,
                'Unemployment rate internal':  regional_unemployment_rate_internal,
                'Price coastal':  price_average_cons_coastal,
                'Price internal':  price_average_cons_internal,
                'INVESTMENT coastal': investment_coastal,
                'INVESTMENT inland': investment_inland,
                'CONSUMPTION coastal': consumption_coastal,
                'CONSUMPTION inland': consumption_inland,
                'Inland productivity average': av_productivity_inland,
                'Coastal productivity average': av_productivity_coastal,
                'Inland productivity growth': gr_productivity_inland,
                'Coastal productivity growth': gr_productivity_coastal,
                
                
                
                "INVESTMENT" : investment,
                "INVENTORIES" : inventories,
                #"Regional_fiscal_balance" : regional_balance,
               "Regional_sum_market_share" : regional_aggregate_market_share,
                  "Regional_average_profits_cons" : regional_average_profits_cons,
                  "Regional_average_profits_cap": regional_average_profits_cap,
                  "Regional_average_NW" : regional_average_nw,
                 # "Capital_price_average" : price_average_cap,
                  # :regional_profits_cap,
                  "Regional_profits_cons" : regional_profits_cons,
                # "labor check " : consumption_labor_check,
                 "Cons_regional_IDs" : cons_ids_region,
                # "Firms_regions" :firm_region,
                 "Minimum_wage" : regional_minimum_wage,
                 "orders": quantity_ordered,
                 "Top_prod": top_prod,
                 "Top_wage": top_wage,
                 'MS_exp': ms_exp, 
                 #'Demand_exp_ratio' : demand_export_rate,
                 "CONSUMPTION" : consumption,
                # 'Sales_firms' : sales_firms
               # "Market_share_normalized" : market_share_normalized,
                #"LD_cap" : ld_cap,
                #"LD_cons" : ld_cons,
                "Debt" : debt,
                'GDP total' : gdp_SA,
                'Price total': price_SA,
                'INVESTMENT total' : investment_SA,
                'CONSUMPTION total' : consumption_SA,
                'Unemployment total' : regional_unemployment_rate_SA,
                 "Population_Region_0_Households" :regional_population_households_region_0,
                 "Population_Region_0_Cons_Firms":regional_population_cons_region_0
                #"MS_track" : ms_region
            }


             )


        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        #self.list_firms.append(self.firms_1_2)
        
            
        '''
        agent_reporters={"Net worth": "net_worth", 
                             'Size' :     lambda x: len(x.employees_IDs) if  x.type == "Cons" else None,
                             'Vintage':   lambda x: len(x.capital_vintage) if  x.type == "Cons" else None,
                             'Price':     'price',
                             'Wage':       'wage',
                             'Prod':     'productivity',
                            # 'Ms':        'market_share',
                             'Region':    lambda x: x.region if  x.type == "Cons" else None,
                             'Lifecycle': lambda x: x.lifecycle if  x.type == "Cons" else None}
        '''