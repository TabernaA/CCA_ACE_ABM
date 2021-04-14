'''
model/modules/goods_market.py

Functions for Goods Market
calculate_price()
calculate_competitiveness()
avg_competitiveness()
calc_market_share()
calc_global_market_share()
calc_avg_regional_wage()
aggregate_regional_employment() --probably not needed
domestic_demand()
export_demand()

'''


''' 
Calculate unit cost of production

PARAMETERS
wage : my wage
prod : my labor productivity 
'''
def calc_prod_cost(wage, prod):
    return wage / prod



''' 
2.5 (7)
Calculate unit price

PARAMETERS
cost : unit cost of production

DEFAULT PARAMETERS
markup : 0.05 default
'''
def calc_price(cost, markup):
    return round((1+markup) * cost,6)



'''
2.5 (8)
Calculate competitiveness of a firm in both regions [0,1]

PARAMETERS
price      : firm's unit price
my_r       : firm's home region

DEFAULT PARAMETERS
trade_cost : trade cost for the foreign region, 0.3 default
'''
'''
def calc_competitiveness_old(price, my_r, trade_cost):
	if my_r == 0:
		return [1/price, 1/price*(1+trade_cost)]
	elif my_r == 1:
		return [1/price*(1+trade_cost) , 1/price ]
'''    
    
    
def calc_competitiveness(price, my_r, trade_cost, trade_cost_exp, unfilled_demand ):
	if my_r == 0:
		return [ round(-1 *price  - 1  * unfilled_demand , 8), round( -1 * price*(1+trade_cost)  -1 * unfilled_demand , 8), round( -1 * price * (1 + trade_cost_exp)  - 1  * unfilled_demand  , 8)]
	elif my_r == 1:
		return [ round(-1 * price * (1+trade_cost) -1 * unfilled_demand , 8) , round(-1 * price  -1  * unfilled_demand, 8), round( -1 * price*(1+ trade_cost_exp + trade_cost) - 1  * unfilled_demand  , 8)]

'''
2.5 (9)
Average competitiveness for my sector in region r

PARAMETERS
r      : region 0 or 1
agents : model.schedule._agents
sector : self.type, a string describing the sector
'''
def avg_competitiveness(r, agents, sector):
	final_comp = 0
	for i in range(len(agents)):
		a = agents[i]
		if a.region == r and a.type == sector:
			final_comp += \
			a.competitiveness[0] * a.market_share[0] + \
			a.competitiveness[1] * a.market_share[1]
	return final_comp




'''
2.5 (10)
Market share of a Consumtion Goods firm in both regions [0,1]

PARAMETERS
MS_prev  : my market share from the previous time step
comp     : my competitiveness in both regions
comp_avg : average competitiveness for my sector in both regions
K        : my capital stock

DEFAULT PARAMETERS
X : scaling factor for level of competitiveness, 0.2 default
'''
def calc_market_share_cons( model, lifecycle, MS_prev, comp, comp_avg, K , r ,chi=1):
    
    min_ms = 0.00001
    
    if (lifecycle == 0):
        
        K_total =  model.datacollector.model_vars['Capital_Regional'][int(model.schedule.time)]

        ms0 = K / K_total[0]
        ms1 = K / K_total[0]
        ms2  = K / K_total[0]
        return [max(ms0, min_ms), max(ms1, min_ms), max(ms2, min_ms) ]
    
     # some minimum market share needed to stay
    ms0 = MS_prev[0] * (1 + chi*(comp[0]/(comp_avg[0])))
    ms1 = MS_prev[1] * (1 + chi*(comp[1]/(comp_avg[1])))
    ms2 = MS_prev[2] * (1 + chi*(comp[2]/(comp_avg[2])))
    
    return [ max( min_ms ,round(ms0, 5)), max( min_ms ,round(ms1, 5)),  max( min_ms ,round(ms2, 5))]
   #return [ round(max(ms0, 0.99*min_ms), 5), round(max(ms1, 0.99*min_ms), 5)]




	#return [ round(max(ms0, 0.99*min_ms), 5), round(max(ms1, 0.99*min_ms), 5)]

'''
Calculate market share of a Capital Goods firm: IN PROGRESS
PARAMETERS
orders : my list of orders
model  : self.model
'''
'''
def calc_market_share_cap(orders, model):
	total_orders = sum(orders)
	total_market0 = 0
	total_market1 = 0
	for i in range(len(model.schedule._agents)):
		a = model.schedule._agents[i]
		if a.type == "Cap":
			total_market0 += sum(a.orders)
			
			if a.region == 0:
				total_market0 += sum(a.orders)
			# need to separate the two markets
			# total_orders from each region
			elif a.region == 1:
				total_market1 += sum(a.orders) 
			

	return [total_orders / (total_market0 + 0.00001),total_orders / (total_market0 + 0.00001)]

'''


''' 
2.5 (11)
global market share of a firm

PARAMETERS
MS : market share [0,1] pair
'''
def calc_global_market_share(MS):
	return (MS[0] + MS[1]) / 2



''' 
average regional wage for region r in [0,1]
'''
def calc_avg_regional_wage(r, agents):
	wage_t = 0
	num_firms = 0
	for i in range(len(agents)):
		if agents[i].region == r and agents[i].type != "Household":
			wage_t += agents[i].wage
			num_firms += 1
	return wage_t / num_firms





# NOT DONE
'''Calculate aggregate regional employment for region r'''
def aggregate_regional_employment(r, agents):
	ARE = 0
	for i in range(len(agents)):
		if agents[i].type == "Household" and agents[i].employer != None and agents[i].region == r:
			ARE += 1
	return ARE




''' 
2.5 (12), (13)
Demand for my product in both regions [0,1]

PARAMETERS
wage : average wage in both regions
AE   : aggregate employment in both regions
AU   : aggregate unemployment in both regions
RUS  : regional unemployment subsidy
MS   : my market share in both regions
'''
def regional_demand(wage, AE, AU, RUS, MS):
	return [(wage[0] * AE[0] * MS[0]) + (RUS[0] * AU[0] * MS[0]) ,  (wage[1] * AE[1] * MS[1]) + (RUS[1] * AU[1] * MS[1])]