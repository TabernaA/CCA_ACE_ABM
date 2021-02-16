'''

model/modules/research_and_development.py

Functions for the Research & Development process
calculateRDBudget()
innovate()
imitate()

'''

import math
import random
import bisect

from scipy.stats import bernoulli
from scipy.stats import beta


''' Calculate total RD budget, IM budget, IN budget
PARAMETERS
S : sales in the previous period
N : money currently available to the firm (net worth)

DEFAULT PARAMETERS
v : fraction of previous sales, 0.05 default
e : dividing between IN and IM, 0.5 default
'''
def calculateRDBudget(S, N, v=0.04, e=0.5):
    if S > 0:
        rd_budget = v*S
    elif N < 0:
        rd_budget = 0  
    else:
        rd_budget = v*N

    in_budget = e*rd_budget
    im_budget = (1-e)*rd_budget     

    return rd_budget, in_budget, im_budget



    ''' Innovation process
    PARAMETERS
    IN   : my innovation budget
    prod : my productivity, [A,B] pair

    DEFAULT PARAMETERS
    Z     : budget scaling factor for Bernoulli draw, 0.3 default
    a     : alpha parameter for Beta distribution, 3 default
    b     : beta parameter for Beta distribution, 3 default
    x_low : lower bound of support vector, -0.15 default
    x_up  : upper bound of support vector,  0.15 default
    '''
def innovate(IN, prod, Z=0.3, a=3, b=3, x_low=-0.15, x_up=0.15):
    in_productivity = [0,0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1-math.exp(-Z*IN)
    #p1 = 1-math.exp(-Z*IN/2)
    #print( "P ", p , "b", b)
    if bernoulli.rvs(p) == 1:
        # new machine productivity (A) from innovation
        a = (1 + x_low + beta.rvs(a,b)*(x_up-x_low))
        in_productivity[0] = prod[0] * a
        #print(a)
        

    #if bernoulli.rvs(p1) == 1: # new production productivity (B) from innovation
       # a1 = (1 + x_low + beta.rvs(a,b)*(x_up-x_low)) 
        in_productivity[1] = prod[1] * a

    return in_productivity




    ''' Imitation process
    PARAMETERS
    IM       : imitation budget
    firm_ids : a list of ids of the firms in my sector (self.model.ids_firms1/2)
    agents   : a dictionary of all agents in the model.schedule._agents
    prod     : my productivity, [A,B] pair
    reg      : my region (0 or 1)

    DEFAULT PARAMETERS
    Z : budget scaling factor for Bernoulli draw, 0.3 default
    e : distance scaling factor for firms from another region, 1.5 default
    '''

def imitate(IM, firm_ids, agents, prod, reg, Z=0.3, e=2):
    im_productivity = [0,0]

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

    return im_productivity




def calculateRDBudgetCCA(S, N, v=0.005, e=0.5):
    if S > 0:
        rd_budget = v*S
    elif N < 0:
        rd_budget = 0  
    else:
        rd_budget = v*N

    in_budget = e*rd_budget
    im_budget = (1-e)*rd_budget     

    return rd_budget, in_budget, im_budget

'''
R : CCA resilience coefficient
'''
def innovate_CCA(IN, R, Z=0.3, a=3, b=3, x_low=-0.10, x_up=0.10):
    in_R = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1-math.exp(-Z*IN/2)
    p1 = 1-math.exp(-Z*IN/2)
    
    
    b0 = bernoulli.rvs(p)
    b1 = bernoulli.rvs(p1)
    #print( "P ", p , "b", b)
    if b0 == 1:
        # new resilience coefficient from innovation
        in_R[0] = R[0]*(1+x_low + beta.rvs(a,b)*(x_up-x_low))      #this is labor_productivity resilience
    
    if b1 == 1:
         in_R[1] = R[1]*(1+x_low + beta.rvs(a,b)*(x_up-x_low))   # this is capital_stock_resilience
    
    
    return in_R





def imitate_CCA(IM, firm_ids, agents, R, reg, Z=0.3, e=1.5):
    im_R = [0, 0]

    # Bernoulli draw to determine success (1) or failure (0)
    p = 1-math.exp(-Z*IM)
    if bernoulli.rvs(p) == 1:
        # store imitation probabilities and the corresponding firms
        imiProb = []
        imiProbID = []

        #for all capital-good firms, compute inverse Euclidean distances
        for id in firm_ids:
            firm = agents[id]
            distance = math.sqrt(pow(R[0] - firm.CCA_resilience[0], 2) + \
                       pow(R[0] - firm.CCA_resilience[0], 2))
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
                im_R[0] = firm.CCA_resilience[0]
                im_R[1] = firm.CCA_resilience[1]

    return im_R