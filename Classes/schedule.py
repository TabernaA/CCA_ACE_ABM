# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:55:28 2020

@author: TabernaA
"""

'''
model/classes/schedule.py
A custom activation class
'''
from collections import defaultdict
from mesa.time import StagedActivation
class StagedActivationByType(StagedActivation):
    def __init__(self, model, stage_list, shuffle=False, shuffle_between_stages=False):
       super().__init__(model, stage_list, shuffle, shuffle_between_stages)
       self.agents_by_type = defaultdict(dict)



    def add(self, agent):
        self._agents[agent.unique_id] = agent
        agent_type = type(agent)
        #if agent.unique_id in self._agents:
         #   raise Exception(
          #     "Agent with unique id {0} already added to scheduler".format(
           #         repr(agent.unique_id)))

            
        self._agents[agent.unique_id] = agent
        self.agents_by_type[agent_type][agent.unique_id] = agent 
    
    
    def remove(self, agent):
        
       del self._agents[agent.unique_id]
       agent_type = type(agent)
       del self.agents_by_type[agent_type][agent.unique_id]
       
    
    def step(self):
        for stage in self.stage_list:
            
            for agent_type in self.agents_by_type:
                agent_keys = list(self.agents_by_type[agent_type].keys())
                if self.shuffle:
                    self.model.random.shuffle(agent_keys)
                for agent_key in agent_keys:
                    #try:
                    getattr(self._agents[agent_key], stage)()
                    #except KeyError:
                        #print("someone left and didn find it")# Run stage
            if self.shuffle_between_stages:
                self.model.random.shuffle(agent_keys)
            self.time += self.stage_time
        self.steps += 1