# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 06:22:04 2017

@author: jacobzweig
"""
import random

class Prisoner:
    def __init__(self, id):
        self.id = id
        self.actions = []
        self.responses = []
        self.hasNotFlippedSwitch = True
        
    def decide(self, light):
        if self.hasNotFlippedSwitch and light == 0:
            action = 1
            self.hasNotFlippedSwitch = False
        else:
            action = 0
        
        return action

    def respond(self):
        return "I ain't talkin"
            
    def step(self, light):
        action = self.decide(light) 
        response = self.respond()
        
        # Update history
        self.actions.append(action)
        self.responses.append(response)
        
        return response, action
    
    
class TimRobbins(Prisoner):
    def __init__(self, id, numPrisoners):
        Prisoner.__init__(self, id)
        self.count = 0
        self.numPrisoners = numPrisoners -1
     
    def decide(self, light):
        if light == 1:
            self.count += 1
            action = 1
        else:
            action = 0
        
        return action
        
    def respond(self):
        if self.count >= self.numPrisoners:
            response = "Yes"
        else: 
            response = "No" 
            
        return response
        
        
class Prison:
    def __init__(self):
        self.light = 0 # cold darkness
        self.status = 1
        self.prisonerLog = {}
        
        self.numPrisoners = 100
        self.prisoners = [Prisoner(id) for id in range (self.numPrisoners-1)]
        self.prisoners.append(TimRobbins(self.numPrisoners, self.numPrisoners))
        self.TimsCount = 0
        
        self.allPrisonersHaveVisited= False
    
    def lightSwitch(self, action):
        if action == 1:
            self.light ^= 1
    
    def checkVisitorLog(self):
        uniqueVisitors = self.prisonerLog.keys()
        if len(uniqueVisitors) == self.numPrisoners:
            self.allPrisonersHaveVisited = True
    
    def checkResponse(self, response):
        if response == "Yes" and self.allPrisonersHaveVisited:
            print("You win!")
            self.status = 0
        elif response == "Yes" and not self.allPrisonersHaveVisited:
            print("Electric chair")
            self.status = 0
        else:
            self.status = 1
    
    def checkCount(self):
        if self.prisoners[-1].count != self.TimsCount:
            print("Count updated to {}".format(self.prisoners[-1].count))
            self.TimsCount = self.prisoners[-1].count
    
    def visit(self):
        prisoner = random.choice(self.prisoners)
        response, action = prisoner.step(self.light)
        self.prisonerLog.setdefault(prisoner.id, []).append(action)

        self.lightSwitch(action)
        self.checkVisitorLog()
        self.checkResponse(response)
        self.checkCount()
        
        
if __name__ == "__main__":
    
    folsom = Prison()
    while folsom.status == 1:
        folsom.visit()
    
        