#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:39:56 2017

@author: jacobzweig
"""


import random
import numpy as np
from collections import deque, namedtuple
import dqn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Prisoner:
    def __init__(self, id, numPrisoners):
        self.id = id
        self.numPrisoners = numPrisoners

        self.actionHistory = [0] #initialize with a 'none' action
        self.stateHistory = deque(maxlen=10)
        
        self.actions = {0:'none', 1:'flip switch', 2:'tell'}
        
    def act(self, light, learner):
        
        # One hot encode ID
        state_id = np.zeros((1,self.numPrisoners))
        state_id[0, self.id] = 1
        
        # One hot encode previous action
        state_previousAction = np.zeros((1, len(self.actions.keys())))
        state_previousAction[0, self.actionHistory[-1]] = 1
        
        currState = np.concatenate((np.array(light, ndmin=2), state_id, state_previousAction), axis=1)
        self.stateHistory.append(currState)
        
        # Make state vector of timepoints x 1 x features
        state = np.array(self.stateHistory)
        action = learner.select_action(state)[0][0]
        
        return state, action

    def step(self, light, learner):
        state, action = self.act(light, learner)
        
        # Update history
        self.actionHistory.append(action)
        return state, action

        
class Prison:
    def __init__(self, learner, nPrisoners):
        self.learner = learner
        self.numPrisoners = nPrisoners
        self.resetPrison()

    def resetPrison(self):
        self.nVisits = 0
        self.light = 0
        self.prisonerLog = {}
        
        # logs for training
        self.stateLog = []
        self.rewardLog = []
        self.actionLog = []

        self.prisoners = [Prisoner(id, self.numPrisoners) for id in range (self.numPrisoners)]
        
        self.allPrisonersHaveVisited = False

    def flipSwitch(self):
        self.light ^= 1
    
    def checkVisitorLog(self):
        uniqueVisitors = self.prisonerLog.keys()
        if len(uniqueVisitors) == self.numPrisoners:
            self.allPrisonersHaveVisited = True
    
    def doAction(self, action):
        # If they say everyone has been there and they're correct, they win
        if action == 2 and self.allPrisonersHaveVisited:
            reward = 1.0
            print("You win!")
            
        # If they say everyone has been there and they're wrong, everyone dies
        elif action == 2 and not self.allPrisonersHaveVisited:
            reward = -1.0
            print("Everyone Died!")
        
        # If they flip the switch, update the light
        elif action == 1: 
            self.flipSwitch()
            reward = 0.0
            
        # If the do nothing, continue
        else:
            reward = 0.0
            
        return reward
    
    
    def visit(self):
        prisoner = random.choice(self.prisoners)
        state, action = prisoner.step(self.light, self.learner)
                
        self.checkVisitorLog() # update our visitor log to see if everyone has visited
        reward = self.doAction(action)
        
        # If we have more than 0 visits, we train
        if self.nVisits > 0:
            if self.rewardLog[-1] == 0:
                sample = Transition(self.stateLog[-1], self.actionLog[-1], state, self.rewardLog[-1])       
            else:
                # If we've gotten a non-zero reward, it's a terminal state
                sample = Transition(self.stateLog[-1], self.actionLog[-1], None, self.rewardLog[-1])  
                
            loss = self.learner.optimize_model(sample)
            # print(loss.data.numpy())
        
        # Append to historical logs
        self.prisonerLog.setdefault(prisoner.id, []).append(action)
        self.stateLog.append(state)
        self.rewardLog.append(reward)
        self.actionLog.append(action)
        
        # Keep track of how many visits we've had
        self.nVisits += 1
        
        # If we've processed a non-zero reward, we reset the prison 
        if self.rewardLog[-1] != 0:
            self.resetPrison()
            
        
        
