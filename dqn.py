# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:04:01 2017

@author: jacobzweig
"""

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DDRQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DDRQN,self).__init__()
                        
        self.gru1 = nn.GRU(input_size,hidden_size)
        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hc = None):          
        x, hc = self.gru1(x,hc)
        x = x[-1]
        x = F.relu(self.embedding(x))
        out = self.out(x)

        return out, hc
    
   
class learner:
    def __init__(self, inputSize, hiddenSize, nActions):
        self.model = DDRQN(inputSize, hiddenSize, nActions)
        self.nActions = nActions
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0
        self.last_sync = 0
    
    
    def select_action(self, state):
        state = torch.from_numpy(state)
        sample = random.random()
        
        # Set the present epsilon threshold and anneal
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > self.eps_threshold:
            # Choose Max(Q|s)
            action = self.model(Variable(state, volatile=True).type(FloatTensor))[0].data.max(1)[1].view(1, 1)
        else:
            # Choose a random action
            action =  LongTensor([[random.randrange(self.nActions)]])
            
        return action.numpy()
        
    
    
    def optimize_model(self, sample):
    
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        if sample.next_state is not None:
            next_state = Variable(torch.from_numpy(sample.next_state), volatile=True).type(FloatTensor)
        state_batch = Variable(torch.from_numpy(sample.state)).type(FloatTensor)
        action_batch = Variable(torch.LongTensor(np.array([[sample.action]]))).view(-1,1)
        reward_batch = Variable(torch.FloatTensor(np.array([[sample.reward]]))).view(-1,1)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch)[0].gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        if sample.next_state is not None:
            next_state_values = self.model(next_state)[0].max(1)[0]
        else:
            next_state_values = torch.zeros(1)

        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss
        
        
        
        
