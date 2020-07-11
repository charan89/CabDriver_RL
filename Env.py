# Import routines

import numpy as np
import math
import random
from gym import spaces

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
#loc_mapping = {'1':2,'2':12,'3':4,'4':7,'5':8}
tm=np.load('TM.npy') #Time Matrix
class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.state_space = [[p,q,r] for p in range(m) for q in range(t) for r in range(d)]
        self.state_size=m+t+d
        self.action_space=[(0,0)] + [(i,j) for i in range(5) for j in range(5) if i!=j]
        self.action_size=len(self.action_space)
        
        self.state_init =random.choice(self.state_space)
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        p,q,r=state
        state_vec=np.zeros((1,self.state_size),dtype=np.int16)[0]
        state_vec[int(p)]=1
        state_vec[int(q)+m]=1
        state_vec[int(r)+m+t]=1
        
        return state_vec



    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        #print(location)
        loc_mapping = {'0':2,'1':12,'2':4,'3':7,'4':8}
        requests = np.random.poisson(loc_mapping[str(location)])
        #print('poissin:',requests)
        sitting_idle=[0]
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m), requests)+sitting_idle # (0,0) is not considered as customer request
        #print(possible_actions_index)
        actions = [self.action_space[i] for i in possible_actions_index]

        #actions.append((0,0))
        #print('actions in request is:',actions)

        return possible_actions_index,actions



    def reward_func(self, state, action, Time_matrix=None):
        """Takes in state, action and Time-matrix and returns the reward"""
        t1=state[1]
        d1=state[2]

        if (action[1]==0 and action[0]==0): #When cab is sitting idle not taking rides action is (0,0)
            reward= -C

        else:
            #delC is time taken from cab current location to pickup location
            delC=tm[state[0]][action[0]][t1][d1]
            
            #Updating day and time after reaching pickup location
            d_new,t_new=self.daytimed(d1,t1,delC) 
            
            #DelT is time taken in taking passenger from one location to other
            delT=tm[action[0]][action[1]][t_new][d_new]
            
            reward=R*delT -C*(delT+delC)
            
        return reward


    def daytimed(self,d,t,delT): # function to change day and time when day is changing after 24hours and taking particular time.
        t_new= t + int(delT)
        d_new=d
        if t_new>=23:
            t_new=t_new-23
            d_new=d+1 if d<6 else 0
        else:
            pass
        return (d_new,int(t_new))
   

    def next_state_func(self, state, action, Time_matrix=None):
        """Takes state and action as input and returns next state"""
        loc=state[0]
        t1=state[1]
        d1=state[2]
        
        #print('action in next_state_func:{} and state:{}'.format(action,state))
        if (action[1]==0 and action[0]==0): #When cab is sitting idle not taking rides action is (0,0)
            d_new,t_new=self.daytimed(d1,t1,1)
            time_taken=1
            next_state=(loc,t_new,d_new)
        else:
            #delT1 is time taken from cab current location to pickup location
            delT1=tm[loc][action[0]][t1][d1]
            
            #Updating day and time after reaching pickup location
            d_new,t_new=self.daytimed(d1,t1,delT1)
            
            #print('d:{},t:{}'.format(d_new,t_new))
            #DelT2 is time taken in taking passenger from one location to other
            delT2=tm[action[0]][action[1]][t_new][d_new]
            
            #print('delT2',delT2)
            d_new,t_new=self.daytimed(d_new,t_new,delT2)
            #print('d:{},t:{}'.format(d_new,t_new))
            next_state=(action[1],t_new,d_new)
            time_taken=int(delT1+delT2)
            
        return next_state,time_taken

    def step(self,state,action): #Defining step function inculcating reward and next_state_func in one function
        next_state,time_taken=self.next_state_func(state,action)
        reward=self.reward_func(state,action)
        return next_state,reward,time_taken


    def reset(self):
        return self.action_space, self.state_space, self.state_init
