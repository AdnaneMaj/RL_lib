from .MarkovProcess import MP
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import math
from utils import Validator
from typing import Tuple

class MRP(MP):
    @Validator.unique_list("S")
    @Validator.bounds(min_val=0,max_val=1,param_name='gamma')
    @Validator.Reward()
    def __init__(self, S, P, gamma:float, R:dict):
        """
        input:
            - gamma : [0,1] coeifficient of reduction
            - R : Two versions, either R(s) or R(s,s')
            - in version one : R[state_id], in version two R[(state1_id,state2_id)]
        """
        super().__init__(S, P)
        self.gamma = gamma

        #constantes
        n = self.n_states
        self.value_vector = np.zeros(n)
        self.appropriate_shape = {0:(n,),1:(n,n)}
    
    def set_R_(self):
        """
            Initialize ordered_R with appropriate shape using R_v
            It will be generalised to work with MRP also
        """
        def set_index(key):
            #Get the index
            if self.R_v==0:
                index = (self.states_encoder[key[0]])
            elif self.R_v==1:
                index = (self.states_encoder[key[0]],self.states_encoder[key[1]])
            elif self.R_v==2:
                index = (self.states_encoder[key[0]],self.actions_encoder[key[1]])
            elif self.R_v==3:
                index = (self.states_encoder[key[0]],self.states_encoder[key[1]],self.actions_encoder[key[1]])
            return index
        
        shape = self.appropriate_shape[self.R_v] #Get the apropriate shape
        ordered_R = np.zeros(shape) #Initialise ordered_R

        # Populate ordered_R based on R_v (version)
        for key,value in self.R.items():
            index = set_index(key)
            ordered_R[index]=value

        self.R_ = ordered_R
    
    def value_function(self,start_state:int, zero: int=1e-9):
        """Direct way of doing it"""

        T = int(math.log(zero,self.gamma)+2)+1 #The maximum T ( gamma**(T_max-2) -> 0 )
        episodes = self.generate_random_episode(start_state,T) #Generate T episodes (at max) starting from start state
        T = min(T,len(episodes)) #Set the T to min of T_max or len(episodes)

        gamma_vector = self.gamma**np.arange(T-1) #(1,g,g**2,...,g**(T-2))

        return np.dot(gamma_vector,self.R_)
    
    #iterative way to compute it
    def value_function_vector(self,start_state_vector = None,verbose=False,**kwargs):
        """
        Either T_max or epislon

        """

        # Validate the kwargs to ensure only one valid key is provided
        if len(kwargs) != 1 or not set(kwargs).issubset({'T_max', 'epsilon'}):
            raise ValueError(
                f"Provide exactly one parameter: either 'T_max' or 'epsilon'. Got: {list(kwargs.keys())}"
            )
        
        if start_state_vector is None:
            start_state_vector = np.zeros((self.n_states,))
        
        def iterate(V):
            """
            Performs a single iteration of the value function.
            if self.R_v = 0 
                v(s) = R(s)+gamma.sum(P(s,s')*v(s'))
            else :
               v(s) = sum(R(s,s')+gamma.P(s,s')*v(s'))
            """
            if self.R_v==0:
                V = self.R_+ self.gamma * np.dot(self.P,V)
            elif self.R_v==1:
                V = self.R_.sum(axis=1) + self.gamma *np.dot(self.P,V)
            if verbose:
                print(V)
            return V

        # Initialize the value vector
        V = np.array(start_state_vector)

        # Choose the correct stopping criterion
        if 'T_max' in kwargs:
            T_max = kwargs['T_max']
            for _ in range(T_max):
                V = iterate(V)
            return V

        elif 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            while True:
                V_new = iterate(V)
                if np.linalg.norm(V - V_new) <= epsilon:
                    break
                V = V_new
            return V_new


    def bellman_solver(self,solution_type:str="Direct"):
        """
        Multiple ways to solve the bellman equation
        As :
            - Direct solution
            - Dynamic programming
            - Monte-carlo
            - Temporal-Difference learning
        """
        N = self.n_states
        return np.linalg.inv((np.eye(N)-self.gamma*self.P))@self.R_