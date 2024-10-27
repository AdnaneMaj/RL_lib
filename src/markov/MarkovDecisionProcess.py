from .MarkovRewardProcess import MRP
from typing import List,Callable
from utils import Validator
import numpy as np

class MDP(MRP):
    @Validator.unique_list("S")
    @Validator.unique_list("A")
    @Validator.shape(param_name="P")
    @Validator.Reward()
    def __init__(self, S, P, gamma, R, A):
        """
        P here is P[s,s',A]
        """
        super().__init__(S,P,gamma,R)
        self.A = A
        self.n_actions = len(A)
        self.actions_encoder,self.actions_decoder = self.encode_decode("A")
        self.P = P

        #constant
        n = self.n_states
        a = self.n_actions
        self.appropriate_shape.update({2:(n,a),3:(n,n,a)})
        self.set_R_()

    @Validator.bounds(min_val=0,max_val="n_actions",param_name="A")
    @Validator.bounds(min_val=0,max_val="n_states",param_name="S")
    @staticmethod
    def random_policy(self,A:int=0,S:int=0):
        """
        A policy is a distribution ovec actions space given states
        """
        random_policy = np.random.rand(self.n_actions,self.n_states)
        random_policy = random_policy / random_policy.sum(axis=0, keepdims=True)
        return random_policy
    
    def MDP_under_policy(self,policy)->MRP:
        """
        An MDP under policy is a MRP
        """
        #check if R if of form (s,a)
        if self.R_v != 2:
            raise ValueError("Can't be performed with this form of reward")
        
        S = self.S
        gamma = self.gamma
        new_P = np.einsum('as,sra->sr', policy, self.P)
        if self.R_v == 2:
            new_R = np.einsum('as,sa->s',policy,self.R)
        elif self.R_v == 3:
            new_R = np.einsum('as,sra->sr',policy,self.R)

        return MRP(S,gamma,new_P,new_R)
    
    def state_value_function(self,s:int,policy):
        """
        The expected return starting from state s, 
        and then following policy π
        """
        if self.R_v:
            V = np.einsum('as,sa->s',policy,np.dot(V,P))
        









        


