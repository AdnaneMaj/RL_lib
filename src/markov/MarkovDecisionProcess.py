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
    
    def find_absorbing_states(self, P=None):
        P = self.P.sum(axis=-1) if P is None else P
        return super().find_absorbing_states(P)

    
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
    
    @staticmethod
    def extract_values(*args, **kwargs):
        verbose = MRP.extract_values(*args, **kwargs)
        policy = kwargs['policy']
        return verbose,policy

    def iterate_V(self,V,*args,**kwargs):
        """
        Performs a single iteration of the state value function.
        if self.R_v = 0 
            v(s) = R(s)+gamma.sum(P(s,s')*v(s'))
        else :
            v(s) = sum(R(s,s')+gamma.P(s,s')*v(s'))
        """
        #extract verbose
        verbose,policy = MDP.extract_values(*args,**kwargs)

        if self.R_v==2:
            V_new = np.einsum('as,sa->s',policy,self.R_+self.gamma*np.einsum('ssa,s->sa', self.P, V))
        if self.R_v==3:
            V_new = np.einsum('as,sa->s',policy,self.R_.sum(axis=1)+self.gamma*np.einsum('ssa,s->sa', self.P, V))
    
        if verbose:
            print(V_new)
        return V_new
    
    def iterate_Q(self,Q,*args,**kwargs):
        """
        Performs a single iteration of the state action function.
        """
        #extract verbose
        verbose,policy = MDP.extract_values(*args,**kwargs)

        if self.R_v==2:
            Q_new = self.R_+self.gamma*np.einsum('ssa,as,sa->sa', self.P, policy, Q)
        elif self.R_v==3:
            Q_new = self.gamma*np.einsum('ssa,as,sa->sa', self.P+self.R_.sum(axis=1), policy, Q)
        if verbose:
            print(Q_new)
        return Q_new








        


