import numpy as np
import random
from ..utils import Validator

class MP():

    @Validator.unique_list(param_name='S')
    @Validator.shape(param_name='P')
    def __init__(self, S:list, P:np.array):
        """
        param :
            S
            n_states
            P

            (Param that does not show here are initialised in validators)
        """

        #functions
        self.absorbing_state = self.find_absorbing_states()
        self.states_encoder,self.states_decoder = self.encode_decode(param="S")

    def encode_decode(self,param:str):
        encoder = {}
        decoder = {}
        for i,p in enumerate(getattr(self,param)):
            encoder[p]=i
            decoder[i]=p

        return encoder,decoder

    # function to find absorbing states
    def find_absorbing_states(self,P=None):
        absorbing_states = []
        P = self.P if P is None else P
        for i, row in enumerate(P):
            if np.count_nonzero(row) == 1 and row[i] == 1:
                absorbing_states.append(i)
        return absorbing_states

    # function to generate a random_episode
    @Validator.bounds(min_val=0,max_val="n_states",param_name="start_state")
    def generate_random_episode(self,start_state:int, max_lenght:int):
        random_episode = [start_state]
        current_state = start_state
        for _ in range(max_lenght):
            next_state = random.randint(0,self.n_states-1)
            while self.P[current_state,next_state] == 0:
                next_state = random.randint(0,self.n_states-1)
            random_episode.append(next_state)
            if next_state in self.absorbing_state:
                break
            current_state = next_state
        return random_episode