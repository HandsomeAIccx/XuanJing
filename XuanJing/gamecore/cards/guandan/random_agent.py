import random
import numpy as np


class RandomAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self):
        pass

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # print("random:",state['legal_actions'])
        if len(state.legal_action) > 1:
            idx = random.randint(0, len(state.legal_action) - 1)
        else:
            idx = 0
        return state.legal_action[idx]

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        # probs = [0 for _ in range(self.num_actions)]
        # for i in state['legal_actions']:
        #     probs[i] = 1/len(state['legal_actions'])

        info = {}
        # info['probs'] = {state['raw_legal_actions'][i]: probs[state['legal_actions'][i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info
