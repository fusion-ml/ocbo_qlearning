import numpy as np
from util.general_utils import *

# def idx_to_state(state_idx, num_rows, num_cols):
#     r_idx = state_idx // num_cols
#     c_idx = state_idx % num_rows
#     return (r_idx, c_idx)
#
# def state_to_idx(state,  num_rows, num_cols):
#     return (state[0] * num_cols) + state[1]

class tabular_env(object):
    """test tabular environment that has same api's as gym"""
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.num_actions = 4
        self.num_rows = self.grid_size[0]
        self.num_cols = self.grid_size[1]
        self.num_states = np.product(self.grid_size)
        # self.state_idx = np.arange(self.num_states).reshape(self.grid_size)
        self.action_idx = np.arange(self.num_actions)
        self.curr_state = None
        self.done = False

        # self.action_labels = {0:'up', 1:'right', 2:'down', 3:'left'}
        self.action_labels = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}

        self.reset()

        """ special environment design """
        # self.A = [(1,1)]
        # self.B = [(1,3)]
        self.A = [(1, 1)]
        self.B = []

    def reset(self):
        rand_state_idx = np.random.choice(self.num_states,1)[0]
        print(rand_state_idx)
        rand_state = self.idx_to_state(rand_state_idx)
        print('Reset to state {}'.format(rand_state))
        self.curr_state = rand_state
        return self.curr_state

    def _action_result(self, state, act_idx):
        """ action is the action index """
        action = self.action_labels[act_idx]
        if action=='U':
          return (state[0]-1, state[1])
        if action=='R':
          return (state[0], state[1]+1)
        if action=='D':
          return (state[0]+1, state[1])
        if action=='L':
          return (state[0], state[1]-1)

    def step(self, action):
        if self.curr_state in self.A:
            reward = 10
            new_state = (2,1)
            self.curr_state = new_state
            self.done = True
            done_info = 'A'
        elif self.curr_state in self.B:
            reward = 5
            new_state = (2,3)
            self.curr_state = new_state
            self.done = False
            done_info = 'B'
        else:
            hyp_new_state = self._action_result(self.curr_state, action)
            if (hyp_new_state[0]<0) or (hyp_new_state[1]<0) or \
              (hyp_new_state[0] > self.grid_size[0] - 1) or \
              (hyp_new_state[1] > self.grid_size[1] - 1):
                reward = -1
                new_state = self.curr_state
            else:
                reward = 0
                new_state = hyp_new_state
                self.curr_state = new_state
            self.done = False
            done_info = 'None'


        return new_state, reward, self.done, done_info

    def idx_to_state(self, state_idx):
        r_idx = state_idx // self.num_cols
        c_idx = state_idx % self.num_cols
        return (r_idx, c_idx)

    def state_to_idx(self, state):
        return (state[0] * self.num_cols) + state[1]
