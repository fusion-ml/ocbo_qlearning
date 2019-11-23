import numpy as np
from numpy.random import normal as normal
from .replay_buffer import *

DIVIDE = "="*100

class agent(object):
    def __init__(self, env, buffer_size=1000):
        self.env = env
        self.num_actions = self.env.num_actions
        self.num_states = self.env.num_states
        self.q_fn_dim = [self.num_states, self.num_actions]

        """ q_fn is a list of arrays, dim is num_states x num_actions"""
        self.q_fn_num_elem = np.product(self.q_fn_dim)
        self.q_fn = np.array([[0.0 for _ in range(self.num_actions)]
                      for _ in range(self.num_states)])
        # self.policy = [None for _ in range(self.num_states)]
        self.policy = np.array([0.0 for _ in range(self.num_states)])
        self.q_fn_idx = np.arange(self.q_fn_num_elem).reshape(self.q_fn_dim)

        self.buffer = ReplayBuffer(buffer_size)

    def _sa_idx(self, state_1, state_2, action):
        return self.q_fn_idx[state_1, state_2, action]

    def update_policy(self):
        for state_idx in range(self.num_states):
            self.policy[state_idx] = np.argmax(self.q_fn[state_idx])


class gaussian_ocbo_agent(agent):
    def __init__(self, env, alpha, eps):
        super(gaussian_ocbo_agent, self).__init__(env)
        self.id = 'gaussian_ocbo'
        """ initial prior """
        self.init_mean = 0.0
        self.init_std = 0.05

        self.alpha = alpha
        self.eps = eps

        """ prior over mean of q(s,a)'s """
        self.prior = np.array([[[self.init_mean, self.init_std]
                       for _ in range(self.num_actions)]
                       for _ in range(self.num_states)])

        self.posterior = self.prior.copy()
        # self.q_fn = np.array([[0 for _ in range(self.num_actions)]
        #               for _ in range(self.num_states)])

        """ assume we know the likelihood std = 0.5 """
        self.like_std = 0.5

        """ best observed values of mean of q for each state """
        # self.best_obs = [0 for _ in range(self.num_states*self.num_actions)]
        # self.best_obs, self.q_fn, self.post_sample all same size
        self.best_obs = np.empty_like(self.q_fn)
        self.best_obs.fill(-np.inf)


        self.init_q_fn()
        self.update_policy()

    def init_q_fn(self):
        # initial prior is mean=0, std=1
        for s in range(self.num_states):
            for a in range(self.num_actions):
                prior_sample = normal(self.prior[s,a,0],
                                        self.prior[s,a,1])
                self.q_fn[s,a] = prior_sample

    def get_policy_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(self.num_actions, 1)[0]
        return self.policy[state]

    def ocbo_selection(self):

        # for i in range(np.product(self.q_fn_dim)):
        post_sample = np.zeros_like(np.array(self.q_fn))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                sa_sample = normal(self.posterior[s, a, 0],
                                        self.posterior[s, a, 1])
                post_sample[s, a] = sa_sample
                # import pdb; pdb.set_trace()

        # post_sample = np.array([normal(x[0], x[1]) for x in
        #                         self.posterior]).reshape(self.q_fn_dim)

        improvement = post_sample - self.best_obs

        next_sa_idx = np.argmax(improvement)
        next_s = next_sa_idx // self.num_actions
        next_a = next_sa_idx % self.num_actions
        assert(np.max(improvement)==improvement[next_s, next_a])

        ### DEBUG PRINT
        # print('Posterior')
        # print(self.posterior)
        # print(DIVIDE)
        print('Post Sample')
        print(post_sample)
        print(DIVIDE)
        print('Best Obs')
        print(self.best_obs)
        print(DIVIDE)
        print('Improvement')
        print(improvement)
        print(DIVIDE)
        print('Next chosen state, action')
        print(next_s, next_a)
        print(DIVIDE)
        print('\n'*2)
        ### DEBUG PRINT

        """ next_s is the state idx of the next state, likewise for next_a """
        return next_s, next_a

    def update_q_fn(self, state, action, obs):
        """
        obs: list of observations from q(s,a)
        """
        old_mean, old_std = self.posterior[state, action]
        num_obs = 1
        sum_obs = np.sum(obs)
        mean_obs = sum_obs/num_obs
        new_std = 1/((1/(old_std**2+1e-10))+(num_obs/(self.like_std**2+1e-10)))
        new_mean = new_std*((old_mean/(old_std**2+1e-10)) + (sum_obs/(self.like_std**2+1e-10)))
        # new_mean = (1/((1/old_std**2)+(num_obs/self.like_std**2))) * ((old_mean/old_std**2) + (sum_obs/self.like_std**2))

        self.posterior[state, action] = [new_mean, new_std]

        orig_q = self.q_fn[state][action]
        self.q_fn[state][action] = orig_q + self.alpha*(mean_obs - orig_q)

    def print_policy(self):
        s_counter = 0
        temp_actions = []
        for s in self.policy:
            temp_actions.append(self.env.action_labels[s])
            s_counter += 1
            if (s_counter % self.env.num_cols) == 0:
                print(temp_actions)
                temp_actions = []
