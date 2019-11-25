import numpy as np
import random
import argparse
from env.tabular_env import *
from algo.agent import *
from util.general_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ep', type=int,
                        default=5000, help="Number of episodes to train on.")
    parser.add_argument('--num_mc', type=int,
                        default=1, help="Number of MC samples to run.")
    parser.add_argument('--max_steps', type=int,
                        default=20, help="Maximum number of steps per ep.")
    parser.add_argument('--alpha', type=int,
                        default=0.5, help="Averaging rate.")
    parser.add_argument('--gamma', type=float,
                        default=0.9, help="Discount rate.")
    parser.add_argument('--eps', type=float,
                        default=0.1, help="Epsilon greedy param.")
    parser.add_argument('--batch_size', type=int,
                        default=20, help="Number of transitions to train on")
    parser.add_argument('--seed', type=int,
                        default=1234, help="Set random seed")
    parser.add_argument('--grid_rows', type=int,
                        default=5, help="Num rows in env")
    parser.add_argument('--grid_cols', type=int,
                        default=5, help="Num cols in env")
    parser.add_argument('--prior_std', type=float,
                        default=100.0, help="Initial prior std")
    parser.add_argument('--like_std', type=float,
                        default=1.5, help="Fixed likelihood std")

    args = parser.parse_args()
    return args

def main():

    """ parse arguments """
    args = parse_args()
    num_ep = args.num_ep
    num_mc = args.num_mc
    max_steps = args.max_steps
    alpha = args.alpha
    gamma = args.gamma
    eps = args.eps
    batch_size = args.batch_size
    seed = args.seed
    prior_std = args.prior_std
    like_std = args.like_std
    np.random.seed(seed)
    random.seed(seed)

    """ make environment and agent """
    env = tabular_env((args.grid_rows, args.grid_cols))
    agent = gaussian_ocbo_agent(env, alpha, eps, prior_std, like_std)
    # agent = q_learning_agent(env, alpha, eps)

    state = env.reset()
    action = None

    # import pdb; pdb.set_trace()
    for ep_idx in range(num_ep):
        if 'ocbo' in agent.id:
            orig_state_idx, orig_action = agent.ocbo_selection()
            orig_state = env.idx_to_state(orig_state_idx)
            action = orig_action
            env.curr_state = orig_state
            print('EP {}: OCBO reset initial state to {}, action to {}'.format(
                                            ep_idx, orig_state, orig_action))
            # action = agent.policy(state)
        else:
            orig_state = state
            orig_action = agent.get_policy_action(env.state_to_idx(state))

        """ play out a single episode"""
        curr_ep_states = [orig_state]
        curr_ep_actions = [orig_action]
        curr_ep_rewards = []
        curr_ep_done = []
        # curr_gamma = 1
        done = False
        curr_ep_steps = 0
        while not done:
            if action is None:
                action = agent.get_policy_action(env.state_to_idx(env.curr_state))
            """ env step """
            prev_state = env.curr_state
            ns, reward, done, info = env.step(action)
            """ store transition in agent's replay buffer """
            agent.buffer.add(prev_state, action, reward, ns, done)

            curr_ep_steps+=1
            curr_ep_rewards.append(reward)
            # curr_gamma *= gamma
            curr_ep_states.append(ns)
            curr_ep_done.append(done)
            if (curr_ep_steps >= max_steps) or done:
                break
            action = int(agent.get_policy_action(env.state_to_idx(ns)))
            curr_ep_actions.append(action)
        print(curr_ep_rewards)
        """ just completed playing a single episode """

        # """ BEGIN: update q_fn based on current episode trajectory """
        # curr_ep_disc_rewards = []
        # last_item = 0
        # for item in curr_ep_rewards[::-1]:
        #     curr_item = item + gamma*last_item
        #     curr_ep_disc_rewards.append(curr_item)
        #     last_item = curr_item
        #
        # curr_ep_disc_rewards = curr_ep_disc_rewards[::-1]
        #
        # for s,a,r in zip(curr_ep_states, curr_ep_actions, curr_ep_disc_rewards):
        #     s_idx = env.state_to_idx(s)
        #     state_best_obs = np.max(agent.best_obs[s_idx])
        #     # max_obs = np.max(mc_obs)
        #     # if state_best_obs < max_obs:
        #     if state_best_obs < r:
        #         agent.best_obs[s_idx].fill(r)
        #         # for i in range(len(agent.best_obs[s_idx])):
        #         #     agent.best_obs[s_idx][i] = r
        #     agent.update_q_fn(s_idx, a, r)
        # """ END: update q_fn based on current episode trajectory """

        """ BEGIN: update q_fn based on batch from replay buffer """
        # import pdb; pdb.set_trace()
        for _ in range(1000):
            tr_sample = agent.buffer.get_batch(batch_size)
            for s, a, r, ns, done in tr_sample:
                s_idx = env.state_to_idx(s)
                ns_idx = env.state_to_idx(ns)
                if 'ocbo' in agent.id:
                    state_best_obs = np.max(agent.best_obs[s_idx])

                if done:
                    end_r = r
                else:
                    end_r = r + gamma*np.max(agent.q_fn[ns_idx])

                if 'ocbo' in agent.id:
                    if state_best_obs < end_r:
                        agent.best_obs[s_idx].fill(end_r)
                        # for i in range(len(agent.best_obs[s_idx])):
                        #     agent.best_obs[s_idx][i] = r
                agent.update_q_fn(s_idx, int(a), end_r)
        """ END: update q_fn based on batch from replay buffer """

        # """ BEGIN: update q_fn based on current episode trajectory """
        # import pdb; pdb.set_trace()
        # curr_ep_len = len(curr_ep_actions)
        # for i in range(curr_ep_len):
        #     s = curr_ep_states[i]
        #     if curr_ep_len == 1:
        #         ns = None
        #     else:
        #         ns = curr_ep_states[i + 1]
        #     a = curr_ep_actions[i]
        #     r = curr_ep_rewards[i]
        #     d = curr_ep_done[i]
        #
        #     s_idx = env.state_to_idx(s)
        #     ns_idx = env.state_to_idx(ns) if ns is not None else None
        #     state_best_obs = np.max(agent.best_obs[s_idx])
        #     if d:
        #         end_r = r
        #     else:
        #         end_r = r + gamma*np.max(agent.q_fn[ns_idx])
        #
        #     if state_best_obs < end_r:
        #         agent.best_obs[s_idx].fill(end_r)
        #         # for i in range(len(agent.best_obs[s_idx])):
        #         #     agent.best_obs[s_idx][i] = r
        #     agent.update_q_fn(s_idx, a, end_r)
        # """ END: update q_fn based on current episode trajectory """

        old_policy = np.copy(agent.policy)
        agent.update_policy()
        new_policy = agent.policy

        # if np.sum(old_policy-new_policy)==0: break

        if ep_idx % 25 ==0:
            agent.print_policy()

    agent.print_policy()

if __name__=='__main__':
    main()
