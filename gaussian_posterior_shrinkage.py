import numpy as np
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_std', type=float, default=1.0,
                        help='Initial prior std of Gaussian')
    parser.add_argument('--init_mu', type=float, default=0.0,
                        help='Initial prior mean of Gaussian')
    parser.add_argument('--like_std', type=float, default=0.05,
                        help='Fixed std of Gaussian likelihood')
    parser.add_argument('--init_like_mu', type=float, default=0.0,
                        help='Initial mean of Gaussian likelihood')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    post = [args.init_mu, args.init_std]
    like = [args.init_like_mu, args.like_std]

    obs = input()
    while obs.isnumeric():
        obs = float(obs)
        print('Obs received:{}'.format(obs))
        new_std = 1/((1/(post[1]**2))+(1/(like[1]**2)))
        new_mean = new_std*((post[0]/(post[1]**2)) + (obs/(like[1]**2)))
        post = [new_mean, new_std]
        print('New posterior: mean {}, std {}'.format(new_mean, new_std))
        obs = input()

if __name__=='__main__':
    main()
