import numpy as np
from sampling_algorithms import SamplingAlgorithms,Bandit
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',type=str,default='')
    parser.add_argument('--algorithm',type=str,default='')
    parser.add_argument('--randomSeed',type=int,default=0)
    parser.add_argument('--epsilon',type=float,default=0.0)
    parser.add_argument('--horizon',type=int,default=0)
    args = parser.parse_args()

    filepath = args.instance
    algorithm = args.algorithm
    seed_val = args.randomSeed
    epsilon = args.epsilon
    horizon = args.horizon

    np.random.seed(seed_val)
    sampler = SamplingAlgorithms(filepath,algorithm,seed_val,epsilon,horizon)
    sampler.evaluate_regret()
    print("{}, {}, {}, {}, {}, {}".format(filepath,algorithm,str(seed_val),str(epsilon),str(horizon),str(sampler.regret)))