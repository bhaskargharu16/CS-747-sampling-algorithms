import random,sys
import numpy as np
def KL(p,q):
    ans = 0
    if p == 1:
        ans =  np.log(1/q)
    elif p == 0:
        ans =  np.log(1/(1-q))
    else:
        if q != 1:
            ans =  p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
        else:
            ans =  np.inf
    return ans
    
class Bandit(object):
    def __init__(self,true_mean):
        self.true_mean = true_mean
        self.empirical_mean = 0
        self.probabilities = np.array([]).astype(float) 
        self.distribution = np.array([]).astype(float)
        self.history = []
        self.ucb_val = 0
        self.kl_ucb_val = 0
        self.success_so_far = 0
        self.failure_so_far = 0
        self.beta_sample = 0
    
    def update(self):
        self.empirical_mean = ((len(self.history)-1) * self.empirical_mean + self.history[-1])/len(self.history)
        return self
    
    def sample(self):
        reward = np.random.binomial(1, self.true_mean)
        if reward:
            self.success_so_far += 1
        else:
            self.failure_so_far += 1
        self.history.append(reward)
        self.update()
        return self

    def populate_ucb(self,timestep):
        self.ucb_val = self.empirical_mean + np.sqrt((2*np.log(timestep))/len(self.history))
        return self
    
    def binary_search_kl_ucb(self,uat, bound):
        low = self.empirical_mean
        high = 1
        while low < high:
            mid = float((low+high)/2)
            val = uat * KL(self.empirical_mean,mid)
            if val <= bound and bound - val <= 0.0001:
                return mid
            elif val > bound:
                high = mid
            else:
                low = mid
        return low

    def populate_kl_ucb(self,timestep):
        if self.empirical_mean == 1:
            self.kl_ucb_val = 1
            return self
        uat = len(self.history)
        c = 3
        bound = np.log(timestep) + c * np.log(np.log(timestep))
        self.kl_ucb_val = self.binary_search_kl_ucb(uat,bound)
        return self
    
    def sample_beta_rv(self):
        self.beta_sample = np.random.beta(self.success_so_far+1,self.failure_so_far+1)
        return self

    def populate_probabilities(self,probabilities):
        self.probabilities = probabilities
        return self
    
    def create_init_distribution(self):
        frequency_dict = {}
        for prob in list(self.probabilities):
            if prob in frequency_dict:
                frequency_dict[prob] += 1
            else:
                frequency_dict[prob] = 1
        unique_sorted_probabilities = np.sort(np.array(list(set(self.probabilities))))
        self.distribution = np.zeros(unique_sorted_probabilities.shape).astype(float)
        for idx,prob in enumerate(list(unique_sorted_probabilities)):
            self.distribution[idx] = frequency_dict[prob]/len(self.probabilities)
        self.distribution = self.distribution * ((unique_sorted_probabilities) ** self.success_so_far)
        self.distribution = self.distribution * ((1-unique_sorted_probabilities) ** self.failure_so_far)
        self.distribution = self.distribution/sum(self.distribution)
        return self

class SamplingAlgorithms(object):
    def __init__(self,inst,al,rs,ep,hz):
        self.instance = inst
        self.arms = [] #list of bandit objects
        self.al = al
        self.seed = rs
        self.ep = ep
        self.hz = hz
        self.reward = 0
        self.regret = 0
        self.permutation = []
        self.all_probabilities = []
        self.distribution_matrix = []
    
    def populate_arms(self):
        file = open(self.instance,'r')
        lines = file.readlines()
        probabilities = []
        for i,line in enumerate(lines):
            probability = float(line.strip())
            self.arms.append(Bandit(probability))
            probabilities.append(probability)
        self.all_probabilities = np.array(probabilities)
        self.permutation = np.array(list(set(probabilities)))
        return self
    
    def get_best_empirical_arm(self):
        pivot = self.arms[0]
        for arm in self.arms[1:]:
            if arm.empirical_mean > pivot.empirical_mean :
                pivot = arm
        return pivot
    
    def get_best_true_arm(self):
        pivot = self.arms[0]
        for arm in self.arms[1:]:
            if arm.true_mean > pivot.true_mean :
                pivot = arm
        return pivot
    
    def get_best_ucb_arm(self):
        pivot = self.arms[0]
        for arm in self.arms[1:]:
            if arm.ucb_val > pivot.ucb_val :
                pivot = arm
        return pivot
    
    def get_best_kl_ucb_arm(self):
        pivot = self.arms[0]
        for arm in self.arms[1:]:
            if arm.kl_ucb_val > pivot.kl_ucb_val :
                pivot = arm
        return pivot
    
    def get_best_thompson_arm(self):
        pivot = self.arms[0]
        for arm in self.arms[1:]:
            if arm.beta_sample > pivot.beta_sample :
                pivot = arm
        return pivot

    def sample_one_by_one(self):
        for i in range(min(len(self.arms),self.hz)):
            self.arms[i].sample()
            self.reward += self.arms[i].history[-1]
        return self

    def epsilon_greedy(self):
        self.sample_one_by_one()
        for idx in range(len(self.arms),self.hz):
            if np.random.binomial(1, self.ep) : #explore
                chosen_arm = np.random.choice(self.arms,1)[0]
                chosen_arm.sample()
                self.reward += chosen_arm.history[-1]
            else : #exploit
                best_arm = self.get_best_empirical_arm()
                best_arm.sample()
                self.reward += best_arm.history[-1]
        return self
    
    def ucb(self):
        self.sample_one_by_one()
        for idx in range(len(self.arms),self.hz):
            self.arms = [x.populate_ucb(idx+1) for x in self.arms]
            best_arm = self.get_best_ucb_arm()
            best_arm.sample()
            self.reward += best_arm.history[-1]
        return self
    
    def kl_ucb(self):
        self.sample_one_by_one()
        for idx in range(len(self.arms),self.hz):
            self.arms = [x.populate_kl_ucb(idx+1) for x in self.arms]
            best_arm = self.get_best_kl_ucb_arm()
            best_arm.sample()
            self.reward += best_arm.history[-1]
        return self
    
    def thompson_sampling(self):
        for idx in range(self.hz):
            self.arms = [x.sample_beta_rv() for x in self.arms]
            best_arm = self.get_best_thompson_arm()
            best_arm.sample()
            self.reward += best_arm.history[-1]
        return self

    def initialise_distribution_matrix(self):
        self.distribution_matrix = np.zeros((len(self.permutation),len(self.arms))).astype(float)
        for i in range(len(self.arms)):
            self.distribution_matrix[:,i] = self.arms[i].distribution
        return self

    def get_best_arm_thompson_hint(self, samples):
        max_sample_yet = max(samples)
        all_max = [(i,j) for i, j in enumerate(samples) if j == max_sample_yet]
        while len(all_max) != 1:
            ind=(self.distribution_matrix.cumsum(0) > np.random.rand(self.distribution_matrix.shape[1]).reshape(1,len(self.arms))).argmax(0)
            samples = self.permutation[ind]
            sample_selected = [(x[0],samples[x[0]]) for x in all_max]
            all_max = [sample_selected[0]]
            for pair in sample_selected[1:]:
                if all_max[-1][1] < pair[1]:
                    all_max = [pair]
                    continue
                if all_max[-1][1] == pair[1]:
                    all_max.append(pair)
        return all_max[0][0]     
    
    def thompson_sampling_with_hint(self):
        self.sample_one_by_one()
        self.permutation = np.sort(self.permutation)
        self.arms = [x.populate_probabilities(self.all_probabilities) for x in self.arms]
        self.arms = [x.create_init_distribution() for x in self.arms]
        self.initialise_distribution_matrix()
        for idx in range(len(self.arms),self.hz):
            ind=(self.distribution_matrix.cumsum(0) > np.random.rand(self.distribution_matrix.shape[1]).reshape(1,len(self.arms))).argmax(0)
            arm_samples = self.permutation[ind]
            best_arm_index = self.get_best_arm_thompson_hint(arm_samples)
            best_arm = self.arms[best_arm_index]
            best_arm.sample()
            arm_reward = best_arm.history[-1]
            if arm_reward :
                self.distribution_matrix[:,best_arm_index] = self.distribution_matrix[:,best_arm_index] * np.array(self.permutation)
            else:
                self.distribution_matrix[:,best_arm_index] = self.distribution_matrix[:,best_arm_index] * (1-np.array(self.permutation))
            self.distribution_matrix[:,best_arm_index] = self.distribution_matrix[:,best_arm_index]/sum(self.distribution_matrix[:,best_arm_index])
            self.reward += arm_reward
        return self
    
    def sample_algorithm(self):
        if self.al == 'epsilon-greedy':
            self.epsilon_greedy()
        elif self.al == 'ucb':
            self.ucb()
        elif self.al == 'kl-ucb':
            self.kl_ucb()
        elif self.al == 'thompson-sampling':
            self.thompson_sampling()
        elif self.al == 'thompson-sampling-with-hint':
            self.thompson_sampling_with_hint()
        return self
    
    def evaluate_regret(self):
        self.populate_arms()
        self.sample_algorithm()
        tpstar = self.get_best_true_arm().true_mean * self.hz
        self.regret = tpstar - self.reward
        return self