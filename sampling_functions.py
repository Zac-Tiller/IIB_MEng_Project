# Imports

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma as gamma_func
from scipy.stats import invgamma
from scipy.special import kv
from scipy.linalg import expm
import time as t
import copy
from numpy.matlib import repmat
from matplotlib.colors import LogNorm
import pandas as pd

# 16th October Session

# reply to Joe Johnson on email

# We need to simulate the gamma process
'''
We do a rejection sampling approach

'''

np.random.seed(10)


class GammaDistr:
    def __init__(self, alpha, beta, rng=np.random.default_rng(10)):
        self.alpha = alpha
        self.beta = beta
        self.distribution = 'Gamma'
        self.rng = rng

        self.T = None
        self.start_time = None
        self.sample_size = None
        self.rate = None

    def set_process_conditions(self, t0, T, END, sample_size, TEST=False):
        self.start_time = t0
        # t0 = t_i
        # T = t_i+1
        self.T = T
        self.sample_size = sample_size
        self.rate = 1/(T-t0) # set 1.0 to T
        if TEST:
            print('dif rate')
            self.rate=1/(T-t0)
        # self.rate = T / (t_i+1 - t_i)

        gsamps = int(10. / self.beta)
        if gsamps < 50:
            gsamps = 50
        elif gsamps > 10000:
            gsamps = 10000
            print('Warning ---> beta too low for a good approximation')
        self.sample_size = gsamps # override sample size


        # print('Change Line 39 back to 1!!!')


class NormalDistr:
    def __init__(self, mean, std, secondary_distr, rng=np.random.default_rng(10)):
        self.mean = mean
        self.std = std
        self.distribution = 'Normal'
        self.rng = rng

        self.secondary_distr = secondary_distr

    def NormalGammaPDF(self, x):
        gamma_distr = self.secondary_distr

        t1 = 2*np.exp(self.mean*x/self.std**2)

        delta = 2*self.std**2/gamma_distr.beta + self.mean**2
        tau = 1/gamma_distr.beta - 0.5

        t2 = np.power(gamma_distr.beta, 1/gamma_distr.beta)*np.sqrt(2*np.pi*(self.std**2))*gamma_func(1/gamma_distr.beta)
        t3 = (np.abs(x)/(self.std**2)) * np.sqrt(delta)
        t4 = (1/gamma_distr.beta) - 0.5
        t5 = (1./self.std**2) * np.sqrt(self.mean**2 + (2*(self.std**2)/gamma_distr.beta)*np.abs(x))


        return (t1 / t2) * (x**2/delta)**(tau/2) * kv(tau,t3)

debug = False

class DistributionSimulator:

    def __init__(self, DistributionObject): #*OtherObject
        self.distribution = DistributionObject.distribution
        self.distr_obj = DistributionObject
        self.rng = DistributionObject.rng
        # self.secondary_distr = OtherObject

        self.sorted_process_set = None
        self.time_arr = None
        self.process_path = None
        self.NG_jump_series = None
        self.task = None

    def plot_simulation_distribution(self, process_endpoints):

        plt.figure(1)
        fig, ax = plt.subplots()
        xx, bins, p = ax.hist(process_endpoints, bins=175, density=True)

        if self.distribution == 'Gamma':  # this was for future if we plot a diff distrs end. classes now handle this - This is to overlay the PDF
            T = self.distr_obj.T
            x = np.linspace(0, 10, 1000)
            # x = self.

            # TODO: had self.distr_obj as gamma obj before
            shape = (self.distr_obj.alpha ** 2) * T / self.distr_obj.beta
            rate = self.distr_obj.beta / self.distr_obj.alpha
            y = stats.gamma.pdf(x, a=shape, loc=0, scale=rate) #TODO: put loc argument = 0?

        else: # self.distribution == 'Normal':
             x = np.linspace(-10, 10, 5000)
             #x = self.NG_jump_series

             y = self.distr_obj.NormalGammaPDF(x)

            # x, y = 0, 0

        ax.plot(x, y)
        # for item in p:
        #     item.set_height(item.get_height()/sum(xx))
        fig.suptitle('Endpoints for {} Process Simulation'.format(self.distribution if self.distribution != 'Normal' else 'Normal Gamma'))
        fig.show()
        return fig, ax

    def tractable_inverse_gamma_tail(self, alpha, beta, x):
        return 1 / ((alpha / beta) * (np.exp(beta * x / alpha ** 2) - 1))

    def acceptance_probability(self, alpha, beta, x):
        return (1 + alpha * x / beta) * np.exp(-alpha * x / beta)

    def process_simulation(self, *prev_sim_data):
        DistributionObject = self.distr_obj
        rng = DistributionObject.rng
        if self.distribution == 'Gamma':
            # start_time = t.time()
            T = DistributionObject.T
            t0 = DistributionObject.start_time
            sample_size = DistributionObject.sample_size
            alpha = DistributionObject.alpha
            beta = DistributionObject.beta

            exp_rvs = rng.exponential(scale=DistributionObject.rate, size=1_000)[:sample_size]
            poisson_epochs = np.cumsum(exp_rvs)

            # generate_jump_sizes = np.vectorize(self.tractable_inverse_gamma_tail)
            # generate_acceptance_probabilities = np.vectorize(self.acceptance_probability)

            jump_sizes = self.tractable_inverse_gamma_tail(alpha, beta, poisson_epochs) #NOTE - used to be caling the np vectorized functions
            acceptance_probabilities = self.acceptance_probability(alpha, beta, jump_sizes)


            # Now need to use this acceptance probability array to accept or reject the samples
            # samples = [0]

            # rnd = np.random.choice([0,1], size = acceptance_probabilities.shape, p = acceptance_probabilities)

            # rnd_gen = np.random.default_rng()
            unif_r = rng.random(len(acceptance_probabilities))[:jump_sizes.shape[0]]

            # accept samples who's acceptance probs are higher than this randomly generated list of numbers

            samples = np.where(acceptance_probabilities > unif_r, jump_sizes, 0.)
            samples = samples[samples>0.]
            if t0 == 0: # if time range is 0-> 1, we start at initial point (0,0). Else, we do not insert a 0
                samples = [0] + list(samples)
            else:
                samples = list(samples)

            if len(samples) == 0:
                jump_times = []
                # jump_times = [t0]
                # samples = [0]
                print('no samples !!!')
            else:
                if t0 == 0:
                    jump_times = [t0] + list(rng.uniform(t0, T, len(samples)-1)) # jump times are uniform spread, property of exponen distr
                else:
                    jump_times = list(rng.uniform(t0, T, len(samples)-1))

            # if len(samples) != len(jump_times):
            #     print('Length Issues')
            jumps_and_times = zip(samples, jump_times)

            self.sorted_process_set = sorted(jumps_and_times, key=lambda x: x[1])
            self.process_path = np.cumsum([jump_time_set[0] for jump_time_set in self.sorted_process_set])
            self.time_arr = [jump_time_set[1] for jump_time_set in self.sorted_process_set]


            # TODO: Ask about the case when time just starts
            # print('Time Elapsed for a Single Gamma Proces Path = {}'.format(t.time() - start_time))
            return self.process_path, self.time_arr, self.sorted_process_set


        if self.distribution == 'Normal': # we want to simulate a N-G process. Run a gamma first, then put it through a normal
            # DO NORMAL DISTR FUNCTIONS
            mean = DistributionObject.mean
            std = DistributionObject.std

            # Create a GammaDistr Object (with necessary params). Use this to make DistrSim Object. Run the Gamma Sim, then return the sorted process set
            hidden_gamma_distr = DistributionObject.secondary_distr
            hidden_gamma_sim = DistributionSimulator(hidden_gamma_distr)
            hidden_gamma_sim.process_simulation() #hidden_gamma_distr) # call .process_simulation to generate jumps and times
            gamma_process_set = hidden_gamma_sim.sorted_process_set

            #TODO: Plot the hidden sparse gamma simulation
            # if self.task != 'SS_SIM':
            #     plt = plotter(hidden_gamma_sim.time_arr, hidden_gamma_sim.process_path, 'Hidden Gamma Sim (Of NG)', 'Time', 'Value')

            # make the gamma process set the sorted process set of the hidden_gamma_distr

            # process_set = prev_sim_data # this is the sorted process set of the gamma which we just ran


            normal_gamma_jump_series = []
            jump_time = list(zip(*gamma_process_set))
            # for jump_time in gamma_process_set:  # process_set is a set of (jump, time). Has length Ns
            normal_gamma_jump_series.append(rng.normal(loc=mean * np.array(jump_time[0]), scale=(std) * np.sqrt(np.array(jump_time[0]))))

            self.process_path = np.cumsum(normal_gamma_jump_series)
            self.time_arr = [tuple[1] for tuple in gamma_process_set]

            self.NG_jump_series = normal_gamma_jump_series

            jumps_and_times = zip(list(normal_gamma_jump_series[0]), list(self.time_arr))

            self.sorted_process_set = sorted(jumps_and_times, key=lambda x: x[1]) #TODO: return same 3 params for both normal and gamma sims

            return self.process_path, self.time_arr #, self.sorted_process_set


    def process_endpoint_sampler(self, iterations, DistributionObject, **kwargs): #TODO:

        '''
        Issue: So, below there is a class instance called gamma_sim and another called gamma_obj
                Right now these are global variables as I have defined these outside the function
                But, I want to pass these class instances to this function process_endpoint_sampler and so
                I tried to use **kwargs (and *args) but when I do so, I cannot figure out how to assign each of the variables in
                the kwargs to gamma_sim and gamma_obj within this function
        '''

        # This is not right, but for example - if kwargs is specified then I want this to happen:
        # gamma_sim = kwargs[0]
        # gamma_obj = kwargs[1]

        process_endpoints = []

        # start = t.time()
        for i in range(0, iterations):
            # if DistributionObject.distribution == 'Normal':
            # #     gamma_sim.process_simulation(gamma_obj) #we generate a new gamma sim to sample from each iteration
            #prev_sim_data = gamma_sim.process_simulation(gamma_obj).sorted_process_set  # TODO: test this. If not work, try and re-do the whole gamma sim each iteration

            #gamma_sim.process_simulation(gamma_obj) #TODO: Right Now This Shadows Name of Outer Scope - so

            self.process_simulation(DistributionObject) #, gamma_sim.sorted_process_set) # this gamma sim is from outer scope

            process_endpoints.append(self.process_path[-1])
        # print('Elapsed time for Endpoint Sampling: {}'.format(t.time() - start))
        return process_endpoints


class StateSpaceSimulation:
    """A Class Enabling Definition of State Space Model to Forward Simulate a Process With, And
    Provides Functionality to Simulate & Observe Such Process
    """

    # calculate jumps in interval
    # calculate mean vec and cov mat
    # sample for n (stoc int === noise)
    # incriment state vector by X_tb = exp(A(tb - ta) Xta + n

    def __init__(self, DistrSimObject, num_obs, t0, T):
        """Pass DistrSim Object, who's attributre distr_obj is our Distribution ie. here, it will be the Gamm distr"""
        # Have to Pass GammaSim object with a simulation already ran
        self.distr_obj = DistrSimObject.distr_obj
        self.distr_sim_obj = DistrSimObject
        self.rng = self.distr_obj.rng
        DistrSimObject.task = 'SS_SIM'

        self.sorted_obs_times = DistrSimObject.time_arr #takes the initial observation times from the initial distrsimObj, which is a simulation based on our initial gamma object

        self.SS_obs_rate = 1.0/(T - t0)
        self.sorted_obs_times = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)
        print(self.sorted_obs_times)



        self.X = np.zeros((2,1))
        self.t_0 = 0

        self.A = np.zeros((2,2))
        self.h = np.zeros((2))

        self.NG_mean = 0
        self.NG_var = 0

    def set_NG_conditions(self, mean, var):
        self.NG_mean = mean
        self.NG_var = var

        print(self.NG_mean)
        print(self.NG_var)


    def define_A_matrix(self, flatterned_A):
        if len(flatterned_A) == 1:
            self.A = flatterned_A[0]
        else:
            self.A[0][0] = flatterned_A[0]
            self.A[0][1] = flatterned_A[1]
            self.A[1][0] = flatterned_A[2]
            self.A[1][1] = flatterned_A[3]

    def define_model_h(self, flatterned_h):
        self.h[0] = flatterned_h[0]
        self.h[1] = flatterned_h[1]


    def generate_jumps_between_time_intervals(self, t_a, t_b):
        gamma_obj_small_interval = GammaDistr(self.distr_obj.alpha, self.distr_obj.beta) #           nCheck time intervals of gamma obj are done correctly
        #give our gamma_object over this small time interval the alpha and beta of the distr_obj; which is the distribution object behind the gamma_sim passed when creating the first instance of the class
        gamma_obj_small_interval.set_process_conditions(t_a, t_b, self.distr_obj.sample_size)

        ### DistrSimObj = self.distr_sim_obj

        gamma_sim_small_interval = DistributionSimulator(gamma_obj_small_interval)
        path, time, jump_time_set = gamma_sim_small_interval.process_simulation() #gamma_obj_small_interval)


        # DistrSimObj.process_simulation(gamma_obj_small_interval) #, gamma_obj_small_interval) # this overwrites the existing process set for the DistrSimObj w a new set of jumps for new gamma_obj_small_interval
        ### self.distr_sim_obj = DistrSimObj # update the DistrSimulationObject attribute of the state space simulator

        
        #TODO: Change this and following dependencies to self.distr_sim_obj_sub_interval !!!!!!!
        self.distr_sim_obj = gamma_sim_small_interval #Update the distr_sim_obj attribute with the recent run for small interval.

        ### process_path, sorted_times, sorted_set = self.distr_sim_obj.process_simulation(self.distr_obj)

        # each time generate_jumps_btw_time_intervals is called in each iter of run_state_space_sim,
        #we need to re-instate our gamma_objject (over the time interval minT, ta, to maxT tb)
        #which will allow us to THEN call aglgo 2 ie run a gamma process sim over ta to tb
        #we then return these jumps
        return jump_time_set


    def calculate_jumps_mean_and_cov(self, start_time, end_time, Mat):
        """Calculate the mean vector given a collection of gamma jumps, gamma jump times, and a time interval"""

        #sum over time interval, matrix exponential (end time - jump time) x associated_gamma_jump
        mean_vec = np.array([ [float(0)],
                              [float(0)] ])
        cov_mat = np.array([ [float(0),float(0)],
                             [float(0),float(0)] ])

        print('large gamma jumps on this sub-interval: {}'.format([tuple[0] for tuple in self.distr_sim_obj.sorted_process_set if tuple[0] > 0.001]))
        # print([tuple[0] for tuple in self.distr_sim_obj.sorted_process_set if tuple[0] > 0.001])

        for jumpsize, jumptime in self.distr_sim_obj.sorted_process_set: # we get the sorted process set for the gamma sim on small interval
            ft_Vi = self.calc_matrix_exponent(end_time, jumptime, Mat)

            # lavengin_ft_Vi = self.calc_lavengin_specific_matrices(end_time, jumptime)
            # ft_Vi = lavengin_ft_Vi

            # self.check_ft_Vi_equality(ft_Vi, lavengin_ft_Vi)

            # print('ft_Vi:')
            # print(ft_Vi)

            mean_vec += (ft_Vi * jumpsize)
            cov_mat += (ft_Vi * ft_Vi.T * jumpsize)
        return mean_vec, cov_mat

    def check_ft_Vi_equality(self, ft_Vi_numpy, ft_Vi_analytical):
        residual = ft_Vi_numpy - ft_Vi_analytical
        if np.abs(np.sum(residual)) > 0.00001:
            print('error with matrix calculation')


    def sample_stoc_int(self, mean_vec, cov_mat, time):

        # time_varying_skew!
        # if 0 <= time < 5:
        #     self.NG_mean = 5
        # else:
        #     self.NG_mean = 0

        m = mean_vec * self.NG_mean
        cov = cov_mat * self.NG_var
        print('cov = {}'.format(cov))
        # cholesky decomposition for sampling of noise
        try:
            cov_chol = np.linalg.cholesky(cov)
            n = cov_chol @ np.column_stack([self.rng.normal(size=2)]) + np.column_stack([m])
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            n = np.zeros(2)
        # print(np.shape(n))
        return n

    def update_state_vec(self, n, start_time, end_time, Mat):
        A = self.A
        if Mat:
            multiplier = expm(A * (end_time - start_time))
        else:
            multiplier = np.exp(A * (end_time - start_time))
        print('n : {}. start = {}, end = {}'.format(n, start_time, end_time))
        print()

        self.X = multiplier*self.X + n # TODO: Ask


    def calc_matrix_exponent(self, t, jump_time, Mat): # This is the ft(Vi) CALCULATION - returns the ft(Vi)
        """Calculate the matrix exponent given an A matrix, time interval end, and times within such time interval"""
        # we need to calc the matrix exponent for every summation in our time interval, as we have
        #exp(A(t - Vi)) where Vi are the jump times (of the gamma sim), and t is the END TIME of our interval.
        #-> this finds later use in our M
        # print('A:')
        # print(self.A)
        #
        # print('h:')
        # print(self.h)

        A = self.A
        h = self.h


        return expm(A*(t - jump_time))@np.reshape(np.array(h), (2,1)) if Mat else np.reshape(np.array(h), (2,1))*np.exp(A*(t-jump_time)) # TODO: have ft_VI something else if we do not use A matrix


    def calc_lavengin_specific_matrices(self, end_time, intermediate_time):
        theta = self.A[1][1]
        lavengin_ft_Vi = np.array( [ [(np.exp(theta*(end_time - intermediate_time))-1)/theta] , [np.exp(theta*(end_time - intermediate_time))] ])
        return lavengin_ft_Vi

    def update_gamma_alpha(self, start, end):
        t = end - start
        self.distr_obj.alpha = t


    def run_SSS_method_two(self, end_time, Mat):

        x0_evolution = [0]
        x1_evolution = [0]

        jump_time_set = self.generate_jumps_between_time_intervals(0, end_time)

        mean = self.NG_mean
        std = self.NG_var

        for i in range(0, len(jump_time_set)-1):
            interval_start = jump_time_set[i][1]
            interval_end = jump_time_set[i+1][1]

            current_jump = jump_time_set[i][0]

            # for n we just sample the weighted norm distr
            n = self.rng.normal(loc=mean * np.array(current_jump), scale=(std) * np.sqrt(np.array(current_jump)))
            self.update_state_vec(n, interval_start, interval_end, Mat)

            x0_evolution.append(self.X[0][0])
            x1_evolution.append(self.X[1][0])

        fig1, ax1 = plt.subplots()
        ax1.scatter(self.distr_sim_obj.time_arr, x0_evolution, color='r', s=4, zorder=2)
        ax1.plot(self.distr_sim_obj.time_arr, x0_evolution, zorder=1, linestyle='--')
        fig1.suptitle(
            'X0 - Skew = {}, Var = {}, Beta = {}, A = {}. Method 2'.format(self.NG_mean, self.NG_var, self.distr_obj.beta,
                                                                 self.A))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.scatter(self.distr_sim_obj.time_arr, x1_evolution, color='r', s=4, zorder=2)
        ax2.plot(self.distr_sim_obj.time_arr, x1_evolution, zorder=1, linestyle='--')
        fig2.suptitle(
            'X1 - Skew = {}, Var = {}, Beta = {}, A = {}. Method 2'.format(self.NG_mean, self.NG_var, self.distr_obj.beta,
                                                                 self.A))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        gamma_path = np.cumsum([jump_time[0] for jump_time in jump_time_set])

        fig3, ax3 = plt.subplots()
        # ax3.scatter(self.distr_sim_obj.time_arr, x1_evolution, color='r', s=4, zorder=2)
        ax3.step(self.distr_sim_obj.time_arr, gamma_path, zorder=1)
        fig3.suptitle(
            'Latent Gamma Path M2. Beta = {}'.format(self.distr_obj.beta))
        plt.xlabel('Time')
        plt.ylabel('Latent gamma jumps')
        plt.show()





    def run_state_space_simulation(self, Mat):
        # debug = True
        x0_evolution = [0]
        x1_evolution = [0]

        latent_gamma_jump_in_time_interval = [0]

        latent_normal_gamma_jump_in_time_interval = [0]

        gamma_jumps = []
        gamma_times = []

        print('len self.sorted_obs_times = {}'.format(len(self.sorted_obs_times)))
        for i in range(0,len(self.sorted_obs_times)-1):
            start_time = self.sorted_obs_times[i]
            end_time = self.sorted_obs_times[i+1]

            # self.update_gamma_alpha(start_time, end_time)

            gamma_jump_time_set = self.generate_jumps_between_time_intervals(start_time, end_time)
            mean_vec, cov_mat = self.calculate_jumps_mean_and_cov(start_time, end_time, Mat)
            n = self.sample_stoc_int(mean_vec, cov_mat, i)
            self.update_state_vec(n, start_time, end_time, Mat)

            x0_evolution.append(self.X[0][0])
            x1_evolution.append(self.X[1][0])

            for jump_time in gamma_jump_time_set:
                gamma_jumps.append(jump_time[0])
                gamma_times.append(jump_time[1])

            gamma_dWt = np.sum(gamma_jumps)
            latent_gamma_jump_in_time_interval.append(gamma_dWt)

            # latent_normal_gamma_jump_in_time_interval.append(n)

        latent_gamma_path = np.cumsum(np.array(latent_gamma_jump_in_time_interval))
        # latent_normal_gamma_path = np.cumsum(np.array(latent_normal_gamma_jump_in_time_interval))

        print('Points observed in state space model: {}'.format(len(x1_evolution)))

        fig1, ax1 = plt.subplots()
        ax1.scatter(self.sorted_obs_times, x0_evolution, color='r', s=4, zorder=2)
        ax1.plot(self.sorted_obs_times, x0_evolution, zorder=1, linestyle='--')
        fig1.suptitle('X0 - Skew = {}, Var = {}, Beta = {}, A = {}. Method 1'.format(self.NG_mean, self.NG_var, self.distr_obj.beta, self.A))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.scatter(self.sorted_obs_times, x1_evolution, color='r', s=4, zorder=2)
        ax2.plot(self.sorted_obs_times, x1_evolution, zorder=1, linestyle='--')
        fig2.suptitle('X1 - Skew = {}, Var = {}, Beta = {}, A = {}. Method 1'.format(self.NG_mean, self.NG_var, self.distr_obj.beta, self.A))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig3, ax3 = plt.subplots()
        # ax3.scatter(self.sorted_obs_times, x1_evolution, color='r', s=4, zorder=2)
        ax3.step(self.sorted_obs_times, latent_gamma_path, zorder=1) #, linestyle='--')
        fig3.suptitle('Latent Gamma Process')
        plt.xlabel('Time')
        plt.ylabel('Gamma Process Path')
        plt.show()

        # fig4, ax4 = plt.subplots()
        # # ax3.scatter(self.sorted_obs_times, x1_evolution, color='r', s=4, zorder=2)
        # ax4.plot(self.sorted_obs_times, latent_normal_gamma_path, zorder=1)  # , linestyle='--')
        # fig4.suptitle('Latent Normal Gamma Process')
        # plt.xlabel('Time')
        # plt.ylabel('Normal-Gamma Process Path')
        # plt.show()


        # plt.plot(self.sorted_obs_times, x0_evolution)
        # plt.title('Evolution of state vector x0')
        # plt.show()

        return fig1, fig2


class StateSpaceSimulator2:

    def __init__(self, DistrSimObject, t0, T, num_obs, MatA):
        self.distr_obj = DistrSimObject.distr_obj
        self.distr_sim_obj = DistrSimObject
        self.rng = self.distr_obj.rng

        # self.sorted_obs_times = DistrSimObject.time_arr  # takes the initial observation times from the initial distrsimObj, which is a simulation based on our initial gamma object

        self.SS_obs_rate = 1.0 / (T - t0)
        self.sorted_obs_times = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)

        self.epochs = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)

        self.random_observation_times = np.cumsum(self.rng.exponential(scale=1/10, size=num_obs))

        self.MatA = MatA
        self.t_0 = 0

        self.forward_sim_latent_jt_set = None

        # predefining matrix structure
        if MatA:
            self.A = np.zeros((2, 2))
            self.X = np.zeros((2, 1))
            self.h = np.zeros((2, 1))
        else:
            self.A = np.array([0])
            self.X = np.array([0])
            self.h = np.array([1])

    def define_A(self, flatterned_A):
        if len(flatterned_A) == 1:
            self.A = np.array([[flatterned_A[0]]])
        else:
            self.A[0][0] = flatterned_A[0]
            self.A[0][1] = flatterned_A[1]
            self.A[1][0] = flatterned_A[2]
            self.A[1][1] = flatterned_A[3]
            
    def define_h(self, flatterned_h):
        if len(flatterned_h) == 1:
            self.h[0] = flatterned_h[0]
        else:
            self.h[0][0] = flatterned_h[0]
            self.h[1][0] = flatterned_h[1]

    def record_state_evolution(self, state_evo_vec, new_state, **kwargs):
        
        if self.MatA:
            state_evo_vec[0].append(new_state[0][0])
            state_evo_vec[1].append(new_state[1][0])

            if 'skew' in kwargs:
                skew = kwargs['skew']

            # if len(state_evo_vec) == 3:
                state_evo_vec[2].append(skew)
            else:
                state_evo_vec[2].append(new_state[2][0])

        else:
            state_evo_vec.append(new_state)
      
        return state_evo_vec


    def calculate_ftVi(self, jump_time, end_time):
        ftVi = expm(self.A * (end_time - jump_time[1])) @ self.h
        return ftVi

    def calculate_jumps_raw_mean_and_cov(self, jump_time_set, start_time, end_time):
        """sum ( ftVi * mu * dGamma_i)
        ftVi = exp(A*(t - Vi)) h    where Vi = time of jump, t = end time """

        mean = np.zeros((2,1))
        cov = np.zeros((2,2))
        for jump_time in jump_time_set:
            jump = jump_time[0]
            time = jump_time[1]

            if start_time < time <= end_time:

                ftVi = self.calculate_ftVi(jump_time, end_time)

                mean += ftVi * jump
                cov += ftVi @ ftVi.T * jump
        return mean, cov

    def sample_gaussian(self, mean, cov):
        try:
            cov_chol = np.linalg.cholesky(cov)
            n = cov_chol @ np.column_stack([self.rng.normal(size=2)]) + np.column_stack([mean])
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            n = np.zeros((2,1))
        return n

    def forward_simulate_step_by_step_gauss(self, skew, beta, var, state_dim):
        dyn_skew = False

        # initialise the skew
        if skew == 'dynamic':
            dyn_skew = True
            # skew = self.rng.normal(loc=0, scale=1)
            skew = 0
            skew_var = 0.5


        if self.MatA and state_dim != 3:
            x_evolution = [[0], [0]]
        elif self.MatA and state_dim == 3:
            x_evolution = [[0], [0], [skew if skew != 'dynamic' else 1]] #CONSTANT SKEW
            print('CONSTANT SKEW (NOTE FOR FUTURE - VARY W TIME)')
        else:
            x_evolution = [0]

        obs_times = self.sorted_obs_times
        # obs_times = np.cumsum(self.rng.exponential(scale=1/10, size=num_obs)) # observation times is a random set of exponenial arrivals

        obs_times = self.random_observation_times

        ng_jumps = [0]
        latent_gamma_path = [0]
        latent_gamma_jump_time_set = []

        for i in range(len(obs_times) - 1):

            start_time = obs_times[i]
            end_time = obs_times[i + 1]

            if dyn_skew:
                print('ATTENTION: now this is fixing skew to be 1 and then drop to 0. change to be BM later - uncomment above code')
                # if start_time < 5.0:
                #     skew = 1
                # else:
                #     skew = 0
                print('ATTENTION 2: Find value of skew_var to pick')
                skew = skew + np.sqrt(skew_var * (end_time - start_time)) * self.rng.normal(loc=0, scale=1)

            step_gamma_obj = GammaDistr(alpha=1, beta=beta)
            step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None,
                                                    sample_size=450)
            step_gamma_sim = DistributionSimulator(step_gamma_obj)

            step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
            # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])

            # latent_gamma_path.append(step_gamma_path[0])

            #TODO: CHECK I AM APPENDING AND PLOTTING THE RIGHT 'STEP GAMMA PATH' AND ALSO CREATE A LIST OF THE JUMP_TIME_SETS

            mean, cov = self.calculate_jumps_raw_mean_and_cov(step_gamma_jump_time_set, start_time, end_time)


            print('start = {}, skew = {}'.format(start_time, skew))
            n = self.sample_gaussian(skew*mean, var*cov) # n is essentially our 'normal gamma jump'... therefore as a check, lets store teh jumps and cumsum them
            ng_jumps.append(n[1][0])



            # generate gamma process on this interval. store the jumps as a variable
            # calculate mean and cov of these jumps
            # sample n from our weighted gaussian
            # update state vector !

            # x_evolution.append(self.update_state_vec(n, start_time, end_time))
            new_state = self.update_state_vec(n, start_time, end_time)
            x_evolution = self.record_state_evolution(x_evolution, new_state, skew=skew)

        ng_path = np.cumsum(ng_jumps)

        # fig, ax = plt.subplots()
        # ax.step(obs_times, latent_gamma_path, zorder=1)
        # fig.suptitle('Latent Gamma - alpha = {}, beta = {}'.format(self.distr_obj.secondary_distr.alpha,
        #                                                                                                     self.distr_obj.secondary_distr.beta))
        # plt.xlabel('Time')
        # plt.ylabel('Gamma Path')
        # plt.show()

        # self.forward_sim_latent_jt_set = latent_gamma_jump_time_set


        return obs_times, x_evolution, ng_path

        # fig1, ax1 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax1.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--')
        # fig1.suptitle('X0 Evolution - step-by-step. A = {}, alpha = {}, beta = {}'.format(self.A,
        #                                                                                                     self.distr_obj.secondary_distr.alpha,
        #                                                                                                     self.distr_obj.secondary_distr.beta))
        # plt.xlabel('Time')
        # plt.ylabel('X0')
        # plt.show()
        #
        # fig2, ax2 = plt.subplots()
        # ax2.scatter(obs_times, x_evolution[1][:], color='r', s=4, zorder=2)
        # ax2.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--')
        # fig2.suptitle('X1 Evolution - step-by-step. A = {}, alpha = {}, beta = {}'.format(self.A,
        #                                                                                     self.distr_obj.secondary_distr.alpha,
        #                                                                                     self.distr_obj.secondary_distr.beta))
        # plt.xlabel('Time')
        # plt.ylabel('X1')
        # plt.show()
        #
        # fig3, ax3 = plt.subplots()
        # ax3.step(obs_times, ng_path)
        # plt.xlabel('time')
        # plt.ylabel('NG Path (sampled n) series')
        # fig3.suptitle('NG Path (n1) for our n sampled from gaussian w')
        # plt.show()



    def show_plots(self, obs_times, x_evolution, ng_path):
        fig1, ax1 = plt.subplots()
        ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        ax1.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--')
        fig1.suptitle('X0 Evolution - step-by-step. A = {}, alpha = {}, beta = {}'.format(self.A,
                                                                                          self.distr_obj.secondary_distr.alpha,
                                                                                          self.distr_obj.secondary_distr.beta))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.scatter(obs_times, x_evolution[1][:], color='r', s=4, zorder=2)
        ax2.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--')
        fig2.suptitle('X1 Evolution - step-by-step. A = {}, alpha = {}, beta = {}'.format(self.A,
                                                                                          self.distr_obj.secondary_distr.alpha,
                                                                                          self.distr_obj.secondary_distr.beta))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig3, ax3 = plt.subplots()
        ax3.scatter(obs_times, x_evolution[2][:], color='r', s=4, zorder=2)
        ax3.plot(obs_times, x_evolution[2][:], zorder=1, linestyle='--')
        fig3.suptitle('X2 Evolution (JUST ON A RUN OF A SSS - MEAN IS KNOWN/SELECTED BEFORE-HAND). A = {}, alpha = {}, beta = {}'.format(self.A,
                                                                                          self.distr_obj.secondary_distr.alpha,
                                                                                          self.distr_obj.secondary_distr.beta))
        plt.xlabel('Time')
        plt.ylabel('X2 (SKEW)')
        plt.show()

        # fig3, ax3 = plt.subplots()
        # ax3.step(obs_times, ng_path)
        # plt.xlabel('time')
        # plt.ylabel('NG Path (sampled n) series')
        # fig3.suptitle('NG Path (n1) for our n sampled from gaussian w')
        # plt.show()



    def forward_simulate(self):
        # we have the jump sizes and the jump times
        # now we need to cycle through the observation times, and if there is a jump within this interval,
        # we apply the state space simulation weighting thing
        obs_times = self.sorted_obs_times
        ng_jump_time_set = self.distr_sim_obj.sorted_process_set
        if self.MatA:
            x_evolution = [[0], [0]]
        else:
            x_evolution = [0]

        for i in range(len(obs_times)-1):
            start_time = obs_times[i]
            end_time = obs_times[i+1]
            n=0
            for j in range(len(ng_jump_time_set)):

                if start_time < ng_jump_time_set[j][1] <= end_time:
                    print('jump encountered: {}'.format(j))
                    # while start_time < ng_jump_time_set[j][0] <= end_time:
                    #     j+=1
                    n += self.calculate_n(ng_jump_time_set[j], end_time)
                if ng_jump_time_set[j][1] > end_time:
                    break
            x_evolution.append(self.update_state_vec(n, start_time, end_time))
            new_state = self.update_state_vec(n, start_time, end_time)
            x_evolution = self.record_state_evolution(x_evolution, new_state)

        fig1, ax1 = plt.subplots()
        ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        ax1.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--')
        fig1.suptitle('X0 Evolution - NG Jumps Passed Via Scalar SDE. A = {}, alpha = {}, beta = {}'.format(self.A,
                                                                                                            self.distr_obj.secondary_distr.alpha,
                                                                                                            self.distr_obj.secondary_distr.beta))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.scatter(obs_times, x_evolution[1][:], color='r', s=4, zorder=2)
        ax2.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--')
        fig2.suptitle('X1 Evolution - NG Jumps Passed Via Scalar SDE. A = {}, alpha = {}, beta = {}'.format(self.A,
                                                                                                            self.distr_obj.secondary_distr.alpha,
                                                                                                            self.distr_obj.secondary_distr.beta))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        return x_evolution


    def calculate_n(self, jump_time_set, end_time):
        if not self.MatA:
            n = np.exp(self.A[0] * (end_time - jump_time_set[1])) * self.h * jump_time_set[0]
        # n = expm(self.A*(end_time - jump_time_set[1])) * self.h * jump_time_set[0]
        else:
            n = expm(self.A*(end_time - jump_time_set[1])) @ self.h * jump_time_set[0]
        return n


    def update_state_vec(self, n, start_time, end_time):
        A = self.A
        if self.MatA:
            multiplier = expm(A * (end_time - start_time))
        else:
            n = n[0]
            multiplier = np.exp(A * (end_time - start_time))
        # print('n : {}. start = {}, end = {}'.format(n, start_time, end_time))
        # print()

        self.X = multiplier @ self.X + n  # TODO: Ask
        return self.X[0][0] if not self.MatA else self.X









class FilterProcess:
    def __init__(self, StateSimObject, kv, k_mu_BM):
        # super().__init__(DistrSimObject, t0, T, num_obs, MatA)

        # self.latent_jump_time_set = StateSpaceSimulator2.forward_sim_latent_jt_set
        self.StateSimObject = StateSimObject # Pass the StateSimObject JUST to have access to its methods, such as, generating a gamma process over some interval, and method to calc jumps mean and cov
        self.obs_times = StateSimObject.sorted_obs_times
        self.obs_times = StateSimObject.random_observation_times

        self.X = np.zeros((3,1))
        self.X = np.array([[131], [0], [0]])
        self.langevin_A = StateSimObject.A
        self.kv = kv
        self.k_mu_BM = k_mu_BM
        self.var = 1 #TODO: inheritance!
        self.sigma_mu_sq = self.var # self.k_mu_BM * self.var
        print('TODO: Inherit this \'var\' from prev. process !!!!')
        print()
        print('The var is init to 1 due to us running the MPF case, so set var to 1 for now')

        self.dynamic_skew_var = self.k_mu_BM

        # Parameters of Kalman / PF Filters for Fixed Skew
        self.caligA = np.zeros((3,3))
        self.caligB = np.array([[1, 0], [0, 1], [0, 0]])
        self.caligH = np.array([[1, 0, 0]])
        
        # Parameters of Kalman / PF Filters for Dynamic Skew - with 2 part Kalman Prediction Steps
        self.F1 = None
        self.F2 = None

        self.ei_cov = None
        self.eii_cov = None

        # Parameters for Dynamic Skew one-time-step-update
        self.D = None
        self.Lambda = None
        self.P = None
        self.M = None
        self.e_state_cov = None # e_state ~ N(0, S) where S = sigma_w^2 * Ce_state , Ce_state = C1 + C2 etc...

        # initialise kalman mean and covariance vec and matrix
        self.kalman_mean = np.zeros((3,1)) #TODO: Change to more informative prior? run a few processes first (good to include in report...)!!!
        self.kalman_cov = 0.8*np.eye(3)
        self.kalman_gain = np.zeros((3,1))

        # storing the noisy obs and true sate underlying evolutions of a process run (generated by calling the Kalman filter)
        self.noisy_obs = None
        self.x_evolution = None


        # particle filter initial configurations - initially, when we make an object of the class FilterProcess, we have No state paths
        self.pf = self.PFSamplingUtils(number_particles=10, data_set=None, true_state_paths=None,
                                       obs_times=self.obs_times, calig_params = [self.caligH, self.caligA, self.caligB, self.kv],
                                       inverse_gauss_params = [1e-05, 1e-05],
                                       rng = self.StateSimObject.rng)

                                                           # data_set=self.noisy_obs,
                                                           # true_state_paths=self.x_evolution, obs_times = self.obs_times) #defining the FilterProcess Class to have a nested class instance - call all the 'helper' functions from this object




    class PFSamplingUtils:

        # ~ 7 mins for 100 particles

        def __init__(self, number_particles, data_set, true_state_paths, obs_times, calig_params, inverse_gauss_params, rng):
            self.Np = number_particles
            self.data_set_y_noisy_obs = data_set
            self.true_x_evo = true_state_paths
            self.obs_times = obs_times
            self.rng = rng

            self.state_evo_dict = None

            self.caligH, self.caligA, self.caligB, self.kv = calig_params
            self.rho, self.eta = inverse_gauss_params

            self.epsilon = 0.5

            self.particle_set = None

        def update_PF_attributes(self, data_set, true_state_paths): # This is called once we run the Kalman Filter once as we collect the noisy obs of that path realisation (ONCE WE HAVE REAL DATA-SET this wont be caled as we feed in data_set to the argument of the init function !)
            self.data_set_y_noisy_obs = data_set
            self.true_x_evo = true_state_paths

        def log_sumexp_util(self, lw, h, x, axis=0, retlog=False):

            "Implementing Eqn 4.1.14"
            
            c = np.max(lw)
            broad_l = np.broadcast_to((lw - c).flatten(), x.T.shape).T

            if retlog:
                return c + np.log(np.sum(np.exp(broad_l) * h(x), axis=axis))
            else:
                return np.exp(c) * np.sum(np.exp(broad_l) * h(x), axis=axis)

        def initialise_particle_state_and_ktx(self, kv, i):

            # a_prior = np.array([[131], [0], [0]])
            a_prior = np.array([[0], [0], [0]])
            C_prior = 1 * np.eye(3)
            C_prior_chol = np.linalg.cholesky(C_prior)
            
            sampled_state = a_prior + (C_prior_chol @ self.rng.standard_normal(3).reshape(-1,1))
            self.particle_set[i]['X'].append(sampled_state)
            
            # initial_covariance = np.zeros((3,3))
            initial_covariance = C_prior # test - marginalising
            self.particle_set[i]['kt_x'].append(self.caligH @ initial_covariance @ self.caligH.T + kv)

            pass

        def normalise_weights(self):

            lweights = np.array([self.particle_set[particle_num]['log-weight'] for particle_num in
                                 list(self.particle_set.keys())]).flatten()

            sumweights = self.log_sumexp_util(lweights, lambda x: 1, np.ones(lweights.shape[0]), retlog=True)

            for particle_num in list(self.particle_set.keys()):
                self.particle_set[particle_num]['log-weight'] = self.particle_set[particle_num]['log-weight'] - sumweights

            return None

        def calculate_ESS(self, method):
            # lweights = an array of all the particles log weights
            lweights = np.array([self.particle_set[particle_num]['log-weight'] for particle_num in
                                 list(self.particle_set.keys())]).flatten()

            # these 2 method all are in the log weight domain
            if method == 'd_inf':
                log_ESS = -np.max(lweights)
            elif method == 'p2':
                log_ESS = -1 * self.log_sumexp_util(lw=2*lweights, h=lambda x: 1., x=np.ones(lweights.shape[0]), retlog=True)
            else:
                raise ValueError('Invalid ESS Method Provided: Please provide \'d_inf\' or \'p2\' in calculate_ESS(.)')
            
            return log_ESS


        def update_particle_K_attributes(self, num, attr_type, gain, mean, cov):
            if attr_type == 'predictive':
                key = 'K_predictive_density'

            else:
                key = 'K_correction_density'

            self.particle_set[num][key]['mean'] = mean
            self.particle_set[num][key]['cov'] = cov
            self.particle_set[num]['K_gain'] = gain


        def predict_y(self):
            y_pred = self.caligH @ self.particle[num]['K_predictive_density']['mean']

            return y_pred

        def compute_state_posterior(self):

            lweights = np.array([self.particle_set[particle_num]['log-weight'] for particle_num in
                                 list(self.particle_set.keys())]).flatten()

            means = np.array([self.particle_set[particle_num]['K_correction_density']['mean']
                              for particle_num in list(self.particle_set.keys())])

            msum = self.log_sumexp_util(lweights, lambda x: x, means, axis=0, retlog=False)

            cov_term = np.array([self.particle_set[particle_num]['K_correction_density']['cov']
                                 + (self.particle_set[particle_num]['K_correction_density']['mean'] @
                                    self.particle_set[particle_num]['K_correction_density']['mean'].T)
                                 for particle_num in list(self.particle_set.keys())])

            csum = self.log_sumexp_util(lweights, lambda x: x, cov_term, axis=0, retlog=False)

            a_mix_t = msum
            c_mix_t = csum - msum @ msum.T

            return a_mix_t, c_mix_t

        def resample_particles(self):

            pass


    def pf_predict_y(self, particle):
        y_pred = self.caligH @ particle['K_predictive_density']['mean']
        return y_pred

    def pf_calc_ktx(self, particle): # my kt_x is JJs Cyt
        # print('Calculating kt_x....')
        ktx = self.caligH @ particle['K_predictive_density']['cov'] @ self.caligH.T + self.kv
        return ktx
    
    def pf_calc_Et(self, particle, y):
        Et = particle['E'][-1] + (y - particle['y_pred'][-1])**2 / particle['kt_x'][-1]
        return Et
        
    def pf_calc_log_weight(self, particle):
        ktx = particle['kt_x'][-1]
        Et = particle['E'][-1]
        Et_prev = particle['E'][-2]

        rho = self.pf.rho
        eta = self.pf.eta

        # particle['count'] += 1

        count = particle['count']

        lw = ((-0.5*np.log(ktx)) - (rho + (count/2.)))*np.log(eta + Et/2.) + (rho + ((count-1)/2.))*np.log(eta + Et_prev/2.)
        return lw


    def pf_multinomial_resample(self):

        print('RE-SAMPLING!')

        lweights = np.array([self.pf.particle_set[particle_num]['log-weight'] for particle_num in list(self.pf.particle_set.keys())]).flatten()

        weights = np.exp(lweights)

        probabilities = np.nan_to_num(weights)
        probabilities = probabilities / np.sum(probabilities)

        # numpy multivariate; arguemts = number trials (N) and pvals - probabilities of each of the p different outcomes
        # the output is a vector of the drawn samples, where the value X_i = [X_0, X_1, ..., X_p] represent the number of times the outcome was i
        selections = np.random.multinomial(self.pf.Np, probabilities)
        new_particles = {}
        
        k = 1
        j = 1
        # k = particle number (starts at 1), j = NEW particle number (starts at 1) & is the key of the dictionary
        for selection in selections:
            if selection != 0:
                for _ in range(selection):
                    # new_particle = {}
                    # for key, value in self.pf.particle_set[k].items():
                    #     new_particle[key] = value
                    # new_particle['log-weight'] = -np.log(self.pf.Np)
                    # new_particles[j] = new_particle

                    new_particle = copy.deepcopy(self.pf.particle_set[k])
                    new_particle['log-weight'] = -np.log(self.pf.Np)
                    new_particles[j] = new_particle

                    # OLD CODE:
                    # new_particles[j] = self.pf.particle_set[k]
                    # new_particles[j]['log-weight'] = -np.log(self.pf.Np)  # reset each weight !!!!
                    j+=1
            k+=1



        self.pf.particle_set = new_particles



    def compute_caligA(self, start_time, end_time, mean_vec):
        A = self.langevin_A
        self.caligA = np.block([[expm(A*(end_time - start_time)),   mean_vec],
                                [np.zeros((1, 2)),                      1.]])


    def compute_caligB(self):
        self.caligB = np.array([[1, 0], [0, 1], [0, 0]])

    def hist_record_states_kalman_mean(self, kalman_state_dict, particle_num, kalman_mean):
        kalman_state_dict[particle_num].append(kalman_mean)
        return kalman_state_dict
        

    def compute_noise_vector(self, var, cov_mat):
        # random noise vector is a sample from a 2D gaussian: N(0, sigma^2 S^~) where S^~ is teh covariance of teh jumps
        # var is the variance of the process 
        try:
            cov_chol = np.linalg.cholesky(var * cov_mat)
            e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=2)]) + np.zeros((2,1)) # used to have + mean here - but now zeros
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            e = np.zeros((2,1))
        return e


    def kalman_predictive_mean(self, t_start, t_end, jumps_mean):
        # update the kalman mean:
        self.kalman_mean = self.caligA @ self.kalman_mean
        return None


    def kalman_predictive_cov(self, var, jumps_cov):
        var = 1 # marginalising
        self.kalman_cov = self.caligA @ self.kalman_cov @ self.caligA.T + var * self.caligB @ jumps_cov @ self.caligB.T
        return None


    def kalman_predict(self, t_start, t_end, var, jumps_mean, jumps_cov):

        self.kalman_predictive_mean(t_start, t_end, jumps_mean) # updates the kalman_mean attribute

        self.kalman_predictive_cov(var, jumps_cov) # updates the kalman_cov attribute

        return None


    def compute_kalman_gain(self, var, kv):
        var = 1 # marginalising
        scaling = self.caligH @ self.kalman_cov @ self.caligH.T + var * kv
        self.kalman_gain = (self.kalman_cov @ self.caligH.T) * (scaling**(-1))


    def kalman_update_mean(self, y):
        self.kalman_mean = self.kalman_mean + self.kalman_gain @ (y - self.caligH@self.kalman_mean)
        return None


    def kalman_update_cov(self):
        self.kalman_cov = self.kalman_cov - self.kalman_gain @ self.caligH @ self.kalman_cov
        return None


    def kalman_update(self, y, var, kv, t_start, t_end):

        self.compute_kalman_gain(var, kv) # uses the kalman PREDICTED cov and mean to update the kalman gain parameter
        self.kalman_update_mean(y) # uses the kalman gain and PREDICTED mean and cov to UPDATE the kalman_mean attribute
        self.kalman_update_cov() # uses the old kalman cov and gain to UPDATE the kalman_cov attribute

        return None

    def sigma_posterior(self, x, count, E):
        rhod = self.pf.rho + (count / 2.)
        etad = self.pf.eta + (E / 2.)
        return -(rhod + 1) * np.log(x) - np.divide(etad, x)


    def update_state_vec_3d(self, e, t_start, t_end, jumps_mean):
        self.compute_caligA(t_start, t_end, jumps_mean)
        # we compute caligA for each fresh set of jumps; do NOT call compute caligA in the Kalman Functions; else we re-compute for no reason
        self.X = self.caligA @ self.X + self.caligB @ e
        return self.X


    def observe_state(self):
        # Kv scales the noise relative to the variance of the process !
        std = np.sqrt(self.var * self.kv)
        obs_noise = self.StateSimObject.rng.normal(loc=0, scale=std)
        obs = self.caligH @ self.X + obs_noise
        return obs[0][0]


    def calculate_F1(self, t2, t1):
        self.F1 = np.block([[expm(self.langevin_A * (t2 - t1)), np.zeros((2,1))],
                  [np.zeros((1, 2)), 1.]])
        
    def calculate_F2(self, mean):
        self.F2 = np.block([[np.eye(2), mean],
                  [np.zeros((1, 2)), 1.]])

    def compute_e_i_cov(self, var, t2, t1):
        self.ei_cov = np.array([[0, 0, 0], [0, 0, 0], [0, 0, var*(t2-t1)]])
    
    
    def compute_e_ii_cov(self, var, cov):
        self.eii_cov = np.block( [[cov, np.zeros((2,1))], [0, 0, 0]])
        
        
    def calculate_e_i(self, t2, t1):

        dt = t2 - t1
        e_i = np.sqrt(dt * self.dynamic_skew_var)*self.StateSimObject.rng.normal(loc=0, scale=1)

        B_TSP_1 = np.zeros((3,1))
        e = B_TSP_1 * e_i

        return e
    
    def calculate_e_ii(self, var, cov_mat):
        
        
        self.compute_e_ii_cov(var, cov_mat)
        
        try:
            cov_chol = np.linalg.cholesky(var * cov_mat)
            e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=2)]) + np.zeros((2,1)) # used to have + mean here - but now zeros
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            e = np.zeros((2,1))
        
        # extend e so that it is right dimension
        e = np.reshape(np.append(e, 0), (3,1))
        
        return e
  


    def update_state_vec_TSP_i(self, e): # TSP = two step procedure
        self.X = self.F1 @ self.X + e
        return self.X

    def update_state_vec_TSP_ii(self, e):
        self.X = self.F2 @ self.X + e
        return self.X
    
    def TSP_i_kalman_predictive_mean(self):
        self.kalman_mean = self.F1 @ self.kalman_mean
    
    def TSP_i_kalman_predictive_cov(self):
        self.kalman_cov = self.F1 @ self.kalman_cov @ self.F1.T + self.ei_cov


    def TSP_ii_kalman_predictive_mean(self):
        self.kalman_mean = self.F2 @ self.kalman_mean

    def TSP_ii_kalman_predictive_cov(self):
        self.kalman_cov = self.F2 @ self.kalman_cov @ self.F2.T + self.eii_cov


    def TSP_i_kalman_predict(self):
        # calculate (kalman) mean (1 / 2 - updated mu)
        self.TSP_i_kalman_predictive_mean()
        # calculate (kalman) cov (1 / 2 - updated mu)
        self.TSP_i_kalman_predictive_cov()
        
    def TSP_ii_kalman_predict(self):
        # calculate (kalman) mean (2 / 2 - updated X with randomness)
        self.TSP_ii_kalman_predictive_mean()
        # calculate (kalman) pred (2 / 2 - updated X with randomness)
        self.TSP_ii_kalman_predictive_cov()

    def runKalmanFilterStepsPFAdapted_JumpByJumpSkew(self, y, t_obs, particle_number, kalman_state_evo_dict, skew_var_controlled):

        # -------------------------------------------------------------------------------------------------------

        # run PF as normal. between data points we then run this adapted gunction where for teh time between
        # data pionts we loop through vi -> vi+1 and continually re-calculate the Kalman Predictive parameters
        # only at the end of the Vi's we encounter our 'data point' at which point we then run a Kalman update
        # takes us back to the main PF loop, where we calculate the Et and kt_x etc to calc the weight
        # and then we go for a new particle, where we start the kalman equations with the prev. 'correction' terms

        # -------------------------------------------------------------------------------------------------------


        # amend caligB so that it is 3x3 now
        self.caligB = np.eye(3)
        print('TODO: if we track skew, set self.caligB=np.eye3 at start')

        obs_times = t_obs
        particle_num = particle_number
        particle = self.pf.particle_set[particle_number]

        # this fn takes in a particle. need to re-update the existing kalman gains, cov and mean of this 'FilterProcess' class for the correct particle
        # these attributes are always set to the particles updated Cov and Mean, because these are computed at the end of the time interval (when we see the new obs) and
        # so are fed into the next tiem interval to calculate the predicted estimates
        self.kalman_gain = self.pf.particle_set[particle_number]['K_gain']
        self.kalman_cov = self.pf.particle_set[particle_number]['K_correction_density'][
            'cov']  # make these the 'corrected' ones, as upon entering each new particle, we want to access the last most recent values ie the 'corrected' densoty (in the simple kalman case, we compute the predictive params first, where these use the previously corrected params to do so
        self.kalman_mean = self.pf.particle_set[particle_number]['K_correction_density']['mean']

        kalman_state_evo_dict = self.hist_record_states_kalman_mean(kalman_state_evo_dict, particle_num,
                                                                    self.kalman_mean)

        self.X = self.pf.particle_set[particle_number]['X'][-1]

        # for i in range(len(t_obs) - 1):
        start_time = obs_times[0]
        end_time = obs_times[1]

        step_gamma_obj = GammaDistr(alpha=1, beta=0.1)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time,
                                              sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
        # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
        # latent_gamma_path.append(step_gamma_path[0])

        # jump_time_set_between_data_points = (0, 0)
        i = 0
        for jump_time_tuple in step_gamma_jump_time_set:
            
            v_t = step_gamma_jump_time_set[i][1]
            if i == len(step_gamma_jump_time_set) - 1:
                v_next_jump_t = end_time # if we are at the last jump in the interval, the next time is actuallt the end time ie when the data point arrives. gamma jt set is the jumps and times between data point
            else:
                v_next_jump_t = step_gamma_jump_time_set[i + 1][1]

            # need to edit the calc jumps raw mean function BELOW : 
            #NOTE - start_time and end_time are our DEFINED s and t ie. time between data points (THIS IS NOT AN ERROR - this just checks some conditions)
            mean, cov = self.StateSimObject.calculate_jumps_raw_mean_and_cov([jump_time_tuple], start_time,
                                                                         end_time)

            e = self.compute_noise_vector(self.var, cov)
            dt = v_next_jump_t - v_t
            # scale in following function = std
            sigma_squared_mu = skew_var_controlled
            # print('Currently sigma_squared_mu = 1 - which may NOT be true / make sure I set this as a function argument')
            e3 = self.StateSimObject.rng.normal(loc=0, scale=np.sqrt(sigma_squared_mu * dt) )
            e = np.append(e, e3).T
            

            new_state = self.update_state_vec_3d(e, v_t, v_next_jump_t,
                                             mean)  # this also re-calculates self.caligA for a new particle's set of jumps (uses the MEAN as input)


            # self.pf.particle_set[particle_num]['X'].append(new_state)  # TODO: Append new state
            #
            # self.pf.state_evo_dict = kalman_state_evo_dict

        # x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

            indep_cov_term = dt * sigma_squared_mu
            cov_appended = np.array( [ [cov[0][0], cov[0][1], 0], [cov[1][0], cov[1][1], 0], [0, 0, indep_cov_term] ])
            self.kalman_predict(v_t, v_next_jump_t, self.var, mean, cov_appended)
            # self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean,
            #                                  self.kalman_cov)
            i += 1

        self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean,
                                          self.kalman_cov)

        self.pf.particle_set[particle_num]['X'].append(new_state)  
        # we record the new state at the END of the predivtion recursions ie when we see a new data point
        # in theory, should be the same regardless of approach we did before

        self.pf.state_evo_dict = kalman_state_evo_dict

        # y = self.observe_state()
        # argument y below is our 'noisy observation' (from the KF we ran before)
        self.kalman_update(y, self.var, self.kv, start_time,
                           end_time)  # updates the kalman C so that in the next time we come in to PREEDICT, ie, have seen a new data point arrive, we use the UPDATED estimate as out t-1 estimate etc
        self.pf.update_particle_K_attributes(particle_num, 'updated', self.kalman_gain, self.kalman_mean,
                                             self.kalman_cov)

        return None

    def runKalmanFilterStepsPFAdapted_TwoStepProcedure(self, y, t_obs, particle_number, kalman_state_evo_dict,
                                                     skew_var_controlled):

        # -------------------------------------------------------------------------------------------------------

        # run PF as normal. between data points we then run this adapted gunction where for teh time between
        # data pionts we loop through vi -> vi+1 and continually re-calculate the Kalman Predictive parameters
        # only at the end of the Vi's we encounter our 'data point' at which point we then run a Kalman update
        # takes us back to the main PF loop, where we calculate the Et and kt_x etc to calc the weight
        # and then we go for a new particle, where we start the kalman equations with the prev. 'correction' terms

        # -------------------------------------------------------------------------------------------------------

        # amend caligB so that it is 3x3 now
        self.caligB = np.eye(3)
        print('TODO: if we track skew, set self.caligB=np.eye3 at start')

        obs_times = t_obs
        particle_num = particle_number
        particle = self.pf.particle_set[particle_number]

        # this fn takes in a particle. need to re-update the existing kalman gains, cov and mean of this 'FilterProcess' class for the correct particle
        # these attributes are always set to the particles updated Cov and Mean, because these are computed at the end of the time interval (when we see the new obs) and
        # so are fed into the next tiem interval to calculate the predicted estimates
        self.kalman_gain = self.pf.particle_set[particle_number]['K_gain']
        self.kalman_cov = self.pf.particle_set[particle_number]['K_correction_density'][
            'cov']  # make these the 'corrected' ones, as upon entering each new particle, we want to access the last most recent values ie the 'corrected' densoty (in the simple kalman case, we compute the predictive params first, where these use the previously corrected params to do so
        self.kalman_mean = self.pf.particle_set[particle_number]['K_correction_density']['mean']

        kalman_state_evo_dict = self.hist_record_states_kalman_mean(kalman_state_evo_dict, particle_num,
                                                                    self.kalman_mean)

        self.X = self.pf.particle_set[particle_number]['X'][-1]

        # for i in range(len(t_obs) - 1):
        start_time = obs_times[0]
        end_time = obs_times[1]

        step_gamma_obj = GammaDistr(alpha=1, beta=0.1)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time,
                                              sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
        # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
        # latent_gamma_path.append(step_gamma_path[0])

        # jump_time_set_between_data_points = (0, 0)
        i = 0
        for jump_time_tuple in step_gamma_jump_time_set:

            v_t = step_gamma_jump_time_set[i][1]
            if i == len(step_gamma_jump_time_set) - 1:
                v_next_jump_t = end_time  # if we are at the last jump in the interval, the next time is actuallt the end time ie when the data point arrives. gamma jt set is the jumps and times between data point
            else:
                v_next_jump_t = step_gamma_jump_time_set[i + 1][1]
            i += 1

            # calculate_F1 (also calculates matrix expon. exp(A(dt)))
            self.calculate_F1(v_next_jump_t, v_t)
            # calculate covariance mat for part i - ie random skew; [ [0 0 0] [0 0 0] [0 0 (sigma^2)dt] ]
            self.compute_e_i_cov(skew_var_controlled, v_next_jump_t, v_t)
            # calculate_e_i (NOTE - think about doing this w a B matrix etc...)
            e = self.calculate_e_i(v_next_jump_t, v_t)
            # Update 3D state vec - this is the DETERMINISTIC part of the state space model + random skew deviation
            new_state = self.update_state_vec_TSP_i(e)

            # run part i/ii of the 2-stage prediction (1 / 2 - updated mu)
            self.TSP_i_kalman_predict()

            # calculate jumps mean m_vec and covariance S_mat to use in F2:
                # NOTE - start_time and end_time are our s and t ie. time between data points
            mean, cov = self.StateSimObject.calculate_jumps_raw_mean_and_cov([jump_time_tuple], start_time, end_time)
            # calculate_F2
            self.calculate_F2(mean)
            # calculate_e_ii ( e1, e2 of the 3D e vector )
            e = self.calculate_e_ii(self.var, cov) # self.var is the variance of the NG sim prev. CHECK THAT WHEN I RUN KALMAN, I RE-MAKE THE NG SIM I USE THIS RIGHT VAR ! 
            # update 3D state vec
            new_state = self.update_state_vec_TSP_ii(e)

            # run part ii/ii of the 2-stage prediction (1 / 2 - updated mu)
            self.TSP_ii_kalman_predict()
            
    

            # we are now at 2/2, so, we NOW record the kalman predictive mean and predictive cov as we are at the
            # 'end' of the cycle and update the right attributes / entries of the dictionary as we will then use
            # these values at the start of the next 2-part-prediction loop

            # TODO: going to have to make new kalman predict / update? functions as existing ones use caligA which now changes..
            # call it TPP_.... (for Two Part Procedure)




            
            

            
            # dt = v_next_jump_t - v_t
            # scale in following function = std
            # sigma_squared_mu = skew_var_controlled
            # print('Currently sigma_squared_mu = 1 - which may NOT be true / make sure I set this as a function argument')
            # e3 = self.StateSimObject.rng.normal(loc=0, scale=np.sqrt(sigma_squared_mu * dt))
            # e = np.append(e, e3).T

            # new_state = self.update_state_vec_3d(e, v_t, v_next_jump_t,
            #                                      mean)  # this also re-calculates self.caligA for a new particle's set of jumps (uses the MEAN as input)

            # self.pf.particle_set[particle_num]['X'].append(new_state)  # TODO: Append new state
            #
            # self.pf.state_evo_dict = kalman_state_evo_dict

            # x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

            # indep_cov_term = dt * sigma_squared_mu
            # cov_appended = np.array([[cov[0][0], cov[0][1], 0], [cov[1][0], cov[1][1], 0], [0, 0, indep_cov_term]])


            # self.kalman_predict(v_t, v_next_jump_t, self.var, mean, cov_appended)
            # self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean,
            #                                  self.kalman_cov)


        self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean,
                                             self.kalman_cov)

        self.pf.particle_set[particle_num]['X'].append(new_state)
        # we record the new state at the END of the predivtion recursions ie when we see a new data point
        # in theory, should be the same regardless of approach we did before

        self.pf.state_evo_dict = kalman_state_evo_dict

        # y = self.observe_state()
        # argument y below is our 'noisy observation' (from the KF we ran before)
        self.kalman_update(y, self.var, self.kv, start_time,
                           end_time)  # updates the kalman C so that in the next time we come in to PREEDICT, ie, have seen a new data point arrive, we use the UPDATED estimate as out t-1 estimate etc
        self.pf.update_particle_K_attributes(particle_num, 'updated', self.kalman_gain, self.kalman_mean,
                                             self.kalman_cov)

        return None



    def calculate_M_and_P_matrix(self, jump_time_set, end_time):

        # M = np.zeros((2, len(jump_time_set)))
        jumps, times = np.array(jump_time_set).T
        times = times + [end_time]

        # # compute the matrix exponential of A*(t - times) for all times and a single value of t
        # expm_vec = np.vectorize(expm)
        # expA = expm_vec(np.multiply(self.langevin_A, end_time - times))

        # # compute the transformed values using numpy broadcasting and vectorization
        h = np.array([[0],[1]])

        # M = jumps * expA @ h        # M = np.zeros((2, len(jump_time_set)))
        # jumps, times = np.array(jump_time_set).T
        # times = times + [end_time]

        dt = end_time - np.array(times)
        dt = dt[:, np.newaxis, np.newaxis]

        # # compute the matrix exponential of A*(t - times) for all times and a single value of t
        expm_vec = np.vectorize(expm, signature='(m,n)->(m,n)')
        expA = expm_vec(self.langevin_A * dt)

        # # compute the transformed values using numpy broadcasting and vectorization
        h = np.array([[0],[1]])

        M = np.array(jumps)[:, np.newaxis, np.newaxis] * expA @ h
        # RESHAPE M
        M = np.reshape(np.transpose(M, axes=(2,1,0)), (2, len(times)))

        # sigma_w == self.var == sigma_w : sigma_w is the var of the NG process, which I call 'var' in the code
        sigma_w = self.var
        P = np.sqrt(np.array(jumps * (sigma_w**2))[:, np.newaxis, np.newaxis] ) * expA @ h
        # RESHAPE P
        P = np.reshape(np.transpose(P, axes=(2,1,0)), (2, len(times)))

        self.M = M
        self.P = P

    def calculate_P_and_M_matrices_2(self, jump_time_set, end_time):

        jumps, times = np.array(jump_time_set).T
        times = times + [end_time]
        h = np.array([[0], [1]])
        # sigma_w = self.var
        sigma_w = 1 # marginalised form?

        M = np.array([[], []])
        P = np.array([[], []])

        jt_set = jump_time_set + [(0, end_time)]
        for jump_time_tuple in jt_set:
            jump = jump_time_tuple[0]
            time = jump_time_tuple[1]
            expA_times_h = expm(self.langevin_A * (end_time - time)) @ h

            M_entry = expA_times_h * jump

            M = np.append(M, M_entry, axis=1)

            P_entry = expA_times_h * np.sqrt(jump * (sigma_w**2))

            P = np.append(P, P_entry, axis=1)
        self.M = M
        self.P = P

    def calculate_D_matrix(self, tdiffs):
        self.D = np.tri(len(tdiffs), len(tdiffs))
        pass

    def calculate_Lambda_matrix(self, tdiffs):
        # self.Lambda = self.sigma_mu_sq * np.diag(tdiffs)

        sigma_mu_sq = self.k_mu_BM * 1 # the 1 should be self.var however we say var = 1 as we marginalised it
        self.Lambda = sigma_mu_sq * np.diag(tdiffs)

        pass


    def produce_skew_matrices(self, start_time, end_time, jump_time_set):

        jtimes = [start_time] + [pair[1] for pair in jump_time_set] + [end_time]
        tdiffs = np.diff(jtimes)

        self.calculate_D_matrix(tdiffs)
        self.calculate_Lambda_matrix(tdiffs)

    def produce_I_matrices(self, start_time, end_time, jump_time_set):

        if len(jump_time_set) == 0:
            jump_time_set = [(0, start_time), (0, end_time)]
        # self.calculate_M_and_P_matrix(jump_time_set, end_time)
        self.calculate_P_and_M_matrices_2(jump_time_set, end_time)

        self.produce_skew_matrices(start_time, end_time, jump_time_set)

    def compute_caligA_dynamic_skew(self, end_time, start_time):
        self.caligA = np.block([[expm(self.langevin_A * (end_time - start_time)), self.M @ np.ones(((np.shape(self.M))[1], 1))],
                                [np.zeros((1, 2)), 1.]])


    def calculate_I_mean_cov(self):
        pass


    def compute_noise_vector_dynamic_skew(self):
        # alpha_t = A alpha_s + B e_state

        # noise vector = e_state ~ N(0, S)
        # S = C1 + C2 : C1 = k_mu_BM * sigma_w_sq * [ [  ] [   ] [  ] ] , C2 = sigma_w_sq * [ [P I P^T 0] [0 0 0] ]
        # S = sigma_w_sq * [ C1 + C2 ]


        # 1/ compute C1
        D_lambda_D = self.D @ self.Lambda @ self.D.T
        MD_lambda_D = self.M @ D_lambda_D
        D_lambda_DM = D_lambda_D @ self.M.T

        C1 = self.k_mu_BM * np.block([[ MD_lambda_D @ self.M.T, MD_lambda_D[:, [-1]] ],
                                      [ D_lambda_DM[-1,:],  D_lambda_D[-1, -1] ]])
        # 2/ compute C2
        C2 = np.block( [ [self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T, np.zeros((2,1))], [np.zeros((1,3))]] )

        # 3/ add them and times by sigma_w_s1 !

        # var = self.var
        var = 1
        S = var * (C1 + C2)

        self.e_state_cov = S

        # e = np.random.multivariate_normal([0,0,0],S)
        # e = np.reshape(e, (3,1))

        try:
            cov_chol = np.linalg.cholesky(S)
            e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=3)]) + np.zeros((3,1)) # used to have + mean here - but now zeros
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            e = np.zeros((3,1))

        return e

    def update_state_vec_3d_dynamic_skew(self, e):
        self.X = self.caligA @ self.X + e
        return self.X


    def runKalmanFilterPFAdapted_DynamicSkewClosedForm(self, y, beta, t_obs, particle_number, kalman_state_evo_dict):

        obs_times = t_obs
        particle_num = particle_number
        particle = self.pf.particle_set[particle_number]

        # this fn takes in a particle. need to re-update the existing kalman gains, cov and mean of this 'FilterProcess' class for the correct particle
        # these attributes are always set to the particles updated Cov and Mean, because these are computed at the end of the time interval (when we see the new obs) and
        # so are fed into the next tiem interval to calculate the predicted estimates
        self.kalman_gain = self.pf.particle_set[particle_number]['K_gain']
        self.kalman_cov = self.pf.particle_set[particle_number]['K_correction_density'][
            'cov']  # make these the 'corrected' ones, as upon entering each new particle, we want to access the last most recent values ie the 'corrected' densoty (in the simple kalman case, we compute the predictive params first, where these use the previously corrected params to do so
        self.kalman_mean = self.pf.particle_set[particle_number]['K_correction_density']['mean']

        kalman_state_evo_dict = self.hist_record_states_kalman_mean(kalman_state_evo_dict, particle_num,
                                                                    self.kalman_mean)

        self.X = self.pf.particle_set[particle_number]['X'][-1]

        start_time = obs_times[0]
        end_time = obs_times[1]

        step_gamma_obj = GammaDistr(alpha=1, beta=beta)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None,
                                              sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
        # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
        # latent_gamma_path.append(step_gamma_path[0])

        self.produce_I_matrices(start_time, end_time, step_gamma_jump_time_set)
        # mean_I, cov_I = self.calculate_I_mean_cov()

        self.compute_caligA_dynamic_skew(end_time, start_time)

        e = self.compute_noise_vector_dynamic_skew()

        new_state = self.update_state_vec_3d_dynamic_skew(e)

        self.pf.particle_set[particle_num]['X'].append(new_state)

        self.pf.state_evo_dict = kalman_state_evo_dict

        # x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

        mean = None
        self.kalman_predict(start_time, end_time, self.var, mean, self.e_state_cov)
        # TODO: MAKE SURE I ENTER RIGHT (MEAN AND) COV !!!
        self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean, self.kalman_cov)

        # y = self.observe_state()
        # argument y below is our 'noisy observation' (from the KF we ran before)
        self.kalman_update(y, self.var, self.kv, start_time,  end_time)  # updates the kalman C so that in the next time we come in to PREEDICT, ie, have seen a new data point arrive, we use the UPDATED estimate as out t-1 estimate etc
        self.pf.update_particle_K_attributes(particle_num, 'updated', self.kalman_gain, self.kalman_mean, self.kalman_cov)


        return None


    def runKalmanFilterStepsPFAdapted(self, y, beta, t_obs, particle_number, kalman_state_evo_dict):
        # y = NOISY OBSERVATION

        obs_times = t_obs
        particle_num = particle_number
        particle = self.pf.particle_set[particle_number]

        # this fn takes in a particle. need to re-update the existing kalman gains, cov and mean of this 'FilterProcess' class for the correct particle
        # these attributes are always set to the particles updated Cov and Mean, because these are computed at the end of the time interval (when we see the new obs) and
        # so are fed into the next tiem interval to calculate the predicted estimates
        self.kalman_gain = self.pf.particle_set[particle_number]['K_gain']
        self.kalman_cov = self.pf.particle_set[particle_number]['K_correction_density']['cov'] # make these the 'corrected' ones, as upon entering each new particle, we want to access the last most recent values ie the 'corrected' densoty (in the simple kalman case, we compute the predictive params first, where these use the previously corrected params to do so
        self.kalman_mean = self.pf.particle_set[particle_number]['K_correction_density']['mean']

        kalman_state_evo_dict = self.hist_record_states_kalman_mean(kalman_state_evo_dict, particle_num, self.kalman_mean)

        self.X = self.pf.particle_set[particle_number]['X'][-1]

        # for i in range(len(t_obs) - 1):
        start_time = obs_times[0]
        end_time = obs_times[1]

        step_gamma_obj = GammaDistr(alpha=1, beta=beta)
        if beta >= 0.5:
            sample_size = 50
        else:
            sample_size = 500
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None,
                                              sample_size=sample_size)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
        # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
        # latent_gamma_path.append(step_gamma_path[0])
        # if len(step_gamma_jump_time_set) == 0:
        #     print()
        #     print()
        #     print('---!!!---!!!---!!!')
        #     print(start_time)
        #     print(end_time)
        #     print('---!!!---!!!---!!!')
        #     step_gamma_jump_time_set = [(0,0)]
        # t2 = step_gamma_path[0]

        mean, cov = self.StateSimObject.calculate_jumps_raw_mean_and_cov(step_gamma_jump_time_set, start_time,
                                                                         end_time)

        e = self.compute_noise_vector(self.var, cov)

        new_state = self.update_state_vec_3d(e, start_time, end_time, mean) # this also re-calculates self.caligA for a new particle's set of jumps (uses the MEAN as input)
        
        self.pf.particle_set[particle_num]['X'].append(new_state) #TODO: Append new state

        self.pf.state_evo_dict = kalman_state_evo_dict

        # x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

        self.kalman_predict(start_time, end_time, self.var, mean, cov)
        self.pf.update_particle_K_attributes(particle_num, 'predictive', self.kalman_gain, self.kalman_mean, self.kalman_cov)

        # y = self.observe_state()
        # argument y below is our 'noisy observation' (from the KF we ran before)
        self.kalman_update(y, self.var, self.kv, start_time, end_time) #updates the kalman C so that in the next time we come in to PREEDICT, ie, have seen a new data point arrive, we use the UPDATED estimate as out t-1 estimate etc
        self.pf.update_particle_K_attributes(particle_num, 'updated', self.kalman_gain, self.kalman_mean, self.kalman_cov)

        return None


    def runKalmanFilter(self, skew, beta, dynamic_skew):
        self.X = np.array([[0], [0], [0]])
        print('Set X_initial for KF as 0, 0, 0')
        obs_times = self.obs_times
        
        # re-initialise our state vector with the given skew:
        if not dynamic_skew['Dynamic']:
            self.X[2][0] = skew
        else:
            self.X[2][0] = 0
            # skew_var_generating_process = dynamic_skew['var']
            skew_var_generating_process = self.dynamic_skew_var
            self.caligB = np.eye(3)
            #skew_var_generating_process = the sigma_squared we set in the generating BM process:
            # mu_i+1 =Mu_i + sqrt( (V_i+1 - V_i) sigma_squared ) N(0,1)
        
        

        # if t_obs == None:
        #     obs_times = self.obs_times
        # else:
        #     obs_times = t_obs
        # jt_set = self.forward_sim_latent_jt_set
        
        
        x_evolution = [ [self.X[0][0]], [self.X[1][0]], [self.X[2][0]]]

        upper_band, lower_band, kalman_mean_line, noisy_obs = [0], [0], [0], [0]
        x1_lower_band, x1_upper_band, kalman_x1_mean = [0], [0], [0]
        x2_lower_band, x2_upper_band, kalman_x2_mean = [0], [0], [0]

        latent_gamma_path = [0]

        'PARAMS TO CHECK WE ARE INFERRING VAR CORRECTLY, for a kalman filter'
        ktx = 0
        E = 0
        count = 0

        for i in range(len(obs_times) - 1):

            start_time = obs_times[i]
            end_time = obs_times[i + 1]

            if not dynamic_skew['Dynamic']:
                if start_time < 5.0:
                    skew = self.X[2][0]
                else:
                    skew = self.X[2][0]
                    self.X[2][0] = skew
                    # print('TODO: time varying skew for KF - update the X parameter; \'update_state_vec_3d\' needs \n to be edited to update the state vec based on a BMotion. righ now I manually set the skew part of X, which in practise is wrong as we dont know skew')
            # else:
            #     skew = skew + np.sqrt(skew_var_generating_process * (end_time - start_time)) * self.StateSimObject.rng.normal(loc=0, scale=1)


            step_gamma_obj = GammaDistr(alpha=1, beta=beta)
            step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None,
                                                  sample_size=450)
            step_gamma_sim = DistributionSimulator(step_gamma_obj)
            step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()
            # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
            if len(step_gamma_path) == 0:
                step_gamma_path = [0]
            latent_gamma_path.append(step_gamma_path[-1]) # was 0 before !

            mean, cov = self.StateSimObject.calculate_jumps_raw_mean_and_cov(step_gamma_jump_time_set, start_time, end_time)

            e = self.compute_noise_vector(self.var, cov)
            if dynamic_skew['Dynamic']:
                # this makes the KF an APPROXIMATION to skew-tracking
                indep_cov_term = (end_time - start_time) * skew_var_generating_process
                cov = np.array([[cov[0][0], cov[0][1], 0], [cov[1][0], cov[1][1], 0], [0, 0, indep_cov_term]])

                e3 = np.sqrt(skew_var_generating_process * (end_time - start_time))*self.StateSimObject.rng.normal(loc=0, scale=1)
                # e3 is the 3rd noise element; the random term of the skew
                e = np.append(e, e3)
                e = np.reshape(e, (3,1))

            new_state = self.update_state_vec_3d(e, start_time, end_time, mean)
            # x_evolution is our True State Path Evolution
            if not dynamic_skew['Dynamic']:
                x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state, skew=skew)
            else:
                x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)
            

            self.kalman_predict(start_time, end_time, self.var, mean, cov)

            y = self.observe_state()

            # calc ktx
            ktx += self.caligH @ self.kalman_cov @ self.caligH.T + self.kv
            # calc E
            E += (y - self.kalman_mean[0]) ** 2 / ktx
            count += 1

            self.kalman_update(y, self.var, self.kv, start_time, end_time) #TODO: Var is the normal gamma var parameter



            upper, lower, kalman_mean_pred = self.extract_confidence_bands(3, state='x0')
            upper_band.append(upper)
            lower_band.append(lower)
            kalman_mean_line.append(kalman_mean_pred)
            noisy_obs.append(y)

            x1upper, x1lower, x1kalman_mean_pred = self.extract_confidence_bands(3, state='x1')
            x1_lower_band.append(x1lower)
            x1_upper_band.append(x1upper)
            kalman_x1_mean.append(x1kalman_mean_pred)


            x2upper, x2lower, x2kalman_mean_pred = self.extract_confidence_bands(3, state='x2')
            x2_lower_band.append(x2lower)
            x2_upper_band.append(x2upper)
            kalman_x2_mean.append(x2kalman_mean_pred)

            # IF WE WANT TO PLOT / RECORD THE PREDICTIVE DENSITIES AT EACH TIME STEP... THEN RECORD THE VALUES BEFORE WE MAKE THE OBSERVATION:
            # y = self.observe_state()
            # self.kalman_update(y, self.var, self.kv, start_time,
            #                    end_time)  # TODO: Var is the normal gamma var parameter
            # noisy_obs.append(y)

        lweights = np.array([0])
        IGMean = (self.pf.eta + E/2) / (self.pf.rho + count/2 -1)
        pred_mean_var = self.pf.log_sumexp_util(lweights, lambda x: x, IGMean, retlog=False)
        print('KALMAN TEST pred mean var. = {}'.format(pred_mean_var))

        self.noisy_observations = noisy_obs
        self.x_evolution = x_evolution
        # This line readies our set-up for particle filtering, having ran a kalman filter (for the SOLE
        # PURPOSE of generating our underlying state simulation + noise-corrupted observations)
        self.pf.update_PF_attributes(data_set=noisy_obs, true_state_paths=x_evolution)


        # showing the state trajectiroes X0 and X0
        # super().show_plots(obs_times, x_evolution, placeholder_not_in_use)

        fig, ax = plt.subplots()
        ax.step(obs_times, np.cumsum(latent_gamma_path), zorder=1)
        fig.suptitle('Latent Gamma')
        plt.xlabel('Time')
        plt.ylabel('Gamma Path')
        plt.show()
        
        x0_data = [lower_band, upper_band, kalman_mean_line, x_evolution, noisy_obs]
        x1_data = [x1_lower_band, x1_upper_band, kalman_x1_mean, x_evolution]
        x2_data = [x2_lower_band, x2_upper_band, kalman_x2_mean]
        self.plot_kalman_results(x0_data, x1_data, x2_data, beta)



        # show the kalman prediction overlaid on real state trajectories, with +-99% confidence bands

        return None


    def runParticleFilter(self, Np, beta, k_cov_init, X0_INIT, state_obs_and_times, show_particle_paths, dynamic_skew):

        # for each particle: we need to store several things
        # 1. particle number
        # 2. Evolution of E
        # 3. (unnormalised) log weight at each time-step
        # 4. evoltuion of its observation y_i (ie an observation state vector for each particle)
        # 5.

        if dynamic_skew['Dynamic']:
            print('Check')
            self.caligB = np.eye(3)

        print()
        print('BETA SET TO 1 in DYNAMIC SKEW CLOSED FORM KF FUNCTION, and set intial cov very low in initialse_particle_state_and_ktx')
        print()

        pf = self.pf
        pf.Np = Np
        num_particles = pf.Np
        self.pf.particle_set = {particle_no: {'X':[], 'log-weight': -np.log(num_particles), 'E': [0], 'kt_x':[], 'count':0,
                                         'y_pred': [], 'K_gain': np.zeros((3,1)), 'K_predictive_density': {'mean': np.array([[X0_INIT], [0], [0]]), 'cov': k_cov_init*np.eye(3)},
                                         'K_correction_density': {'mean': np.array([[X0_INIT], [0], [0]]), 'cov': k_cov_init*np.eye(3)}}
                            for particle_no in list(np.arange(1,num_particles+1,1))}

        particle_evo_dict = {}
        for i in range(1,num_particles+1):
            pf.initialise_particle_state_and_ktx(self.kv, i) #TODO: initialise particle from prior - Initialise 'k_x' as H@C_1@H + kv, initialise y_pred as H@A1|1. Initialise the means and covs as: [0, 0, 0], and np.zeros(3,3)
            particle_evo_dict[i] = []

            # pf.particle_set[i]['X'] = particle
            #set E_0 (i) = 0

        if state_obs_and_times == 'pre-gen':
            times = pf.obs_times # THESE ARE THE FULL COLLECTION OF OBSERVATION TIMES
            observation_data = pf.data_set_y_noisy_obs
        elif state_obs_and_times != 'pre-gen':
            # UNZIP argument to extract the state obs and times
            times = state_obs_and_times[0]
            observation_data = state_obs_and_times[1]

            pass

        print('CHECK INITIALISATION OF KALMAN PRED AND UPDT MEAN AND COV IN THE PARTICLE SET DICTIONARY! CHECK ORDER OF HOW THINGS ARE UPDATED')


        state_mean = []
        state_cov = []
        var_estimate_list = []

        print('CHECK: Do we start count at 0 or 1? ie is the first sampled state t=0 or t=1, and therefore, should count == the length of the final state vector?')
        for t in range(1,len(times)):
            start_time = times[t-1]
            end_time = times[t]
            print()
            print('-------------')
            print('time = {}'.format(end_time))
            print('-------------')

            noisy_observation = observation_data[t] # our observation is at the time interval end as we are trying to predict for the end of the time interval, before seeing the 'true' state evolution

            for pn in range(1, num_particles+1):
                self.pf.particle_set[pn]['count'] += 1

            for i in range(1, num_particles+1): # i = particle number (i: 1 -> Np)

                if i == num_particles / 5 or i == 2/5 * num_particles or i == 3/5 * num_particles or i == 4/5 * num_particles or i == num_particles:
                    print('PARTICLE NUM: {} / {}'.format(i, num_particles))

                # simulate jumps in interval t-1 -> t (a particle over this time period)
                # calculate PREDICTED kalman a and C
                # observe new OBSERVATION (noisy) data point y_i
                # calculate UPDATED kalman paramaters a and C
                
                if not dynamic_skew['Dynamic']:
                    sigma_squared_mu = 'NotDynamic'
                    self.runKalmanFilterStepsPFAdapted(y=noisy_observation, beta=beta, t_obs=[start_time, end_time], particle_number=i, kalman_state_evo_dict=particle_evo_dict)
                else:
                    # This is the value which WE control, when WE run the PF
                    sigma_squared_mu = self.dynamic_skew_var
                    # self.runKalmanFilterStepsPFAdapted_JumpByJumpSkew(y=noisy_observation, t_obs=[start_time, end_time], particle_number=i, kalman_state_evo_dict=particle_evo_dict, skew_var_controlled=sigma_squared_mu)
                    # self.runKalmanFilterStepsPFAdapted_TwoStepProcedure(y=noisy_observation, t_obs=[start_time, end_time], particle_number=i, kalman_state_evo_dict=particle_evo_dict, skew_var_controlled=sigma_squared_mu)

                    self.runKalmanFilterPFAdapted_DynamicSkewClosedForm(y=noisy_observation, beta=beta, t_obs=[start_time, end_time], particle_number=i, kalman_state_evo_dict=particle_evo_dict)

                y_hat = self.pf_predict_y(particle=self.pf.particle_set[i])
                self.pf.particle_set[i]['y_pred'].append(y_hat)

                ktx = self.pf_calc_ktx(particle=self.pf.particle_set[i])
                self.pf.particle_set[i]['kt_x'].append(ktx)

                # CALCULATE UN-NORM SMC WEIGHT FOR THE JUMPS 1 -> t for this particle i using marginal sigma. sq case

                Et = self.pf_calc_Et(particle=self.pf.particle_set[i], y=noisy_observation)
                self.pf.particle_set[i]['E'].append(Et)

                # HERE HERE WE now impliment that fat marginalised sigma squared equation ... as we have calculated the Et and the ktx necessary for such expression

                # pf.particle_set[i]['count'] += 1
                # print('count = {}'.format(self.pf.particle_set[i]['count']))
                log_weight = self.pf_calc_log_weight(particle=self.pf.particle_set[i])
                self.pf.particle_set[i]['log-weight'] += log_weight # update log weight; in w_new = w_old * p -> log domain -> log_w = log_w_old + log p

                # print('len X for particle {} = {}'.format(i, len(self.pf.particle_set[i]['X'])))
            # calculate Lt here (optional?)

            pf.normalise_weights()

            mixed_mean, mixed_cov = pf.compute_state_posterior()

            log_ESS = pf.calculate_ESS(method='d_inf')
            # print('log_ESS = {}. epsilon * N = {}'.format(log_ESS, pf.epsilon * num_particles))
            if log_ESS < np.log(pf.epsilon * num_particles): # CHECK RESAMPLING LOGIC !!! NOT SURE IT IS DOING CORECT THING (NOTE - the < ought to be changed to > ! ) I THINK THAT WE DONT GET THE SAME NUMBER OF PARTICLES AGAIN... CHECK TOMO MORN!
                self.pf_multinomial_resample()

            # now calculate the predictive density parameters and store a series of a_mix and C_mix so that I can plot the 'bands' !
            # Calcualte density pi(alpha_t | y_1:t) using weights as Mixture of gaussian weights. density is parameterised by some mean and covariance;

            # mixed_mean, mixed_cov = pf.compute_state_posterior()


            state_mean.append(mixed_mean)
            state_cov.append(mixed_cov)

            lweights = np.array([self.pf.particle_set[particle_num]['log-weight'] for particle_num in
                                                          list(self.pf.particle_set.keys())]).flatten()
            Es = np.array([self.pf.particle_set[particle_num]['E'][-1]
                                for particle_num in list(self.pf.particle_set.keys())])
            E = self.pf.log_sumexp_util(lweights, lambda x: x, Es, retlog=False)
            count = int(self.pf.particle_set[1]['count'])
            var_estimate = (self.pf.eta + E/2.)/(self.pf.rho+count/2.+1.)

            var_estimate_list.append(var_estimate[0][0])

        # gather a vector of all particles end weights.
        lweights = np.array([self.pf.particle_set[particle_num]['log-weight'] for particle_num in
                              list(self.pf.particle_set.keys())]).flatten()
        # weights =
        # gather a vector of the Inverse Gammas mean, for each particle IG(alpha, beta); IG mean = beta / alpha -1
        # IG mean = (eta + Et/2) / (rho+t/2 - 1)
        IGmeans = np.array([ (self.pf.eta + (self.pf.particle_set[particle_num]['E'][-1]).flatten()/2) / (self.pf.rho + self.pf.particle_set[particle_num]['count']/2 -1)
                    for particle_num in list(self.pf.particle_set.keys())])
        # use the log sum exp function, but have h(x) as the function which calculates the mean of the IG distr?
        pred_mean_var = self.pf.log_sumexp_util(lweights, lambda x: x, IGmeans, retlog=False)
        print('pred mean var. = {}'.format(pred_mean_var))



        print('VAR ESTIMATE = {}'.format(var_estimate))


        fig101, ax101 = plt.subplots()
        ax101.plot(np.linspace(1, len(var_estimate_list), len(var_estimate_list)), np.array(var_estimate_list))
        fig101.suptitle('Var Estimate Evolution w/ timestep')
        plt.show()

        xx = np.linspace(0.1, 5, 10000)
        alpha = self.pf.rho + count/2
        gamma_alpha = gamma_func(alpha) # part of the norm constant
        B = self.pf.eta + E[0][0] / 2
        beta_to_alpha = (B)**(alpha) # part of the norm constant
        inv_gamma_pdf = []
        for x in xx:
            inv_gamma_pdf.append( -(alpha+1)*np.log(x) - B/x )

        fig999, ax999 = plt.subplots()
        ax999.plot(xx,inv_gamma_pdf, 'r-', lw=1, alpha=0.6, label='invgamma pdf')
        fig999.suptitle('Inv Gamma Distr For Estimated Var')
        plt.show()





        # a = 1
        # x = np.linspace(invgamma.ppf(0.01, a),
        #                 invgamma.ppf(0.99, a), 100)
        # fig99, ax99 = plt.subplots()
        # ax99.plot(x, invgamma.pdf(x, a, loc = (self.pf.rho + count/2), scale = (self.pf.eta + E/2)),
        #         'r-', lw=5, alpha=0.6, label='invgamma pdf')
        # plt.show()

        lweights = np.array([self.pf.particle_set[particle_num]['log-weight'] for particle_num in
                                 list(self.pf.particle_set.keys())]).flatten()
        Es = np.array([self.pf.particle_set[particle_num]['E'][-1]
                              for particle_num in list(self.pf.particle_set.keys())])
        count = int(self.pf.particle_set[1]['count'])
        mixture = np.zeros(10000)
        mode = 0.
        mean = 0.
        axis = np.linspace(0.1, 5, 10000)
        E = self.pf.log_sumexp_util(lweights, lambda x: x, Es, retlog=False)
        mixture = self.sigma_posterior(axis, count, E)
        mixture = mixture[0]
        mode = (self.pf.eta + E / 2.) / (self.pf.rho + count / 2. + 1.)
        mean = (self.pf.eta + E / 2.) / (self.pf.rho + count / 2. - 1.)
        fig90009, ax90009 = plt.subplots()
        # ax90009.plot(axis, mixture)
        ax90009.plot(axis[500:7500], mixture[500:7500]) # - self.pf.log_sumexp_util(mixture, lambda x: 1., np.zeros(mixture.shape[0]), retlog=False))
        fig90009.suptitle('Inv Gamma - JJ METHOD')
        plt.show()

        
        SCALE_FACTOR_ESTIMATE = 1
        print('manually scaling var by: {}'.format(SCALE_FACTOR_ESTIMATE))
        x0lower, x0upper, x0mid = self.compute_CIs(means=state_mean, covs=SCALE_FACTOR_ESTIMATE* var_estimate * state_cov, stdevs=3, state='x0')
        x1lower, x1upper, x1mid = self.compute_CIs(means=state_mean, covs=SCALE_FACTOR_ESTIMATE* var_estimate * state_cov, stdevs=3, state='x1')
        x2lower, x2upper, x2mid = self.compute_CIs(means=state_mean, covs=SCALE_FACTOR_ESTIMATE* var_estimate * state_cov, stdevs=3, state='x2')


        x0_pf_data = [x0lower, x0mid, x0upper]
        x1_pf_data = [x1lower, x1mid, x1upper]
        x2_pf_data = [x2lower, x2mid, x2upper]

        range_start = 20
        fig2, ax2 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, pf.true_x_evo[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK

        ax2.plot(times[range_start:], x0mid[range_start -1:], linestyle='--', label='PF Mean')
        ax2.scatter(times[range_start:], observation_data[range_start:], marker='x', s=4, color='k', label='Noisy State Observations')
        ax2.fill_between(times[range_start:], x0upper[range_start - 1:], x0lower[range_start - 1:], alpha=0.4, label='PF +-99% CI')
        ax2.legend()
        fig2.suptitle('X0 Evolution - Particle Filter - Updated - kv = {}. Np = {} \n beta ={}, k_cov_init = {}'.format(self.kv, num_particles, beta, k_cov_init))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig3, ax3 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK

        ax3.plot(times[range_start:], x1mid[range_start - 1:], linestyle='--', label='PF Mean')
        if state_obs_and_times == 'pre-gen':
            ax3.scatter(times[:], pf.true_x_evo[1][:], marker='x', s=4, color='k', label='True X1 State Underlying')
            ax3.plot(times, pf.true_x_evo[1][:], linestyle='--', alpha=0.7, color='r', label='True X1 State Underlying')
        ax3.fill_between(times[range_start:], x1upper[range_start - 1:], x1lower[range_start - 1:], alpha=0.4, label='PF +-99% CI')
        ax3.legend()
        fig3.suptitle('X1 Evolution - Particle Filter - Updated - kv = {}. Np = {}\n beta ={}, k_cov_init = {}'.format(self.kv, num_particles, beta, k_cov_init))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig4, ax4 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK

        ax4.plot(times[range_start:], x2mid[range_start - 1:], linestyle='--', label='PF Mean')
        # ax3.scatter(times, pf.true_x_evo[1][:], marker='x', s=4, color='k', label='True X1 State Underlying')
        if state_obs_and_times == 'pre-gen':
            ax4.plot(times, pf.true_x_evo[2][:], linestyle='--', alpha=0.7, color='r', label='True X2 State Underlying')
        ax4.fill_between(times[range_start:], x2upper[range_start - 1:], x2lower[range_start - 1:], alpha=0.4, label='PF +-99% CI')
        ax4.legend()
        fig4.suptitle('X2 - Particle Filter - k_mu_BM = {} kv = {}. Np = {} \n beta = {}, k_cov_init = {}'.format(self.k_mu_BM, self.kv, num_particles, beta, k_cov_init))
        plt.xlabel('Time')
        plt.ylabel('X2')
        plt.show()



        # self.plot_particles_2()

        # self.plot_2D_hist(x0_pf_data, x1_pf_data)
        # self.jj_plot()
        if show_particle_paths:
            self.show_results(x0_pf_data, x1_pf_data, x2_pf_data, sigma_squared_mu)



    def jj_plot(self):

        times = self.pf.obs_times

        x0_unr = []
        x1_unr = []

        x0s = []
        x1s = []
        threeDtimes = []
        for num in self.pf.particle_set.keys():
            particle_states = self.pf.particle_set[num]['X']

            state_traj = [state[0][0] for state in particle_states]
            state_traj_x1 = [state[1][0] for state in particle_states]

            x0s.append(state_traj)
            threeDtimes.append(list(self.pf.obs_times))

            x1s.append(state_traj_x1)

            for state in particle_states:
                x0 = state[0][0]
                x0_unr.append(x0)

                x1 = state[1][0]
                x1_unr.append(x1)
        
        
        
        fig, ax1 = plt.subplots()

        N = self.pf.Np
        num_fine = 2000
        t_fine = np.linspace(np.min(times), np.max(times), num_fine)
        # ax1.axvline(x=times[-2], linestyle='--', color='orange')
        # ax2.axvline(x=times[-2], linestyle='--', color='orange')
        states_fine = np.empty((num_fine, N), dtype=float)
        grads_fine = np.empty((num_fine, N), dtype=float)
        skews_fine = np.empty((num_fine, N), dtype=float)
        for i in range(N):
            states_fine[:, i] = np.interp(t_fine, times, x0s[i])
            grads_fine[:, i] = np.interp(t_fine, times, x1s[i])
            # skews_fine[:, i] = np.interp(t_fine, times, skews[:, i])
        states_fine = (states_fine.T).flatten()
        grads_fine = (grads_fine.T).flatten()
        # skews_fine = (skews_fine.T).flatten()
        t_fine = repmat(t_fine, N, 1).flatten()
        cmap = plt.cm.plasma
        cmap.set_bad(cmap(0))

        sh, sxedges, syedges = np.histogram2d(t_fine, states_fine, bins=[400, 200], density=True)
        pcm = ax1.pcolormesh(sxedges, syedges, sh.T, cmap='plasma', norm=LogNorm(vmax=1.5), rasterized=True)

        plt.show()


    def show_results(self, x0data, x1data, x2data, sigma_squared_mu):

        x0lower, x0mid, x0upper = x0data
        x1lower, x1mid, x1upper = x1data
        x2lower, x2mid, x2upper = x2data

        x0_unr = []
        x1_unr = []

        x0s = []
        x1s= []
        x2s = []
        threeDtimes = []
        for num in self.pf.particle_set.keys():
            particle_means = self.pf.state_evo_dict[num]

            state_traj = [state[0][0] for state in particle_means]
            state_traj_x1 = [state[1][0] for state in particle_means]
            state_traj_x2 = [state[2][0] for state in particle_means]

            x0s.append(state_traj)
            threeDtimes.append(list(self.pf.obs_times))

            x1s.append(state_traj_x1)
            x2s.append(state_traj_x2)

            for state in particle_means:
                x0 = state[0][0]
                x0_unr.append(x0)

                x1 = state[1][0]
                x1_unr.append(x1)

        observation_data = self.pf.data_set_y_noisy_obs

        times = self.pf.obs_times
        unr_times = list(times) * self.pf.Np

        fig, ax = plt.subplots()

        # Loop through each particle and create a line
        for i in range(len(x0s)):
            # Create a line for the current particle
            line, = ax.plot(threeDtimes[i][:len(x0s[i])], x0s[i], '-', linewidth=1, alpha=0.05, color='black')

        ax.plot(times, self.pf.true_x_evo[0][:], zorder=1, linestyle='--', color='r', alpha=0.7,
                  label='True X0 State')  # HUHJKHK
        #
        ax.plot(times, x0mid, linestyle='--', label='PF Mean')
        ax.scatter(times, observation_data, marker='x', s=4, color='k', label='Noisy State Observations')
        ax.fill_between(times, x0upper, x0lower, alpha=0.1, label='PF +-99% CI')
        ax.legend()
        fig.suptitle('X0 Evolution - Particle Filter - Updated - kv = {}. Np = {}'.format(self.kv, self.pf.Np))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig, ax = plt.subplots()

        # Loop through each particle and create a line
        for i in range(len(x1s)):
            # Create a line for the current particle
            line, = ax.plot(threeDtimes[i][:len(x1s[i])], x1s[i], '-', linewidth=1, alpha=0.05, color='black')

        ax.plot(times, self.pf.true_x_evo[1][:], zorder=1, linestyle='--', color='r', alpha=0.7,
                label='True X1 State')  # HUHJKHK
        #
        ax.plot(times, x1mid, linestyle='--', label='PF Mean')
        # ax.scatter(times, observation_data, marker='x', s=4, color='k', label='Noisy State Observations')
        ax.fill_between(times, x1upper, x1lower, alpha=0.1, label='PF +-99% CI')
        ax.legend()
        fig.suptitle('X1 Evolution - Particle Filter - Updated - kv = {}. Np = {}'.format(self.kv, self.pf.Np))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig, ax = plt.subplots()

        # Loop through each particle and create a line
        for i in range(len(x1s)):
            # Create a line for the current particle
            line, = ax.plot(threeDtimes[i][:len(x2s[i])], x2s[i], '-', linewidth=1, alpha=0.05, color='black')

        ax.plot(times, self.pf.true_x_evo[2][:], zorder=1, linestyle='--', color='r', alpha=0.7,
                label='True X2 State')  # HUHJKHK
        #
        ax.plot(times, x2mid, linestyle='--', label='PF Mean')
        # ax.scatter(times, observation_data, marker='x', s=4, color='k', label='Noisy State Observations')
        ax.fill_between(times, x2upper, x2lower, alpha=0.1, label='PF +-99% CI')
        ax.legend()
        fig.suptitle('X2 - Particle Filter - sigma_sq_mu_controlled = {} . kv = {}. Np = {}'.format(sigma_squared_mu, self.kv, self.pf.Np))
        plt.xlabel('Time')
        plt.ylabel('X2')
        plt.show()


    def plot_particles_2(self):

        # Create some sample data
        num_particles = 100
        num_timesteps = 100
        x = np.random.normal(0, 1, size=(num_particles, num_timesteps))
        y = np.random.normal(0, 1, size=(num_particles, num_timesteps))

        # Create a 2D histogram of the x and y values for all timesteps
        hist, xedges, yedges = np.histogram2d(x.flatten(), y.flatten(), bins=500)
        # Convert the histogram to a density and transpose it
        density = hist / np.sum(hist)
        density = density.T
        # Create a figure and an axis
        fig, ax = plt.subplots()
        # Plot the density as an image
        im = ax.imshow(density, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno')
        # Add a colorbar
        fig.colorbar(im, ax=ax)
        # Set the title and axis labels
        ax.set_title('Particle Trajectory')
        ax.set_xlabel('time')
        ax.set_ylabel('state')
        # Show the final figure
        plt.show()


    def plot_particles(self):

        particles = [[0.5, 0.6], [0.6, 0.5], [0.7, 0.8]]

        # Create a 2D histogram
        hist, xedges, yedges = np.histogram2d([p[0] for p in particles], [p[1] for p in particles], bins=[400, 200], density=True)

        # Create a meshgrid for the histogram
        x, y = np.meshgrid(xedges[:-1], yedges[:-1])

        # Plot the 2D histogram
        fig, ax = plt.subplots()
        im = ax.pcolormesh(x, y, hist.T, cmap='Blues')
        fig.colorbar(im, ax=ax)

        # Add the inferred mean as a separate point
        # mean = np.mean(particles, axis=0)
        # ax.scatter(mean[0], mean[1], color='red', marker='o', s=100)

        ax.set_xlabel('time')
        ax.set_ylabel('state')
        plt.show()


    def plot_2D_hist(self, x0data, x1data):

        x0lower, x0mid, x0upper = x0data
        x1lower, x1mid, x1upper = x1data

        x0_unr = []
        x1_unr = []
        
        x0s = []
        threeDtimes = []
        for num in self.pf.particle_set.keys():
            particle_states = self.pf.particle_set[num]['X']
            
            state_traj = [state[0][0] for state in particle_states]
            
            x0s.append(state_traj)
            threeDtimes.append(list(self.pf.obs_times))

            for state in particle_states:
                x0 = state[0][0]
                x0_unr.append(x0)
                
                
                x1 = state[1][0]
                x1_unr.append(x1)

        observation_data = self.pf.data_set_y_noisy_obs

        times = self.pf.obs_times
        unr_times = list(times)*self.pf.Np

        fig, ax = plt.subplots()

        # Loop through each particle and create a line
        for i in range(len(x0s)):
            # Create a line for the current particle
            line, = ax.plot(threeDtimes[i], x0s[i], '-', linewidth=1, alpha=0.05, color='black')

        # Create a 2D histogram of the lines
        # hist, xedges, yedges = np.histogram2d(np.array(x0s).flatten(), np.array(threeDtimes).flatten(), bins=50)

        # Convert the histogram to a density and transpose it
        # density = hist / np.max(hist)
        # density = density.T

        # Plot the density as an image
        # im = ax.imshow(density, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='gray_r', alpha=0.5)

        # Add a colorbar
        # fig.colorbar(im, ax=ax)

        # Set the title and axis labels
        ax.set_title('Particle Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Show the final figure
        plt.show()
        

        # Create a 2D histogram of the x and y values for all timesteps
        # hist, xedges, yedges = np.histogram2d(unr_times, x0_unr, bins=50)
        # # Convert the histogram to a density and transpose it
        # density = hist / np.sum(hist)
        # density = density.T
        # # Create a figure and an axis
        # fig, ax = plt.subplots()
        # # Plot the density as an image
        # im = ax.imshow(density, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        # # Add a colorbar
        # fig.colorbar(im, ax=ax)
        # # Set the title and axis labels
        # ax.set_title('Particle Trajectory')
        # ax.set_xlabel('time')
        # ax.set_ylabel('x0')
        # # Show the final figure
        # plt.show()






        # fig2, ax2 = plt.subplots()
        #
        # ax2.hist2d(unr_times, x0_unr)

        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, self.pf.true_x_evo[0][:], zorder=1, linestyle='--', color='r', alpha=0.7,
        #          label='True X0 State')  # HUHJKHK
        #
        # ax2.plot(times, x0mid, linestyle='--', label='PF Mean')
        # ax2.scatter(times, observation_data, marker='x', s=4, color='k', label='Noisy State Observations')
        # ax2.fill_between(times, x0upper, x0lower, alpha=0.4, label='PF +-99% CI')
        # ax2.legend()
        # fig2.suptitle('X0 Evolution - Particle Filter - Updated - kv = {}. Np = {}'.format(self.kv, self.pf.Np))
        # plt.xlabel('Time')
        # plt.ylabel('X0')
        # plt.show()
        

    def compute_CIs(self, means, covs, stdevs, state):

        mid = []
        lower = []
        upper = []
        
        if state == 'x0':
            idx = 0
        elif state == 'x1':
            idx = 1
        elif state == 'x2':
            idx = 2
        else:
            raise ValueError('Please Enter a Valid State: x0 or x1 as argument to function call')
        
        for mean, cov in zip(means, covs):
            x_mean = mean[idx][0]
            x_stdev = np.sqrt(cov[idx][idx])

            mid.append(x_mean)
            lower.append(x_mean - stdevs*x_stdev)
            upper.append(x_mean + stdevs*x_stdev)
        return lower, upper, mid


    def plot_kalman_results(self, x0data, x1data, x2data, beta):

        lower_band, upper_band, kalman_mean_line, x_evolution, noisy_obs = x0data
        x1_lower_band, x1_upper_band, kalman_x1_mean, x_evolution = x1data
        x2_lower_band, x2_upper_band, kalman_x2_mean = x2data

        obs_times = self.obs_times

        # fig1, ax1 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax1.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--')
        # fig1.suptitle('X0 True Evolution. kv = {}'.format(self.kv))
        # plt.xlabel('Time')
        # plt.ylabel('X0')
        # plt.show()

        fig2, ax2 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        ax2.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State')
        ax2.plot(obs_times, kalman_mean_line, linestyle='--', label='Kalman Mean')
        ax2.scatter(obs_times, noisy_obs, marker='x', s=4, color='k', label='Noisy State Observations')
        ax2.fill_between(obs_times, upper_band, lower_band, alpha=0.4, label='Kalman +-99% CI')
        ax2.legend()
        fig2.suptitle('X0 Evolution - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig3, ax3 = plt.subplots()
        ax3.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X1 State')
        ax3.plot(obs_times, kalman_x1_mean, linestyle='--', label='Kalman Mean')
        ax3.fill_between(obs_times, x1_upper_band, x1_lower_band, alpha=0.4, label='Kalman +-99% CI')
        ax3.legend()
        fig3.suptitle('X1 Evolution - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig4, ax4 = plt.subplots()
        ax4.plot(obs_times, x_evolution[2][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X1 State')
        ax4.plot(obs_times, kalman_x2_mean, linestyle='--', label='Kalman Mean')
        ax4.fill_between(obs_times, x2_upper_band, x2_lower_band, alpha=0.4, label='Kalman +-99% CI')
        ax4.legend()
        fig4.suptitle('X2 (SKEW - const. for now) - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X2')
        plt.show()
        

    def extract_confidence_bands(self, stdevs, state):
        """Extract confidence bands. Given the a and C vec and mat for KFilter, at time t, extract a1 and C00.
        Plot a1 ie the mean (expected value) and +- 2 sigma (95%) Hopefully our predictions capture the observations in the bands"""
        kalman_mean = self.kalman_mean
        kalman_cov = self.kalman_cov

        if state == 'x0':
            i = 0
            
        if state == 'x1':
            i = 1

        if state == 'x2':
            i = 2

        mean = kalman_mean[i][0]
        var = np.sqrt(kalman_cov[i][i])
        upper = mean + stdevs*var
        lower = mean - stdevs*var

        return upper, lower, mean


    def quad_plot(self, x0lower, x0upper, x0mean, x0true, x0noisyObs):

        titles = [['X0 - True Evolution', 'X0 - True Evolution & Kalman Preds'],
                  ['X1 - True Evolution', 'X1 - True Evolution & Kalman Preds']]

        arr = [[0, 1],
               [2, 3]]
        obs_times = self.obs_times
        l = 0

        fig, axs = plt.subplots(2, 2, num=1, figsize=(72, 72))

        for j, k in enumerate(axs):
            for column, ax in enumerate(k):

                if l == 0:
                    x0true = x0true[0][:]
                    ax.plot(obs_times, x0true, label='True X0', linestyle='-')
                    ax.scatter(obs_times, x0true,  color='r', s=4, zorder=2)
                    ax.legend(prop={'size': 30})
                    ax.set_title(titles[j][column], fontsize=60)
                    ax.tick_params(axis='both', which='major', labelsize=35)
                    ax.yaxis.offsetText.set_fontsize(35)

                if l == 1:
                    ax.plot(obs_times, x0true, label='True X0', linestyle='-')
                    ax.plot(obs_times, x0upper, label='Kalman +99% CI', linestyle='-.')
                    ax.plot(obs_times, x0lower, label='Kalman -99% CI', linestyle='-.')
                    ax.plot(obs_times, x0mean, label='Kalman Mean X0')
                    ax.scatter(obs_times, x0noisyObs, label='Noisy X0 Observations', color='r', s=4)

                    ax.legend(prop={'size': 30})
                    ax.set_title(titles[j][column], fontsize=60)
                    ax.tick_params(axis='both', which='major', labelsize=35)
                    ax.yaxis.offsetText.set_fontsize(35)

                if l == 2:
                    ax.plot(0,0)

                if l == 3:
                    ax.plot(0,0)


                l+=1

        fig.text(0.5, 0.05, 'Time', ha='center', va='center', fontsize='72')
        fig.text(0.05, 0.5, 'State Value', ha='center', va='center', fontsize='72', rotation='vertical')
        fig.text(0.5, 0.93, 'State Space Simulations: (Insert param values)', ha='center', va='center', fontsize='72')

        fig.savefig('KFilter.png')
        return fig

        # fig.show()
        # plt.show()



def plotter(x, y, title, x_lbl, y_lbl):
    plt.figure(99)
    plt.step(x,y)
    plt.title(title)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()
    return plt



def load_finance_data():
    finance_df = pd.read_csv(
                  '/Users/zactiller/Documents/IIB_MEng_Project/Finance_Data/CHFJPY-2022-04.csv',
        skiprows=2)
    finance_df.columns = ['Pair', 'Date - Time', 'Bid', 'Ask']
    finance_df = finance_df.drop_duplicates(subset=['Date - Time'], keep="first")

    # finance_df = finance_df[35000:35500][::3] FOR GBPJPY-2023-01
    finance_df = finance_df[:300] # FOR CHFJPY

    finance_df['Date - Time'] = pd.to_datetime(finance_df['Date - Time'], format='%Y%m%d %H:%M:%S.%f')
    print(finance_df)

    return finance_df

def return_data_and_time_series(finance_data):
    "Returns time series starting at 0 and going to t (eg 115 seconds)"

    time_series = finance_data['Date - Time'].diff().fillna(pd.Timedelta(seconds=0)).cumsum().dt.total_seconds().values
    # data_series = (finance_data['Bid'].values + finance_data['Ask'].values)/2

    data_series = finance_data['Bid'].values # for CHF/JPY

    return time_series, data_series


"""
Now I want to pass our state vector Xt via some system.
dXt = A Xt dt + h dZt
We can define A and h to be state transition matrices 

We get Xt = exp(At) Xo + stoc.int(f_t)

2.2.9 states the stoc.int collapses to a sum, where the int is the effect of passing each of the jumps dZi
through the system at time Vi for Vi <= t

to Forward simulate (for the Variance Gamma):
sample from the integral  - because I(ft) is a guass rv; sum of gauss rvs = gauss rv
mean vector is given by: 3.1.2
Cov is given by : 3.1.3

The key eqns 



"""


if __name__ == '__main__':

    '''
    
    Process Simulations
    
    '''

    '''
    Generate Gamma Distr Object & Set The Conditions For The Process
    '''
    gamma_obj = GammaDistr(alpha=1, beta=0.1)
    gamma_obj.set_process_conditions(t0=0, T=1, END = None, sample_size=300) # sample_size refers to number of random times we generate

    '''
    Create a Simulation Object With the Parameters From the Existing Gamma Distribution Object
    Run The Simulation on This Object & Gather the Path, Times and set of Jumps and Times together
    '''
    gamma_sim = DistributionSimulator(gamma_obj) # create our simulation
    # path, time, jump_time_set = gamma_sim.process_simulation() #gamma_obj) # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
    # plt = plotter(time, path, 'Gamma Process Simulation TEST', 'Time', 'Value')

    '''
    Now Call .process_endpoint_sampler to Run The Sim 10,000 Times For Our Gamma Object (which defines the parameters of the gamma distr)
    & Plot Histogram of Endpoints. NOTE - in streamlit, we will want the sliders to correspond to changing the params of gamma_obj, and re-making the gamma obj (and gamma sim?)
    '''
    # fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(10_000, gamma_obj))


    '''
    Define a Normal Distr Object To Be Used to Create a Normal Gamma Simulation Object
    '''
    normal_obj = NormalDistr(mean=2.5, std=np.sqrt(1), secondary_distr=gamma_obj) # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use
    #re-write a new gamma_obj to use?
    #gamma_obj_2 = GammaDistr(alpha=0.5, beta=0.5)
    normal_gamma_sim = DistributionSimulator(normal_obj)

    '''
    Run The Process Simulation, Using The Normal Obj & The Sorted Process Set of The Desired Gamma Simulation Object to Sample From
    '''
    # path, time = normal_gamma_sim.process_simulation() #normal_obj) #, gamma_sim.sorted_process_set)
    # plotter(time, path, 'Normal Gamma Process Sim TEST', 'Time', 'Value')

    '''
    Now Call .process_endpoint_sampler With The Desired Normal Object, Gamma Object & Gamma Simulation Object to Sample 
    Process Endpoints & Plot
    '''
    # fig, ax = normal_gamma_sim.plot_simulation_distribution(normal_gamma_sim.process_endpoint_sampler(10_000, normal_obj)) #, G_obj=gamma_obj, G_sim_obj=gamma_sim))


    '''
    State Space Simulations
    
    -> Define an A matrix: [ [a1, a2], [a3, a4] ]
    -> Define a H matrix: [h1, h2]
    
    '''

    SS_gamma_obj = GammaDistr(alpha=1, beta=1)
    SS_gamma_obj.set_process_conditions(t0=0, T=1, END=None,
                                     sample_size=500)  # sample_size refers to number of random times we generate
    SS_gamma_sim = DistributionSimulator(SS_gamma_obj)
    path, time, jump_time_set = SS_gamma_sim.process_simulation()

    SS_normal_obj = NormalDistr(mean=0, std=np.sqrt(1),
                              secondary_distr=SS_gamma_obj)  # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use
    # SS_normal_gamma_sim = DistributionSimulator(SS_normal_obj)
    # path, time = SS_normal_gamma_sim.process_simulation()

    state_simulation = StateSpaceSimulation(SS_gamma_sim, num_obs=100, t0=0.0, T=10.0) # Create the State Space class with our Gamma Sim object; gamma_sim = DistributionSimulator(gamma_obj)

    state_simulation.define_model_h([0,1])
    theta = 0.0
    # state_simulation.define_A_matrix([0,1,0,theta])
    state_simulation.define_A_matrix([0,1,0,theta])
    state_simulation.set_NG_conditions(SS_normal_obj.mean, (SS_normal_obj.std)**2)

    # state_simulation.run_state_space_simulation(Mat=True)



    #TODO: Method 2, 0 skew, high sample size = smooth(ish) state evolution path
    # state_simulation.run_SSS_method_two(10, Mat=True)


    """SPARSE GAMMA PROCESS - TESTING STATE SPACE SIMULATIONS"""

    sparse_gamma_obj = GammaDistr(alpha=1, beta=1)
    sparse_gamma_obj.set_process_conditions(t0=0, T=10, END=None, sample_size=450)
    sparse_gamma_sim = DistributionSimulator(sparse_gamma_obj)

    sg_path, sg_time, sg_jump_time_set = sparse_gamma_sim.process_simulation()
    # plt = plotter(sg_time, sg_path, 'SPARSE GAMMA SIM', 'Time', 'Value')

    # -------------
    skew = 1
    # -------------

    sparse_normal_obj = NormalDistr(mean=skew, std=np.sqrt(1), secondary_distr=sparse_gamma_obj)  # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use
    sparse_normal_gamma_sim = DistributionSimulator(sparse_normal_obj)
    sng_path, sng_time = sparse_normal_gamma_sim.process_simulation()
    # plt = plotter(sng_time, sng_path, 'SPARSE NG SIM', 'Time', 'Value')


    """ Lets Define Our State Space Object First. Gives Us Option to Run a state space sim (without any filters)"""

    state_simulation = StateSpaceSimulator2(sparse_normal_gamma_sim, t0=0.0, T=10.0, num_obs=200, MatA=True)
    # ^^ This T does not really control much... end time is approx num_obs * 0.1 (I have set the expon rate to be 0.1 when generating random obs times)
    theta = -1
    state_simulation.define_A([0,1,0,theta])
    state_simulation.define_h([0, 1])

    # state_simulation.forward_simulate() IGNORE?
    "dX0 = x^. dt"
    "dX1 = theta x^. dt + dZt" #ie if theta -> 0, x1 varies more like our jump process

    # skew = 'dynamic'
    skew = 1
    beta = 0.1
    obs_times, x_evolution, ng_path = state_simulation.forward_simulate_step_by_step_gauss(skew=skew, beta=beta, var=1, state_dim=3)
    #                                                                                                                   # Insight: our jump process (n) depends on theta - because ftVi = f(A) = f(theta)
    state_simulation.show_plots(obs_times, x_evolution, ng_path)
    #TODO: Maybe show the latent gamma path of this process

    matchTest = GammaDistr(alpha=1, beta=beta)
    matchTest.set_process_conditions(t0=0, T=obs_times[-1], END=None, sample_size=450)
    matchTest = DistributionSimulator(matchTest)
    mTest_path, mTest_time, mTest_jump_time_set = matchTest.process_simulation()

    fig, ax = plt.subplots()
    ax.step(mTest_time, mTest_path, zorder=1)
    fig.suptitle('match test Gamma Path')
    plt.xlabel('Time')
    plt.ylabel('Gamma Path')
    plt.show()

    path = [0]
    for i in range(len(obs_times)-1):
        matchTest = GammaDistr(alpha=1, beta=beta)
        matchTest.set_process_conditions(t0=obs_times[i], T=obs_times[i+1], END=obs_times[-1], sample_size=450, TEST=True)
        matchTest = DistributionSimulator(matchTest)
        mTest_path, mTest_time, mTest_jump_time_set = matchTest.process_simulation()

        if len(mTest_path) == 0:
            mTest_path = [0]

        path.append(mTest_path[-1])

    fig, ax = plt.subplots()
    ax.step(obs_times, np.cumsum(path), zorder=1)
    fig.suptitle('match test Gamma Path - step by step')
    plt.xlabel('Time')
    plt.ylabel('Gamma Path')
    plt.show()



    """ Use our state_simulation object - an instance of StateSpaceSimulator2 class - in the FilterProcess Class """
    Filter = FilterProcess(state_simulation, kv=0.002, k_mu_BM=0)
    # investigate 'known jumps' ? see whether i do the implementation differently?

    # time_series, obs_data_series = return_data_and_time_series(load_finance_data())

    # when we run a KF to 'pre-gen' our data; make sure we set the beta's in the PF_KalmanFunctionsAdapted to the same (ie. we have done the 'inference' but we know for 100% certainty)
    beta = 1
    Filter.runKalmanFilter(skew=0.5, beta = beta, dynamic_skew={'Dynamic': False, 'var': 1}) # RUNNING THE KF CREATES A NEW State Space Sim w/ new data - make mu dynamic here !

    """ FilterProcess also has a particle filter option""" # pre-gen; use the states from the runKalmanFilter test
    # data = [time_series, obs_data_series]
    data = 'pre-gen'
    beta = 1
    Filter.runParticleFilter(Np=50, beta=beta, k_cov_init=0.2, X0_INIT = 0,
                             state_obs_and_times=data, show_particle_paths=True, dynamic_skew={'Dynamic': False})
    # note - i am scaling our estimated variance by hand within the code

    # ATTENTION: Initial Covariance ! initialise_particle_state_and_ktx function set to 0.7, as is case in Base Filter class
    # AND also in the big particle initialisation dictionary !!!!!!!!!



    """
    check E is beign calculated correctly, and kt_x correctly 
    
    """

