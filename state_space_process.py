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

from processes import *

class StateSpaceSimulator2:

    def __init__(self, t0, T, num_obs, MatA):
        # self.distr_obj = DistrSimObject.distr_obj
        # self.distr_sim_obj = DistrSimObject

        self.rng = np.random.default_rng(11)
        # self.rng = self.distr_obj.rng

        # self.sorted_obs_times = DistrSimObject.time_arr  # takes the initial observation times from the initial distrsimObj, which is a simulation based on our initial gamma object

        self.SS_obs_rate = 1.0 / (T - t0)
        self.sorted_obs_times = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)

        self.epochs = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)

        self.random_observation_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

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

        mean = np.zeros((2, 1))
        cov = np.zeros((2, 2))
        for jump_time in jump_time_set:
            jump = jump_time[0]
            time = jump_time[1]

            if start_time < time <= end_time:

                ftVi = self.calculate_ftVi(jump_time, end_time)

                mean += ftVi * jump
                cov += ftVi @ ftVi.T * jump

        return mean, cov


    def calc_mean_vec_jj(self, jsizes, jtimes, end_time, theta):

        vec2 = np.exp(theta * (end_time - jtimes))
        vec1 = (vec2 - 1.) / theta
        return np.sum(np.array([vec1 * jsizes,
                                vec2 * jsizes]), axis=1).reshape(-1, 1)

    def sample_gaussian(self, mean, cov):
        try:
            cov_chol = np.linalg.cholesky(cov)
            n = cov_chol @ np.column_stack([self.rng.normal(size=2)]) + np.column_stack([mean])
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            n = np.zeros((2, 1)) + np.column_stack([mean])
            print('Chol Truncated - adding mean here do we need to do this? May need to do this in particle/kalman filters too')
        return n

    def forward_simulate_step_by_step_gauss(self, skew, beta, var, state_dim):
        dyn_skew = False

        # initialise the skew
        if skew == 'dynamic':
            dyn_skew = True
            # skew = self.rng.normal(loc=0, scale=1)
            skew = 0
            skew_var = 0.001

        if self.MatA and state_dim != 3:
            x_evolution = [[0], [0]]
        elif self.MatA and state_dim == 3:
            x_evolution = [[0], [0], [skew if skew != 'dynamic' else 1]]  # CONSTANT SKEW
            print('CONSTANT SKEW (NOTE FOR FUTURE - VARY W TIME)')
        else:
            x_evolution = [0]

        obs_times = self.random_observation_times

        ng_jumps = [0]
        latent_gamma_jumps = []
        latent_gamma_times = []

        latent_gamma_jump_time_set = []

        for i in range(len(obs_times) - 1):

            start_time = obs_times[i]
            end_time = obs_times[i + 1]

            if dyn_skew:
                # print(
                #     'ATTENTION: now this is fixing skew to be 1 and then drop to 0. change to be BM later - uncomment above code')
                # if start_time < 5.0:
                #     skew = 1
                # else:
                #     skew = 0
                print('ATTENTION 2: Find value of skew_var to pick')
                skew = skew + np.sqrt(skew_var * (end_time - start_time)) * self.rng.normal(loc=0, scale=1)

            step_gamma_obj = GammaDistr(alpha=1, beta=beta)
            step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None,sample_size=450)
            step_gamma_sim = DistributionSimulator(step_gamma_obj)
            step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

            latent_gamma_jump_time_set = latent_gamma_jump_time_set + step_gamma_jump_time_set

            # latent_gamma_path.extend(step_gamma_paths)
            # latent_gamma_times.extend(step_gamma_time)

            # latent_gamma_path.append(step_gamma_jumps[0])

            # TODO: CHECK I AM APPENDING AND PLOTTING THE RIGHT 'STEP GAMMA PATH' AND ALSO CREATE A LIST OF THE JUMP_TIME_SETS

            mean, cov = self.calculate_jumps_raw_mean_and_cov(step_gamma_jump_time_set, start_time, end_time)

            # mean_vec = self.calc_mean_vec(jsizes=step_gamma_jumps, jtimes=step_gamma_time)
            # cov_mat =


            print('start = {}, skew = {}'.format(start_time, skew))
            n = self.sample_gaussian(skew * mean,
                                     var * cov)  # n is essentially our 'normal gamma jump'... therefore as a check, lets store teh jumps and cumsum them
            ng_jumps.append(n[1][0])

            # generate gamma process on this interval. store the jumps as a variable
            # calculate mean and cov of these jumps
            # sample n from our weighted gaussian
            # update state vector !

            # x_evolution.append(self.update_state_vec(n, start_time, end_time))
            new_state = self.update_state_vec(n, start_time, end_time)
            x_evolution = self.record_state_evolution(x_evolution, new_state, skew=skew)

        ng_path = np.cumsum(ng_jumps)



        fig, ax = plt.subplots()
        ax.step(list(map(lambda x: x[1], latent_gamma_jump_time_set)), np.cumsum(list(map(lambda x: x[0], latent_gamma_jump_time_set))), zorder=1)
        fig.suptitle('Latent Gamma of SS Sim - alpha = 1, beta = {}'.format(beta))
        plt.xlabel('Time')
        plt.ylabel('Gamma Path')
        plt.show()

        # self.forward_sim_latent_jt_set = latent_gamma_jump_time_set

        self.show_plots(obs_times, x_evolution, beta, ng_path)

        return obs_times, x_evolution, ng_path



    def show_plots(self, obs_times, x_evolution, beta, ng_path):
        fig1, ax1 = plt.subplots()
        ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        ax1.plot(obs_times, x_evolution[0][:], zorder=1, linestyle='--')
        fig1.suptitle('X0 Evolution - step-by-step. A = {}, alpha = 1, beta = {}'.format(self.A,
                                                                                          beta))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.scatter(obs_times, x_evolution[1][:], color='r', s=4, zorder=2)
        ax2.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--')
        fig2.suptitle('X1 Evolution - step-by-step. A = {}, alpha = 1, beta = {}'.format(self.A,
                                                                                          beta))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig3, ax3 = plt.subplots()
        ax3.scatter(obs_times, x_evolution[2][:], color='r', s=4, zorder=2)
        ax3.plot(obs_times, x_evolution[2][:], zorder=1, linestyle='--')
        fig3.suptitle(
            'X2 Evolution (JUST ON A RUN OF A SSS - MEAN IS KNOWN/SELECTED BEFORE-HAND). A = 1, alpha = {}, beta = {}'.format(
                self.A,
                beta))
        plt.xlabel('Time')
        plt.ylabel('X2 (SKEW)')
        plt.show()


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

        for i in range(len(obs_times) - 1):
            start_time = obs_times[i]
            end_time = obs_times[i + 1]
            n = 0
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
            n = expm(self.A * (end_time - jump_time_set[1])) @ self.h * jump_time_set[0]
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