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

class ExtendedStateSpaceModel:

    def __init__(self, beta, kv, sigmasq, kmu, p, initial_state, flatterned_A):

        self.rng = np.random.default_rng(150)

        # self.SS_obs_rate = 1.0 / (T - t0)
        # self.sorted_obs_times = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)
        #
        # self.epochs = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)
        #
        # self.random_observation_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

        self.X = np.array([ [initial_state[0]],
                            [initial_state[1]],
                            [initial_state[2]] ])
        self.jump_times = []
        self.jump_sizes = []

        # self.theta = theta
        self.beta = beta
        self.kv = kv
        self.sigmasq = sigmasq
        self.kmu = kmu
        self.p = p

        # predefining model matrix structure
        self.langevin_A = np.zeros((2, 2))
        self.B = np.eye(3)
        self.h = np.array([[1, 0, 0]]) # observation matrix

        # define necessary closed-form skew matrices
        self.M = None
        self.P = None
        self.Lambda = None
        self.L = None
        self.e_state_cov = None

        # storing real state evolutions and noisy_obsevrations
        self.noisy_obs = [np.array([0])]
        self.x0 = [self.X[0][0]]
        self.x1 = [self.X[1][0]]
        self.x2 = [self.X[2][0]]

        self.state_sequence = [self.X]

        self.define_langevin_A(flatterned_A)

        self.theta = self.langevin_A[1][1]

    def define_langevin_A(self, flatterned_A):
        self.langevin_A[0][0] = flatterned_A[0]
        self.langevin_A[0][1] = flatterned_A[1]
        self.langevin_A[1][0] = flatterned_A[2]
        self.langevin_A[1][1] = flatterned_A[3]

    def compute_caligA(self, end_time, start_time):
        caligA = np.block(
            [[expm(self.langevin_A * (end_time - start_time)), self.M @ np.ones(((np.shape(self.M))[1], 1))],
             [np.zeros((1, 2)), 1.]])
        return caligA

    def produce_I_matrices(self, start_time, end_time, jump_time_set):

        if len(jump_time_set) == 0:
            jump_time_set = [(0, start_time), (0, end_time)]
        # self.calculate_M_and_P_matrix(jump_time_set, end_time)
        self.calculate_P_and_M_matrices_2(jump_time_set, end_time)

        self.produce_skew_matrices(start_time, end_time, jump_time_set)

    def calculate_P_and_M_matrices_2(self, jump_time_set, end_time):

        jumps, times = np.array(jump_time_set).T
        times = times + [end_time]
        h = np.array([[0], [1]])
        # sigma_w = self.var
        sigma_w = 1  # marginalised form?

        M = np.array([[], []])
        P = np.array([[], []])

        jt_set = jump_time_set + [(0, end_time)]
        for jump_time_tuple in jt_set:
            jump = jump_time_tuple[0]
            time = jump_time_tuple[1]
            expA_times_h = expm(self.langevin_A * (end_time - time)) @ h

            M_entry = expA_times_h * jump

            M = np.append(M, M_entry, axis=1)

            P_entry = expA_times_h * np.sqrt(jump * (self.sigmasq ** 2))

            P = np.append(P, P_entry, axis=1)
        self.M = M
        self.P = P

    def produce_skew_matrices(self, start_time, end_time, jump_time_set):

        jtimes = [start_time] + [pair[1] for pair in jump_time_set] + [end_time]
        tdiffs = np.diff(jtimes)

        self.calculate_L_matrix(tdiffs)
        self.calculate_Lambda_matrix(tdiffs)

    def calculate_L_matrix(self, tdiffs):
        self.L = np.tri(len(tdiffs), len(tdiffs))

    def calculate_Lambda_matrix(self, tdiffs):
        # self.Lambda = self.sigma_mu_sq * np.diag(tdiffs)

        sigma_mu_sq = self.kmu * self.sigmasq  # the 1 should be self.var however we say var = 1 as we marginalised it
        self.Lambda = sigma_mu_sq * np.diag(tdiffs)

    def compute_noise_vector_dynamic_skew(self):
        # alpha_t = A alpha_s + B e_state

        # noise vector = e_state ~ N(0, S)
        # S = C1 + C2 : C1 = k_mu_BM * sigma_w_sq * [ [  ] [   ] [  ] ] , C2 = sigma_w_sq * [ [P I P^T 0] [0 0 0] ]
        # S = sigma_w_sq * [ C1 + C2 ]

        # 1/ compute C1
        L_lambda_L = self.L @ self.Lambda @ self.L.T
        ML_lambda_L = self.M @ L_lambda_L
        L_lambda_LM = L_lambda_L @ self.M.T

        # NOTE - got rid of additional k_mu scaling here
        C1 = np.block([[ML_lambda_L @ self.M.T, ML_lambda_L[:, [-1]]],
                                      [L_lambda_LM[-1, :], L_lambda_L[-1, -1]]])
        # 2/ compute C2
        C2 = np.block([[self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T, np.zeros((2, 1))], [np.zeros((1, 3))]])

        # 3/ add them and times by sigma_w_s1 !

        # var = self.var
        var = 1
        S = var * (C1 + C2)

        self.e_state_cov = S

        # e = np.random.multivariate_normal([0,0,0],S)
        # e = np.reshape(e, (3,1))

        if self.kmu == 0: # cholesky fails as if kmu is 0, we have a 2x2 with zero padding
            try:
                cov_chol = np.linalg.cholesky(np.block([self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T]))
                e = cov_chol @ np.column_stack([self.rng.normal(size=2)]) + np.zeros(
                    (2, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
                e = np.zeros((2, 1))
            e = np.append(e, 0).reshape(3,1)
        else:
            try:
                cov_chol = np.linalg.cholesky(S)
                e = cov_chol @ np.column_stack([self.rng.normal(size=3)]) + np.zeros(
                    (3, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
                # truncate innovation to zero if the increment is too small for Cholesky decomposition
                # print('Chol Truncated.')
                e = np.zeros((3, 1))

        # e = np.random.multivariate_normal([0,0,0], S)

        return e, S

    def propagate(self, start_time, end_time):

        step_gamma_obj = GammaDistr(alpha=1, beta=self.beta)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

        self.produce_I_matrices(start_time, end_time, step_gamma_jump_time_set) #INVESTIGATE if step_gamma is a PATH or JUMPS

        caligA = self.compute_caligA(end_time, start_time)

        e, S = self.compute_noise_vector_dynamic_skew()

        self.X = caligA @ self.X + self.B @ e

        noisy_observation = self.h @ self.X + np.sqrt(self.sigmasq * self.kv) * self.rng.normal()

        self.noisy_obs.append(noisy_observation.flatten())
        self.x0.append(self.X[0][0])
        self.x1.append(self.X[1][0])
        self.x2.append(self.X[2][0])

        self.state_sequence.append(self.X)

    def simulate_state_space_model(self, num_obs):
        self.random_observation_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

        for i in range(len(self.random_observation_times)-1):

            start_time = self.random_observation_times[i]
            end_time = self.random_observation_times[i+1]

            self.propagate(start_time, end_time)

        return self.state_sequence, self.random_observation_times, self.noisy_obs

    def apply_kalman_filtering(self):
        # APPLY KALMAN FILTERING, IF I WANTED TO SHOW IN REPORT? ie if PF not work!
        pass

    def show_plots(self):
        obs_times = self.random_observation_times

        # plt.rcParams["text.usetex"] = True  # Enable LaTeX rendering
        # plt.rcParams["font.family"] = "serif"  # Set font family to serif (for LaTeX compatibility)
        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        # Adjust the font sizes
        plt.rcParams.update({
            'font.size': 14,  # Set axis label and title font size
            'xtick.labelsize': 11.5,  # Set x-axis tick label size
            'ytick.labelsize': 11.5  # Set y-axis tick label size
        })

        # Create a single figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

        # Plot 1
        ax1.scatter(obs_times, self.x0, color='r', s=4, zorder=2)
        ax1.plot(obs_times, self.x0, zorder=1, linestyle='--')
        ax1.set_ylabel('$X_0, \mathrm{Position}$')
        ax1.grid(True, linewidth=0.5)

        # Plot 2
        ax2.scatter(obs_times, self.x1, color='r', s=4, zorder=2)
        ax2.plot(obs_times, self.x1, zorder=1, linestyle='--')
        ax2.set_ylabel('$X_1, \mathrm{Velocity}$')
        ax2.grid(True, linewidth=0.5)
        # ax2.set_xlabel(r'$\mathrm{Time, t}$')


        # Plot 3
        ax3.scatter(obs_times, self.x2, color='r', s=4, zorder=2)
        ax3.plot(obs_times, self.x2, zorder=1, linestyle='--')
        ax3.set_xlabel(r'$\mathrm{Time, t}$')
        ax3.set_ylabel('$X_2, \mathrm{Skew}$')
        ax3.grid(True, linewidth=0.5)
        ax3.set_xlabel(r'$\mathrm{Time, t}$')

        # Add a shared title with parameter values
        # fig.suptitle(r'$\mathrm{Evolution} $\mathrm{of}$ $X_0$, $X_1$, $\mathrm{and}$ $X_2$'+
        #              '\n' +
        #              r'$\beta$: {:.2f}, $\kappa_\mu$: {:.2f}, $\theta$: {:.2f}'.format(
        #     self.beta, self.kmu, self.theta))

        fig.suptitle(r'$\mathrm{Evolution}$ $\mathrm{of}$ $X_0$, $X_1$ $\mathrm{and}$ $X_2$'  +
                     '\n' +
                     r'$\beta$: {:.2f}, $\kappa_\mu$: {:.4f}, $\sigma^2$: {:.2f}, $\theta$: {:.2f}'.format(self.beta, self.kmu, self.sigmasq, self.theta),
                     y=0.95)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.05)

        plt.show()

if __name__ == "__main__":
    theta = -1
    ssmodel = ExtendedStateSpaceModel(beta=1.75, kv=0.01, kmu=0.0001, sigmasq=1, p=1,
                                      initial_state=[0,0,0], flatterned_A=[0,1,0,theta])
    true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=105)
    ssmodel.show_plots()
    print('Remember to return artificial data')


# check that gmama distr object stuffs does not need state sim objects....