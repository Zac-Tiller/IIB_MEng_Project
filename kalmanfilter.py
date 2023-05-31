import numpy as np
import random
import matplotlib as mpl
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

from state_space_process import StateSpaceSimulator2
from processes import *

#
# plt.style.use('seaborn')
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False
# })

class FreeStandingKalmanFilter(StateSpaceSimulator2):
    def __init__(self, kv, k_mu_BM, theta, initial_state, t0, T, num_obs):
        # self.StateSimObject = StateSimObject  # Pass the StateSimObject JUST to have access to its methods, such as, generating a gamma process over some interval, and method to calc jumps mean and cov


        ss = StateSpaceSimulator2(t0, T, num_obs, MatA=True)
        ss.define_A(flatterned_A=[0, 1, 0, theta])
        ss.define_h(flatterned_h=[0,1])
        self.StateSimObject = ss

        self.rng = ss.rng

        # self.obs_times = StateSimObject.random_observation_times
        self.obs_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

        self.X = np.array([[initial_state[0]], [initial_state[1]], [initial_state[2]]])
        self.langevin_A = np.array([ [0, 1], [0, theta] ])
        self.kv = kv
        self.k_mu_BM = k_mu_BM
        self.var = 1  # TODO: inheritance!
        self.sigma_mu_sq = self.var  # self.k_mu_BM * self.var

        self.dynamic_skew_var = self.k_mu_BM * self.var

        self.caligA = np.zeros((3, 3))
        self.caligB = np.array([[1, 0], [0, 1], [0, 0]])
        self.caligH = np.array([[1, 0, 0]])

        # initialise kalman mean and covariance vec and matrix
        self.kalman_mean = np.zeros((3, 1))  # TODO: Change to more informative prior? run a few processes first (good to include in report...)!!!
        self.kalman_cov = 0.7 * np.eye(3)
        self.kalman_gain = np.zeros((3, 1))

        # storing the noisy obs and true sate underlying evolutions of a process run (generated by calling the Kalman filter)
        self.noisy_obs = None
        self.x_evolution = None

    def compute_caligA(self, start_time, end_time, mean_vec):
        A = self.langevin_A
        self.caligA = np.block([[expm(A * (end_time - start_time)), mean_vec],
                                [np.zeros((1, 2)), 1.]])

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
            e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=2)]) + np.zeros(
                (2, 1))  # used to have + mean here - but now zeros
        except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
            e = np.zeros((2, 1))
        return e

    def kalman_predictive_mean(self, t_start, t_end, jumps_mean):
        # update the kalman mean:
        self.kalman_mean = self.caligA @ self.kalman_mean
        return None

    def kalman_predictive_cov(self, var, jumps_cov):
        var = 1  # marginalising
        self.kalman_cov = self.caligA @ self.kalman_cov @ self.caligA.T + var * self.caligB @ jumps_cov @ self.caligB.T
        return None

    def kalman_predict(self, t_start, t_end, var, jumps_mean, jumps_cov):

        self.kalman_predictive_mean(t_start, t_end, jumps_mean)  # updates the kalman_mean attribute

        self.kalman_predictive_cov(var, jumps_cov)  # updates the kalman_cov attribute

        return None

    def compute_kalman_gain(self, var, kv):
        var = 1  # marginalising
        scaling = self.caligH @ self.kalman_cov @ self.caligH.T + var * kv
        self.kalman_gain = (self.kalman_cov @ self.caligH.T) * (scaling ** (-1))

    def kalman_update_mean(self, y):
        self.kalman_mean = self.kalman_mean + self.kalman_gain @ (y - self.caligH @ self.kalman_mean)
        return None

    def kalman_update_cov(self):
        self.kalman_cov = self.kalman_cov - self.kalman_gain @ self.caligH @ self.kalman_cov
        return None

    def kalman_update(self, y, var, kv, t_start, t_end):

        self.compute_kalman_gain(var, kv)  # uses the kalman PREDICTED cov and mean to update the kalman gain parameter
        self.kalman_update_mean(
            y)  # uses the kalman gain and PREDICTED mean and cov to UPDATE the kalman_mean attribute
        self.kalman_update_cov()  # uses the old kalman cov and gain to UPDATE the kalman_cov attribute

        return None

    def update_state_vec_3d(self, e, t_start, t_end, jumps_mean):
        # self.compute_caligA(t_start, t_end, jumps_mean)
        # we compute caligA for each fresh set of jumps; do NOT call compute caligA in the Kalman Functions; else we re-compute for no reason
        self.X = self.caligA @ self.X + self.caligB @ e
        return self.X

    def observe_state(self):
        # Kv scales the noise relative to the variance of the process !
        std = np.sqrt(self.var * self.kv)
        obs_noise = self.StateSimObject.rng.normal(loc=0, scale=std)
        obs = self.caligH @ self.X + obs_noise
        return obs[0][0]

    # --------------------------------------------------

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

            P_entry = expA_times_h * np.sqrt(jump * (sigma_w ** 2))

            P = np.append(P, P_entry, axis=1)
        self.M = M
        self.P = P

    def produce_skew_matrices(self, start_time, end_time, jump_time_set):

        jtimes = [start_time] + [pair[1] for pair in jump_time_set] + [end_time]
        tdiffs = np.diff(jtimes)

        self.calculate_D_matrix(tdiffs)
        self.calculate_Lambda_matrix(tdiffs)

    def calculate_D_matrix(self, tdiffs):
        self.D = np.tri(len(tdiffs), len(tdiffs))
        pass

    def calculate_Lambda_matrix(self, tdiffs):
        # self.Lambda = self.sigma_mu_sq * np.diag(tdiffs)

        sigma_mu_sq = self.k_mu_BM * 1  # the 1 should be self.var however we say var = 1 as we marginalised it
        self.Lambda = sigma_mu_sq * np.diag(tdiffs)

        pass

    def compute_caligA_dynamic_skew(self, end_time, start_time):
        self.caligA = np.block(
            [[expm(self.langevin_A * (end_time - start_time)), self.M @ np.ones(((np.shape(self.M))[1], 1))],
             [np.zeros((1, 2)), 1.]])

    def compute_noise_vector_dynamic_skew(self):
        # alpha_t = A alpha_s + B e_state

        # noise vector = e_state ~ N(0, S)
        # S = C1 + C2 : C1 = k_mu_BM * sigma_w_sq * [ [  ] [   ] [  ] ] , C2 = sigma_w_sq * [ [P I P^T 0] [0 0 0] ]
        # S = sigma_w_sq * [ C1 + C2 ]

        # 1/ compute C1
        D_lambda_D = self.D @ self.Lambda @ self.D.T
        MD_lambda_D = self.M @ D_lambda_D
        D_lambda_DM = D_lambda_D @ self.M.T

        # NOTE - got rid of additional k_mu scaling here
        C1 = np.block([[MD_lambda_D @ self.M.T, MD_lambda_D[:, [-1]]],
                                      [D_lambda_DM[-1, :], D_lambda_D[-1, -1]]])
        # 2/ compute C2
        C2 = np.block([[self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T, np.zeros((2, 1))], [np.zeros((1, 3))]])

        # 3/ add them and times by sigma_w_s1 !

        # var = self.var
        var = 1
        S = var * (C1 + C2)

        self.e_state_cov = S

        # e = np.random.multivariate_normal([0,0,0],S)
        # e = np.reshape(e, (3,1))

        if self.k_mu_BM == 0:
            try:
                cov_chol = np.linalg.cholesky(np.block([self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T]))
                e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=2)]) + np.zeros(
                    (2, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
                e = np.zeros((2, 1))
            e = np.append(e, 0).reshape(3,1)
        else:
            try:
                cov_chol = np.linalg.cholesky(S)
                e = cov_chol @ np.column_stack([self.StateSimObject.rng.normal(size=3)]) + np.zeros(
                    (3, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
                # truncate innovation to zero if the increment is too small for Cholesky decomposition
                # print('Chol Truncated.')
                e = np.zeros((3, 1))

        # e = np.random.multivariate_normal([0,0,0], S)

        return e

    def update_state_vec_3d_dynamic_skew(self, e):
        self.X = self.caligA @ self.X + e
        return self.X

    # --------------------------------------------------

    def runKalmanFilterDynamicClosedForm(self, skew, beta, dynamic_skew):

        self.X = np.array([[0], [0], [skew]])
        print('Set X_initial for KF as 0, 0, 0')
        obs_times = self.obs_times


        # self.X[2][0] = 0
        # skew_var_generating_process = dynamic_skew['var']
        skew_var_generating_process = self.dynamic_skew_var
        # if self.k_mu_BM == 0:
        #     raise ValueError('k_mu_BM is 0. This function deals with dynamic skew, so please make a positive integer when initialising FreeStandingKalmanFilter')

        self.caligB = np.eye(3)
        # skew_var_generating_process = the sigma_squared we set in the generating BM process:
        # mu_i+1 =Mu_i + sqrt( (V_i+1 - V_i) sigma_squared ) N(0,1)

        x_evolution = [[self.X[0][0]], [self.X[1][0]], [self.X[2][0]]]
        upper_band, lower_band, kalman_mean_line, noisy_obs = [0], [0], [0], [0]
        x1_lower_band, x1_upper_band, kalman_x1_mean = [0], [0], [0]
        x2_lower_band, x2_upper_band, kalman_x2_mean = [0], [0], [0]

        latent_gamma_jump_time_set = []

        N_mean_27 = [np.array([0])]
        N_var_27 = [np.array([0])]
        Ys = [0]

        C_norms = []
        ktxs = []
        var_mode_est = []

        'PARAMS TO CHECK WE ARE INFERRING VAR CORRECTLY, for a kalman filter'
        ktx = 0
        E = 0
        count = 0

        for i in range(len(obs_times) - 1):

            start_time = obs_times[i]
            end_time = obs_times[i + 1]

            skew = skew + np.sqrt(skew_var_generating_process * (end_time - start_time)) * self.StateSimObject.rng.normal(loc=0, scale=1)

            step_gamma_obj = GammaDistr(alpha=1, beta=beta)
            step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
            step_gamma_sim = DistributionSimulator(step_gamma_obj)
            step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

            latent_gamma_jump_time_set = latent_gamma_jump_time_set + step_gamma_jump_time_set

            self.produce_I_matrices(start_time, end_time, step_gamma_jump_time_set)
            # mean_I, cov_I = self.calculate_I_mean_cov()

            self.compute_caligA_dynamic_skew(end_time, start_time)

            e = self.compute_noise_vector_dynamic_skew()

            mean = None  # EDIT function, does not need a mean arg...
            self.kalman_predict(start_time, end_time, self.var, mean, self.e_state_cov)

            # see old structure from particle equation

            new_state = self.update_state_vec_3d_dynamic_skew(e)

            y = self.observe_state()
            Ys.append(y)
            # calc ktx
            ktx = self.caligH @ self.kalman_cov @ self.caligH.T + self.kv
            ktxs.append(ktx)
            # calc E
            E += (y - self.kalman_mean[0]) ** 2 / ktx
            count += 1

            x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

            IGMean = (1e-05 + E / 2) / (1e-05 + count / 2 - 1)
            IGMode = (1e-05 + E / 2) / (1e-05 + count / 2 + 1)

            var_mode_est.append(IGMode)


            self.kalman_update(y, self.var, self.kv, start_time, end_time)

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

        fig899, ax899 = plt.subplots()
        ax899.plot(np.arange(0,len(var_mode_est), 1) , np.array(var_mode_est).flatten())
        plt.show()

        rhod = 1e-05 + count / 2
        etad = 1e-05 + E / 2
        IGMean = (1e-05 + E / 2) / (1e-05 + count / 2 - 1)
        IGMode = (1e-05 + E / 2) / (1e-05 + count / 2 + 1)
        # pred_mean_var = self.pf.log_sumexp_util(lweights, lambda x: x, IGMean, retlog=False)
        print('KALMAN TEST pred MEAN var. = {}'.format(IGMean))
        print('KALMAN TEST pred MODE var. = {}'.format(IGMode))

        xx = np.linspace(0.1, 3, 3000)
        sig_post = -(rhod + 1) * np.log(xx) - np.divide(etad, xx)

        fig90009, ax90009 = plt.subplots()
        # ax90009.plot(axis, mixture)
        ax90009.plot(xx[:], sig_post.flatten()[
                            :])  # - self.pf.log_sumexp_util(mixture, lambda x: 1., np.zeros(mixture.shape[0]), retlog=False))
        fig90009.suptitle(r'$\mathrm{Inv\ Gamma\ Distr}$')
        ax90009.set_ylim(-300, None)
        plt.show()

        self.noisy_observations = noisy_obs
        self.x_evolution = x_evolution

        # This line readies our set-up for particle filtering, having ran a kalman filter (for the SOLE
        # PURPOSE of generating our underlying state simulation + noise-corrupted observations)

        # self.pf.update_PF_attributes(data_set=noisy_obs, true_state_paths=x_evolution)

        fig, ax = plt.subplots()
        ax.step(list(map(lambda x: x[1], latent_gamma_jump_time_set)),
                np.cumsum(list(map(lambda x: x[0], latent_gamma_jump_time_set))), zorder=1)
        fig.suptitle('Latent Gamma of our SS sim. alpha = 1, beta = {}'.format(beta))
        plt.xlabel('Time')
        plt.ylabel('Gamma Path')
        plt.show()

        x0_data = [lower_band, upper_band, kalman_mean_line, x_evolution, noisy_obs]
        x1_data = [x1_lower_band, x1_upper_band, kalman_x1_mean, x_evolution]
        x2_data = [x2_lower_band, x2_upper_band, kalman_x2_mean]
        self.plot_kalman_results(x0_data, x1_data, x2_data, beta, dynamic_skew['Dynamic'])

        # show the kalman prediction overlaid on real state trajectories, with +-99% confidence bands

        return obs_times, Ys, x_evolution


    def runKalmanFilter(self, skew, beta, dynamic_skew):

        # if dynamic_skew['Dynamic']:
        #     self.runKalmanFilterDynamicClosedForm(skew, beta, dynamic_skew)

        if dynamic_skew['Dynamic'] != False:
            raise ValueError('Dynamic Skew is set to True. Call runKalmanFilterDynamicClosedForm instead, because,'
                             'this function deals with just constant skew case')
        if self.k_mu_BM != 0:
            raise ValueError('k_mu_BM > 0. This function deals with NON-dynamic skew, so please make k_mu_BM = 0 when initialising FreeStandingKalmanFilter')

        self.X = np.array([[0], [0], [skew]])
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
            # skew_var_generating_process = the sigma_squared we set in the generating BM process:
            # mu_i+1 =Mu_i + sqrt( (V_i+1 - V_i) sigma_squared ) N(0,1)

        x_evolution = [[self.X[0][0]], [self.X[1][0]], [self.X[2][0]]]
        upper_band, lower_band, kalman_mean_line, noisy_obs = [0], [0], [0], [0]
        x1_lower_band, x1_upper_band, kalman_x1_mean = [0], [0], [0]
        x2_lower_band, x2_upper_band, kalman_x2_mean = [0], [0], [0]

        latent_gamma_jump_time_set = []

        N_mean_27 = [np.array([0])]
        N_var_27 = [np.array([0])]
        Ys = [0]

        C_norms = []
        ktxs = []

        'PARAMS TO CHECK WE ARE INFERRING VAR CORRECTLY, for a kalman filter'
        ktx = 0
        E = 0
        count = 0

        for i in range(len(obs_times) - 1):

            start_time = obs_times[i]
            end_time = obs_times[i + 1]

            # step-change in skew
            if not dynamic_skew['Dynamic']:
                if start_time < 5.0:
                    skew = self.X[2][0]
                else:
                    skew = self.X[2][0]
                    self.X[2][0] = skew
                    # print('TODO: time varying skew for KF - update the X parameter; \'update_state_vec_3d\' needs \n to be edited to update the state vec based on a BMotion. righ now I manually set the skew part of X, which in practise is wrong as we dont know skew')
            else:
                skew = skew + np.sqrt(skew_var_generating_process * (end_time - start_time)) * self.StateSimObject.rng.normal(loc=0, scale=1)

            step_gamma_obj = GammaDistr(alpha=1, beta=beta)
            step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
            step_gamma_sim = DistributionSimulator(step_gamma_obj)
            step_gamma_path, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

            latent_gamma_jump_time_set = latent_gamma_jump_time_set + step_gamma_jump_time_set

            # latent_gamma_jump_time_set.append(step_gamma_jump_time_set[0])
            # if len(step_gamma_path) == 0:
            #     step_gamma_path = [0]
            # latent_gamma_path.append(step_gamma_path[-1])  # was 0 before !

            mean, cov = self.StateSimObject.calculate_jumps_raw_mean_and_cov(step_gamma_jump_time_set, start_time,
                                                                             end_time)

            e = self.compute_noise_vector(self.var, cov)
            self.compute_caligA(start_time, end_time, mean)


            if dynamic_skew['Dynamic']:
                # this makes the KF an APPROXIMATION to skew-tracking
                indep_cov_term = (end_time - start_time) * skew_var_generating_process
                cov = np.array([[cov[0][0], cov[0][1], 0], [cov[1][0], cov[1][1], 0], [0, 0, indep_cov_term]])

                e3 = np.sqrt(skew_var_generating_process * (end_time - start_time)) * self.StateSimObject.rng.normal(
                    loc=0, scale=1)
                # e3 is the 3rd noise element; the random term of the skew
                e = np.append(e, e3)
                e = np.reshape(e, (3, 1))

            # PREDICT -> THEN PROPAGATE X  -> OBSERVE (NOISY but new X) -> UPDATE
            # NOTE - for the PF, we dont need this .update_state_vec_3d because the observations (data sequence) is given!
            self.kalman_predict(start_time, end_time, self.var, mean, cov)

            new_state = self.update_state_vec_3d(e, start_time, end_time, mean)

            # x_evolution is our True State Path Evolution
            if not dynamic_skew['Dynamic']:
                x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state, skew=skew)
            else:
                x_evolution = self.StateSimObject.record_state_evolution(x_evolution, new_state)

            # self.kalman_predict(start_time, end_time, self.var, mean, cov)

            y = self.observe_state()

            C_norms.append(np.linalg.norm(self.kalman_cov, ord='fro'))

            # calc ktx
            ktx = self.caligH @ self.kalman_cov @ self.caligH.T + self.kv
            ktxs.append(ktx)
            # calc E
            E += (y - self.kalman_mean[0]) ** 2 / ktx
            count += 1

            self.kalman_update(y, self.var, self.kv, start_time, end_time)  # TODO: Var is the normal gamma var parameter

            # normal distr p(y_t | y_1:t-1) = N(y_t ; y^_t, sigma^2 * ktx)
            N_mean_27.append(self.kalman_mean[0])
            N_var_27.append(self.var * ktx.flatten())
            Ys.append(y)

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
        rhod = 1e-05 + count / 2
        etad = 1e-05 + E / 2
        IGMean = (1e-05 + E / 2) / (1e-05 + count / 2 - 1)
        IGMode = (1e-05 + E / 2) / (1e-05 + count / 2 + 1)
        # pred_mean_var = self.pf.log_sumexp_util(lweights, lambda x: x, IGMean, retlog=False)
        print('KALMAN TEST pred MEAN var. = {}'.format(IGMean))
        print('KALMAN TEST pred MODE var. = {}'.format(IGMode))


        xx = np.linspace(0.1, 3, 3000)
        sig_post = -(rhod + 1) * np.log(xx) - np.divide(etad, xx)

        fig90009, ax90009 = plt.subplots()
        # ax90009.plot(axis, mixture)
        ax90009.plot(xx[:], sig_post.flatten()[:])  # - self.pf.log_sumexp_util(mixture, lambda x: 1., np.zeros(mixture.shape[0]), retlog=False))
        fig90009.suptitle(r'$Inv Gamma Distr$')
        plt.show()


        fig111, ax111 = plt.subplots()
        ax111.scatter(self.obs_times[:], np.array(Ys).flatten(), marker='x', s=4, color='r', label='Data Observations (Noisy)')
        ax111.scatter(self.obs_times[:], N_mean_27, marker='x', s=4, color='k', label='y_hat ie predicted y')
        ax111.fill_between(self.obs_times[:], np.array(Ys).flatten() - 3*np.sqrt(np.array(N_var_27).flatten()),
                           np.array(Ys).flatten() + 3*np.sqrt(np.array(N_var_27).flatten())
                           , alpha=0.4, label='+-99% CI')
        fig111.suptitle('Density p(y_t | y_1:t-1, J_1:t, sigma^2) and +-99% CIs')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('y value (observations)')

        fig115, ax115 = plt.subplots()
        ax115.scatter(self.obs_times[1:], C_norms, marker='x', s=4, color='r', label='Fro Norm')
        fig115.suptitle('Fro Norm of Kalman Cov Mat (pred)')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Fro Norm')

        fig115, ax115 = plt.subplots()
        ax115.scatter(self.obs_times[1:], ktxs, marker='x', s=4, color='r', label='ktx evo')
        fig115.suptitle('ktx evolution')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('ktx evolution')


        self.noisy_observations = noisy_obs
        self.x_evolution = x_evolution

        # This line readies our set-up for particle filtering, having ran a kalman filter (for the SOLE
        # PURPOSE of generating our underlying state simulation + noise-corrupted observations)

        # self.pf.update_PF_attributes(data_set=noisy_obs, true_state_paths=x_evolution)

        fig, ax = plt.subplots()
        ax.step(list(map(lambda x: x[1], latent_gamma_jump_time_set)), np.cumsum(list(map(lambda x: x[0], latent_gamma_jump_time_set))), zorder=1)
        fig.suptitle('Latent Gamma of our SS sim. alpha = 1, beta = {}'.format(beta))
        plt.xlabel('Time')
        plt.ylabel('Gamma Path')
        plt.show()

        x0_data = [lower_band, upper_band, kalman_mean_line, x_evolution, noisy_obs]
        x1_data = [x1_lower_band, x1_upper_band, kalman_x1_mean, x_evolution]
        x2_data = [x2_lower_band, x2_upper_band, kalman_x2_mean]
        self.plot_kalman_results(x0_data, x1_data, x2_data, beta, dynamic_skew['Dynamic'])

        # show the kalman prediction overlaid on real state trajectories, with +-99% confidence bands

        return obs_times, Ys, x_evolution

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
        upper = mean + stdevs * var
        lower = mean - stdevs * var

        return upper, lower, mean

    def plot_kalman_results(self, x0data, x1data, x2data, beta, dynamic_skew):
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
        if not dynamic_skew:
            fig2.suptitle('X0 Evolution - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        else:
            fig2.suptitle('X0 Evolution - KALMAN Filter - kv = {}, beta = {}. dyn skew'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        fig3, ax3 = plt.subplots()
        ax3.plot(obs_times, x_evolution[1][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X1 State')
        ax3.plot(obs_times, kalman_x1_mean, linestyle='--', label='Kalman Mean')
        ax3.fill_between(obs_times, x1_upper_band, x1_lower_band, alpha=0.4, label='Kalman +-99% CI')
        ax3.legend()
        if not dynamic_skew:
            fig3.suptitle('X1 Evolution - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        else:
            fig3.suptitle('X1 Evolution - KALMAN Filter - kv = {}, beta = {}. dyn skew'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        fig4, ax4 = plt.subplots()
        ax4.plot(obs_times, x_evolution[2][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X1 State')
        ax4.plot(obs_times, kalman_x2_mean, linestyle='--', label='Kalman Mean')
        ax4.fill_between(obs_times, x2_upper_band, x2_lower_band, alpha=0.4, label='Kalman +-99% CI')
        ax4.legend()
        if not dynamic_skew:
            fig4.suptitle('X2 Evolution - KALMAN Filter - kv = {}, beta = {}'.format(self.kv, beta))
        else:
            fig4.suptitle('X2 Evolution - KALMAN Filter - kv = {}, beta = {}. dyn skew'.format(self.kv, beta))
        plt.xlabel('Time')
        plt.ylabel('X2')
        plt.show()

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
            lower.append(x_mean - stdevs * x_stdev)
            upper.append(x_mean + stdevs * x_stdev)
        return lower, upper, mid

