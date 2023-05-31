import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import copy
from tqdm import tqdm
from sss import ExtendedStateSpaceModel
from processes import *
from tracking_data import *
from datahandler import *
import streamlit as st
from time import time


def util_logsumexp(lw, h, x, axis=0, retlog=False):
	"""Calculates the log of a sum of exponentiated values in a numerically stable way"""
	c = np.max(lw)
	broad_l = np.broadcast_to((lw-c).flatten(), x.T.shape).T
	if retlog:
		return c + np.log(np.sum(np.exp(broad_l) * h(x), axis=axis))
	else:
		return np.exp(c) * np.sum(np.exp(broad_l) * h(x), axis=axis)


class Particle(ExtendedStateSpaceModel):
    '''
    A class to represent a particle which is used in the Particle Filter
    Inherits from the State Space Model
    '''

    def __init__(self, beta, kv, sigmasq, kmu, p, kw, initial_state, flatterned_A):

        self.beta = beta
        self.kv = kv
        self.kw = kw
        self.sigmasq = sigmasq
        self.kmu = kmu
        self.p = p
        self.rho = 1e-05
        self.eta = 1e-05

        # particles have kalman params
        self.kmean_updt = np.array([ [initial_state[0]],
                            [initial_state[1]],
                            [initial_state[2]] ])
        self.kcov_updt = kw*np.array([[0.5, 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 0.8]])
        # initialise particle kalman means, and use this as random state in each particle in ExtendedSSM
        self.kmean_updt = self.kmean_updt + (self.kcov_updt @ np.random.randn(3)).reshape(-1, 1)

        self.kmean_pred = 0
        self.kcov_pred = 0

        # particles have a (log)weight, count and an E
        self.logweight = 0.
        self.count = 0.
        self.E = 0.

        # each particle is an instance of ExtendedStateSpaceModel, so, lets inherit the attributes and methods
        ExtendedStateSpaceModel.__init__(self, beta, kv, sigmasq, kmu, p, self.kmean_updt, flatterned_A)



    def kalman_predict(self, start_time, end_time):
        # generate some jumps of our latent gamma process
        step_gamma_obj = GammaDistr(alpha=1, beta=self.beta)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

        self.produce_I_matrices(start_time, end_time,
                                step_gamma_jump_time_set)  # INVESTIGATE if step_gamma is a PATH or JUMPS

        caligA = self.compute_caligA(end_time, start_time)

        e, S = self.compute_noise_vector_dynamic_skew()

        # kalman prediction
        self.kmean_pred = caligA @ self.kmean_updt
        self.kcov_pred = caligA @ self.kcov_updt @ caligA.T + self.B @ S @ self.B.T

        return self.kmean_pred, self.kcov_pred


    def calc_logweight(self, noisy_obs):
        self.count += 1
        y_pred = self.h @ self.kmean_pred
        ktx = self.h @ self.kcov_pred @ self.h.T + self.kv
        E_prev = self.E

        E_addition = np.square(noisy_obs - y_pred)/ktx

        E_curr = E_prev + E_addition
        # self.E += np.square(noisy_obs - y_pred)/ktx

        lw = -0.5*np.log(ktx) - (self.rho + (self.count/2.))*np.log(self.eta + E_curr/2) + (self.rho + ((self.count-1.)/2.))*np.log(self.eta + E_prev/2.)
        self.E = E_curr
        return lw


    def kalman_update(self, noisy_obs):

        kgain = (self.kcov_pred @ self.h.T) / ( (self.h @ self.kcov_pred @ self.h.T) + self.kv).reshape(-1,1)
        # kalman correction
        self.kmean_updt = self.kmean_pred + kgain * (noisy_obs - self.h @ self.kmean_pred)
        self.kcov_updt = self.kcov_pred - (kgain @ self.h @ self.kcov_pred)

        logweight = self.calc_logweight(noisy_obs)
        self.logweight += logweight


    def propagate_particle(self, t_start, t_end, noisy_obs):
        self.kalman_predict(t_start, t_end)

        self.kalman_update(noisy_obs)



class ParticleFilter:

    def __init__(self, beta, kv, sigmasq, kmu, p, kw, initial_state, flatterned_A, data, Np, epsilon, two_D_tracking):
        self.times = data[0]
        self.observation_data = data[1]
        self.true_x_evo = False

        self.Np = Np

        self.log_resample_limit = np.log(self.Np*epsilon)
        self.log_marginal_likelihood = 0

        self.beta = beta
        self.kv = kv
        self.kw = kw
        self.sigmasq = sigmasq
        self.kmu = kmu
        self.p = p
        self.rho = 1e-05
        self.eta = 1e-05

        # create our particles - not initial state is provided, from which we generate random initialisations of particle starting states
        self.particles = [Particle(beta, kv, sigmasq, kmu, p, kw, initial_state, flatterned_A) for _ in range(Np)]
        if two_D_tracking:
            self.particles_2 = [Particle(beta, kv, sigmasq, kmu, p, kw, initial_state, flatterned_A) for _ in range(Np)]

        self.normalise_weights()

    def normalise_weights(self):
        lweights = np.array([particle.logweight for particle in self.particles]).reshape(-1,1)
        sum_weights = util_logsumexp(lweights, lambda x : 1., np.ones(lweights.shape[0]), retlog=True)

        for particle in self.particles:
            particle.logweight = particle.logweight - sum_weights

        return sum_weights

    def calculate_ESS(self, method):
        # lweights = an array of all the particles log weights
        lweights = np.array([particle.logweight for particle in self.particles])

        # these 2 method all are in the log weight domain
        if method == 'd_inf':
            log_ESS = -np.max(lweights)
        elif method == 'p2':
            log_ESS = -1 * self.log_sumexp_util(lw=2 * lweights, h=lambda x: 1., x=np.ones(lweights.shape[0]),
                                                retlog=True)
        else:
            raise ValueError(
                'Invalid ESS Method Provided: Please provide \'d_inf\' or \'p2\' in calculate_ESS(.)')

        return log_ESS


    def resample_particles(self):

        lweights = np.array([particle.logweight for particle in self.particles]).flatten()

        weights = np.exp(lweights)
        probs = np.nan_to_num(weights)
        probs = probs / np.sum(probs)

        selections = np.random.multinomial(self.Np, probs)
        new_particles = []
        for idx in range(self.Np):
            for _ in range(selections[idx]):
                new_particles.append(copy.copy(self.particles[idx]))

        self.particles = new_particles

        for particle in self.particles:
            particle.logweight = -np.log(self.Np)


    def compute_state_posterior(self):
        lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
        means = np.array([particle.kmean_updt for particle in self.particles])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_updt + (particle.kmean_updt @ particle.kmean_updt.T) for particle in self.particles])

        csum = util_logsumexp(lweights, lambda x: x, cov_term, axis=0, retlog=False) - msum @ msum.T

        return msum, csum

    def compute_state_posterior_predictive(self):
        lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
        means = np.array([particle.kmean_pred for particle in self.particles])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_pred + (particle.kmean_pred @ particle.kmean_pred.T) for particle in self.particles])

        csum = util_logsumexp(lweights, lambda x: x, cov_term, axis=0, retlog=False) - msum @ msum.T

        return msum, csum

    def estimate_marginalised_var(self, plot_density=False):
        lweights = np.array([particle.logweight for particle in self.particles])
        # individual_particle_IG_means = (self.eta + np.array([particle.E for particle in self.particles])/2) / (self.rho + int(self.particles[0].count) - 1)
        # pred_mean_var = util_logsumexp(lweights, lambda x: x, individual_particle_IG_means, retlog=False)
        Es = np.array([particle.E for particle in self.particles])
        E = util_logsumexp(lweights, lambda x: x, Es, retlog=False)

        if plot_density:
            xx = np.linspace(0.0000000001, 2, 100000)
            alpha = self.rho + int(self.particles[0].count) / 2
            # gamma_alpha = gamma_func(alpha)  # part of the norm constant

            Es = np.array([particle.E for particle in self.particles])
            E = util_logsumexp(lweights, lambda x: x, Es, retlog=False)

            B = self.eta + E / 2
            # beta_to_alpha = (B) ** (alpha)  # part of the norm constant
            inv_gamma_pdf = []
            for x in xx:
                inv_gamma_pdf.append(-(alpha + 1) * np.log(x) - B / x)

            pred_mean_var = (self.eta + (E / 2)) / (self.rho + int(self.particles[0].count) / 2 - 1)

            pred_mode_var = (self.eta + (E / 2)) / (self.rho + int(self.particles[0].count) / 2 + 1)

            fig999, ax999 = plt.subplots()
            ax999.plot(xx, np.array(inv_gamma_pdf).flatten(), 'r-', lw=1, alpha=0.6, label='invgamma pdf')
            fig999.suptitle('Inv Gamma Distr For Estimated Var')
            ax999.set_xlim(pred_mean_var - 3*pred_mean_var, pred_mean_var + 3*pred_mean_var)
            plt.show()

        pred_mean_var = (self.eta + (E/2)) / (self.rho + int(self.particles[0].count)/2 - 1)

        pred_mode_var = (self.eta + (E/2)) / (self.rho + int(self.particles[0].count)/2 + 1)

        return pred_mean_var, pred_mode_var

        # Es = np.array([particle.E for particle in self.particles])
        # count = int(self.particles[0].count)
        #
        # E = util_logsumexp(lweights, lambda x: x, Es, retlog=False)
        #
        # # rhod = self.rho + (count / 2.)
        # # etad = self.eta + (E / 2.)
        # if plot_density:
        #     xx = np.linspace(0.1, 5, 10000)
        #     alpha = self.rho + count / 2
        #     # gamma_alpha = gamma_func(alpha)  # part of the norm constant
        #     B = self.eta + E / 2
        #     # beta_to_alpha = (B) ** (alpha)  # part of the norm constant
        #     inv_gamma_pdf = []
        #     for x in xx:
        #         inv_gamma_pdf.append(-(alpha + 1) * np.log(x) - B / x)
        #
        #     fig999, ax999 = plt.subplots()
        #     ax999.plot(xx, np.array(inv_gamma_pdf).flatten(), 'r-', lw=1, alpha=0.6, label='invgamma pdf')
        #     fig999.suptitle('Inv Gamma Distr For Estimated Var')
        #     ax999.set_ylim(-300, None)
        #     plt.show()
        #
        # mode = (self.eta + E / 2.) / (self.rho + count / 2. + 1.)
        # mean = (self.eta + E / 2.) / (self.rho + count / 2. - 1.)
        #
        # return mode, mean



    def run_particle_filter(self, mse_calcs, known_real_states=False):

        plot_placeholder = st.empty()
        self.plot_initial_data_st()

        if known_real_states != False:
            self.true_x_evo = known_real_states
            self.true_x_evo = np.reshape(self.true_x_evo, (int(np.size(self.true_x_evo)/3), 3))

        init_mean, init_cov = self.compute_state_posterior()
        state_mean = [init_mean]
        state_cov = [init_cov]
        mode_var_estimates = []
        mean_var_estimates = []

        st_plot_means = []
        st_plot_covs = []
        st_plot_times = []

        MSE = []

        times=self.times
        with tqdm(total = len(times)*self.Np) as pbar:

            for t in range(1, len(times)):
                t_start = times[t-1]
                t_end = times[t]

                noisy_observation = self.observation_data[t-1]

                for particle in self.particles:
                    particle.propagate_particle(t_start, t_end, noisy_observation)
                    pbar.update(1)


                incremental_log_like = self.normalise_weights()
                self.log_marginal_likelihood += incremental_log_like
                log_ESS = self.calculate_ESS(method='d_inf')

                if log_ESS < self.log_resample_limit:
                    self.resample_particles()

                mixed_mean, mixed_cov = self.compute_state_posterior()

                if mse_calcs:
                    predictive_mean, predictive_cov = self.compute_state_posterior_predictive()
                    MSE.append( (predictive_mean[0][0] - noisy_observation)**2)



                state_mean.append(mixed_mean)
                state_cov.append(mixed_cov)

                # mode_var, mean_var = self.estimate_marginalised_var()
                pred_mean_var, pred_mode_var = self.estimate_marginalised_var()
                mean_var_estimates.append(pred_mean_var)
                mode_var_estimates.append(pred_mode_var)

                st_plot_means.append(mixed_mean)
                st_plot_covs.append(mixed_cov)
                st_plot_times = times[:t]

                self.update_st_plot(st_plot_times, st_plot_means, st_plot_covs)
                # time.sleep(0.2)

        self.plot_var_convergence(mode_var_estimates[5:], mean_var_estimates[5:])

        pred_mean_var, pred_mode_var = self.estimate_marginalised_var(plot_density=True)
        print('Estimated Var: {} MEAN'.format(pred_mean_var))
        print('Estimated Var: {} MODE'.format(pred_mode_var))

        if mse_calcs:
            print('Total Predictive MSE: {}'.format(np.sum(MSE)))

        self.display_pf_plots(state_mean, state_cov, known_real_states, pred_mode_var)


    def plot_var_convergence(self, var_estimate_list, mean_var_list):

        fig101, ax101 = plt.subplots()
        ax101.plot(np.linspace(1, len(var_estimate_list), len(var_estimate_list)), np.array(var_estimate_list).flatten(), label='Mode')
        ax101.plot(np.linspace(1, len(mean_var_list), len(mean_var_list)), np.array(mean_var_list).flatten(), label = 'Mean')

        fig101.suptitle('Var Estimate Evolution w/ timestep')
        plt.show()

    def plot_initial_data_st(self):

        plt.figure(figsize=(8, 6))

        range_start = 0
        observation_data = np.array(self.observation_data).flatten()
        times = np.array(self.times).flatten()

        plt.scatter(times[range_start:], observation_data[range_start:], marker='x', s=4, color='k',
                    label='Noisy State Observations')
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.legend()
        st.pyplot()

    def update_st_plot(self, times, means, covs):

        plt.figure(figsize=(8,6))

        pred_var = 0.00001
        var_SF = 1
        var = 1

        x0lower, x0upper, x0mid = self.compute_CIs(means=means,
                                                   covs=var_SF * var * covs, stdevs=3,
                                                   state='x0', pred_var=pred_var)

        plt.plot(times, np.array([mean[0] for mean in means]).flatten(), label='PF Mean')
        plt.fill_between(times, x0lower, x0upper, alpha=0.2, label='+- 99% CI')
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.legend()
        st.pyplot()






    def display_pf_plots(self, state_mean, state_cov, known_real_states, pred_var, var=1, var_SF=1):

        x0lower, x0upper, x0mid = self.compute_CIs(means=state_mean,
                                                   covs=pred_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x0')
        x1lower, x1upper, x1mid = self.compute_CIs(means=state_mean,
                                                   covs=pred_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x1')
        x2lower, x2upper, x2mid = self.compute_CIs(means=state_mean,
                                                   covs=pred_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x2')

        x0_pf_data = [x0lower, x0mid, x0upper]
        x1_pf_data = [x1lower, x1mid, x1upper]
        x2_pf_data = [x2lower, x2mid, x2upper]

        # --------------------------------------------------------------------------------
        range_start=0

        observation_data = np.array(self.observation_data).flatten()

        fig2, ax2 = plt.subplots()
        if known_real_states != False:
            ax2.scatter(times, self.true_x_evo[:,0], color='r', s=4, zorder=2)
            ax2.plot(times, self.true_x_evo[:,0], zorder=1, linestyle='--', color='r', alpha=0.7,
                     label='True X0 State')  # HUHJKHK

        ax2.plot(times[range_start:], x0mid[range_start:], linestyle='--', label='PF Mean')
        ax2.scatter(times[range_start:], observation_data[range_start:], marker='x', s=4, color='k',
                    label='Noisy State Observations')
        ax2.fill_between(times[range_start:], x0upper[range_start:], x0lower[range_start:], alpha=0.4,
                         label='PF +-99% CI')
        ax2.legend()
        fig2.suptitle(
            'X0 Evolution - Particle Filter - Updated - k_mu_BM = {}, kv = {}. Np = {} \n beta ={}, k_cov_init = {}'.format(self.kmu,
                                                                                                        self.kv,
                                                                                                        self.Np,
                                                                                                        self.beta,
                                                                                                        self.kw))
        plt.xlabel('Time')
        plt.ylabel('X0')
        plt.show()

        # --------------------------------------------------------------------------------

        fig3, ax3 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK

        ax3.plot(times[range_start:], x1mid[range_start:], linestyle='--', label='PF Mean')
        if known_real_states != False:
            ax3.scatter(times[:], self.true_x_evo[:,1], marker='x', s=4, color='k',
                        label='True X1 State Underlying')
            ax3.plot(times, self.true_x_evo[:,1], linestyle='--', alpha=0.7, color='r',
                     label='True X1 State Underlying')
        ax3.fill_between(times[range_start:], x1upper[range_start:], x1lower[range_start:], alpha=0.4,
                         label='PF +-99% CI')
        ax3.legend()
        fig3.suptitle(
            'X1 Evolution - Particle Filter - Updated - k_mu_BM = {}, kv = {}. Np = {}\n beta ={}, k_cov_init = {}'.format(self.kmu,
                                                                                                        self.kv,
                                                                                                        self.Np,
                                                                                                        self.beta,
                                                                                                        self.kw))
        plt.xlabel('Time')
        plt.ylabel('X1')
        plt.show()

        # --------------------------------------------------------------------------------

        fig4, ax4 = plt.subplots()
        # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK

        ax4.plot(times[range_start:], x2mid[range_start:], linestyle='--', label='PF Mean')
        # ax3.scatter(times, pf.true_x_evo[1][:], marker='x', s=4, color='k', label='True X1 State Underlying')
        if known_real_states != False:
            ax4.plot(times, self.true_x_evo[:,2], linestyle='--', alpha=0.7, color='r',
                     label='True X2 State Underlying')
        ax4.fill_between(times[range_start:], x2upper[range_start:], x2lower[range_start:], alpha=0.4,
                         label='PF +-99% CI')
        ax4.legend()
        fig4.suptitle(
            'X2 - Particle Filter - k_mu_BM = {} kv = {}. Np = {} \n beta = {}, k_cov_init = {}'.format(self.kmu,
                                                                                                        self.kv,
                                                                                                        self.Np,
                                                                                                        self.beta,
                                                                                                        self.kw))
        plt.xlabel('Time')
        plt.ylabel('X2')
        plt.show()

        pass

    def compute_CIs(self, means, covs, stdevs, state, pred_var=0):
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




def main():
    theta = -0.2778
    #     # ssmodel = ExtendedStateSpaceModel(beta=1, kv=0.01, kmu=0, sigmasq=1, p=1,
    #     #                                   initial_state=[0, 0, 0.5], flatterned_A=[0, 1, 0, theta])
    #     # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=200)
    #     # ssmodel.show_plots()
    #
        # tracking_df = load_tracking_data()
        # player_x, player_y, times = show_player_path(tracking_df, 10, start='18:30', end='18:45')

    times, noisy_data = return_data_and_time_series(load_finance_data())

    pf = ParticleFilter(beta=1, kv=2, sigmasq=1, kmu=0.0001, p=1, kw=0.01,
                        initial_state=[131.87, 0, 0], flatterned_A=[0, 1, 0, theta],
                        data=[times, noisy_data], Np=2, epsilon=0.5, two_D_tracking=False)
    pf.run_particle_filter(mse_calcs=False, known_real_states=False)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     theta = -0.2778
#     # ssmodel = ExtendedStateSpaceModel(beta=1, kv=0.01, kmu=0, sigmasq=1, p=1,
#     #                                   initial_state=[0, 0, 0.5], flatterned_A=[0, 1, 0, theta])
#     # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=200)
#     # ssmodel.show_plots()
#
#     # tracking_df = load_tracking_data()
#     # player_x, player_y, times = show_player_path(tracking_df, 10, start='18:30', end='18:45')
#
#     times, noisy_data = return_data_and_time_series(load_finance_data())
#
#     pf = ParticleFilter(beta=1, kv=2, sigmasq=1, kmu=0.0001, p=1, kw=0.01,
#                         initial_state=[131.87, 0, 0], flatterned_A=[0, 1, 0, theta],
#                         data=[times, noisy_data], Np=100, epsilon=0.5, two_D_tracking=False)
#     pf.run_particle_filter(known_real_states=False)


