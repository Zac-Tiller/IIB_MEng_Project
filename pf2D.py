import numpy as np
import random
import pandas as pd
import copy
from tqdm import tqdm
from sss import ExtendedStateSpaceModel
from processes import *
from tracking_data import *
from datahandler import *
from mplsoccer import Pitch


def util_logsumexp(lw, h, x, axis=0, retlog=False):
	c = np.max(lw)
	broad_l = np.broadcast_to((lw-c).flatten(), x.T.shape).T
	if retlog:
		return c + np.log(np.sum(np.exp(broad_l) * h(x), axis=axis))
	else:
		return np.exp(c) * np.sum(np.exp(broad_l) * h(x), axis=axis)



class Particle(ExtendedStateSpaceModel):
    """particle class - instantiatie for the x and y directions"""

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



    def kalman_predict(self, start_time, end_time, gamma_jt_set):
        # generate some jumps of our latent gamma process
        # step_gamma_obj = GammaDistr(alpha=1, beta=self.beta)
        # step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
        # step_gamma_sim = DistributionSimulator(step_gamma_obj)
        # step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

        self.produce_I_matrices(start_time, end_time,
                                gamma_jt_set)  # INVESTIGATE if step_gamma is a PATH or JUMPS

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


    def propagate_particle(self, t_start, t_end, noisy_obs, gamma_jt_set):
        self.kalman_predict(t_start, t_end, gamma_jt_set)

        self.kalman_update(noisy_obs)



class ParticleFilter2D:

    def __init__(self, beta, kv, sigmasq, kmu, p, kw, initial_state, flatterned_A, data, Np, epsilon, x_init, y_init):
        self.times = data[0]
        self.x_observation_data = data[1]
        self.y_observation_data = data[2]

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

        x_init_state = [x_init, initial_state[1], initial_state[2]]
        y_init_state = [y_init, initial_state[1], initial_state[2]]

        # create our particles - not initial state is provided, from which we generate random initialisations of particle starting states
        self.particles = [Particle(beta, kv, sigmasq, kmu, p, kw, x_init_state, flatterned_A) for _ in range(Np)]

        self.particles2 = [Particle(beta, kv, sigmasq, kmu, p, kw, y_init_state, flatterned_A) for _ in range(Np)]

        self.normalise_weights()

    def normalise_weights(self):
        x_lweights = np.array([particle.logweight for particle in self.particles]).reshape(-1,1)
        x_sum_weights = util_logsumexp(x_lweights, lambda x : 1., np.ones(x_lweights.shape[0]), retlog=True)

        y_lweights = np.array([particle.logweight for particle in self.particles2]).reshape(-1, 1)
        y_sum_weights = util_logsumexp(y_lweights, lambda x: 1., np.ones(y_lweights.shape[0]), retlog=True)

        for particle in self.particles:
            particle.logweight = particle.logweight - x_sum_weights

        for particle in self.particles2:
            particle.logweight = particle.logweight - y_sum_weights

        sum_weights = x_sum_weights * y_sum_weights

        return sum_weights

    def calculate_ESS_x(self, method):
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

    def calculate_ESS_y(self, method):
        # lweights = an array of all the particles log weights
        lweights = np.array([particle.logweight for particle in self.particles2])

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


    def resample_particles_x(self):

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

    def resample_particles_y(self):

        lweights = np.array([particle.logweight for particle in self.particles2]).flatten()

        weights = np.exp(lweights)
        probs = np.nan_to_num(weights)
        probs = probs / np.sum(probs)

        selections = np.random.multinomial(self.Np, probs)
        new_particles = []
        for idx in range(self.Np):
            for _ in range(selections[idx]):
                new_particles.append(copy.copy(self.particles2[idx]))

        self.particles2 = new_particles

        for particle in self.particles2:
            particle.logweight = -np.log(self.Np)


    def compute_state_posterior_x(self):
        lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
        means = np.array([particle.kmean_updt for particle in self.particles])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_updt + (particle.kmean_updt @ particle.kmean_updt.T) for particle in self.particles])

        csum = util_logsumexp(lweights, lambda x: x, cov_term, axis=0, retlog=False) - msum @ msum.T

        return msum, csum


    def compute_state_posterior_predictive_x(self):
        lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
        means = np.array([particle.kmean_pred for particle in self.particles])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_pred + (particle.kmean_pred @ particle.kmean_pred.T) for particle in self.particles])

        csum = util_logsumexp(lweights, lambda x: x, cov_term, axis=0, retlog=False) - msum @ msum.T

        return msum, csum

    def compute_state_posterior_predictive_y(self):
        lweights = np.array([(particle.logweight) for particle in self.particles2]).reshape(-1, 1)
        means = np.array([particle.kmean_pred for particle in self.particles2])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_pred + (particle.kmean_pred @ particle.kmean_pred.T) for particle in self.particles2])

        csum = util_logsumexp(lweights, lambda x: x, cov_term, axis=0, retlog=False) - msum @ msum.T

        return msum, csum


    def compute_state_posterior_y(self):
        lweights = np.array([(particle.logweight) for particle in self.particles2]).reshape(-1, 1)
        means = np.array([particle.kmean_updt for particle in self.particles2])

        msum = util_logsumexp(lweights, lambda x: x, means, axis=0, retlog=False)

        cov_term = np.array([particle.kcov_updt + (particle.kmean_updt @ particle.kmean_updt.T) for particle in self.particles2])

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


    def propogate_particles(self, particle_x, particle_y, t_start, t_end, x_noisy_observation, y_noisy_observation):

        # generate some jumps of our latent gamma process - same for each particle!
        step_gamma_obj = GammaDistr(alpha=1, beta=self.beta)
        step_gamma_obj.set_process_conditions(t0=t_start, T=t_end, END=None, sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

        particle_x.propagate_particle(t_start, t_end, x_noisy_observation, step_gamma_jump_time_set)
        particle_y.propagate_particle(t_start, t_end, y_noisy_observation, step_gamma_jump_time_set)


    def run_particle_filter(self, pitch_obj, known_real_states=False):

        if known_real_states != False:
            self.true_x_evo = known_real_states
            self.true_x_evo = np.reshape(self.true_x_evo, (int(np.size(self.true_x_evo)/3), 3))

        xinit_mean, xinit_cov = self.compute_state_posterior_x()
        xstate_mean = [xinit_mean]
        xstate_cov = [xinit_cov]

        yinit_mean, yinit_cov = self.compute_state_posterior_y()
        ystate_mean = [yinit_mean]
        ystate_cov = [yinit_cov]

        mode_var_estimates = []
        mean_var_estimates = []

        x_MSE = []
        y_MSE = []

        times = self.times

        with tqdm(total = len(times)*self.Np) as pbar:

            for t in range(1, len(times)):
                t_start = times[t-1].item()
                t_end = times[t].item()

                x_noisy_observation = self.x_observation_data[t-1].item()
                y_noisy_observation = self.y_observation_data[t-1].item()


                for particle_x, particle_y in zip(self.particles, self.particles2):

                    self.propogate_particles(particle_x, particle_y, t_start, t_end, x_noisy_observation, y_noisy_observation)

                    # particle.propagate_particle(t_start, t_end, noisy_observation)
                    pbar.update(1)


                incremental_log_like = self.normalise_weights()
                self.log_marginal_likelihood += incremental_log_like
                log_ESS_x = self.calculate_ESS_x(method='d_inf')

                if log_ESS_x < self.log_resample_limit:
                    self.resample_particles_x()

                log_ESS_y = self.calculate_ESS_y(method='d_inf')

                if log_ESS_y < self.log_resample_limit:
                    self.resample_particles_y()



                x_mixed_mean, x_mixed_cov = self.compute_state_posterior_x()
                y_mixed_mean, y_mixed_cov = self.compute_state_posterior_y()


                xstate_mean.append(x_mixed_mean)
                xstate_cov.append(x_mixed_cov)

                ystate_mean.append(y_mixed_mean)
                ystate_cov.append(y_mixed_cov)

                predictive_mean_x, predictive_cov_x = self.compute_state_posterior_predictive_x()
                x_MSE.append((predictive_mean_x[0][0] - x_noisy_observation) ** 2)

                predictive_mean_y, predictive_cov_y = self.compute_state_posterior_predictive_y()
                y_MSE.append((predictive_mean_y[0][0] - y_noisy_observation) ** 2)


                # mode_var, mean_var = self.estimate_marginalised_var()
                # pred_mean_var, pred_mode_var = self.estimate_marginalised_var()
                # mean_var_estimates.append(pred_mean_var)
                # mode_var_estimates.append(pred_mode_var)

        # self.plot_var_convergence(mode_var_estimates[5:], mean_var_estimates[5:])
        #
        # pred_mean_var, pred_mode_var = self.estimate_marginalised_var(plot_density=True)
        # print('Estimated Var: {} MEAN'.format(pred_mean_var))
        # print('Estimated Var: {} MODE'.format(pred_mode_var))

        pred_mode_var = 1.05

        self.display_pf_plots(xstate_mean, xstate_cov, known_real_states, pred_mode_var, dimension='x')

        self.display_pf_plots(ystate_mean, ystate_cov, known_real_states, pred_mode_var, dimension='y')

        self.plot_over_pitch(pitch_obj, ystate_mean, xstate_mean)

        print('Total Predictive (x-hat - x) MSE: {}'.format(np.sum(x_MSE) / len(x_MSE)))
        print('Total Predictive (y-hat - y) MSE: {}'.format(np.sum(y_MSE) / len(y_MSE)))


    def plot_over_pitch(self, pitch_obj, y_state_mean, x_state_mean):

        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        # Adjust the font sizes
        plt.rcParams.update({
            'font.size': 15,  # Set axis label and title font size
            'xtick.labelsize': 13,  # Set x-axis tick label size
            'ytick.labelsize': 13  # Set y-axis tick label size
        })


        # Create scatter plot with time-based color mapping
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a soccer pitch
        pitch = Pitch(pitch_type='statsbomb', line_color='white',
                      pitch_color='#E5F2E5')

        # Draw the soccer pitch
        pitch.draw(ax=ax)

        times = self.times
        colors = times / 60

        scatter = ax.scatter(
            self.x_observation_data,
            self.y_observation_data,
            c=colors.flatten(), cmap='RdYlGn_r', s=10)

        # Set color bar label
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time (Min.Sec)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        ax.plot(np.array(x_state_mean)[:, 0, 0], np.array(y_state_mean)[:, 0, 0], linestyle='--')
        cbar.ax.set_position([0.76, 0.1, 0.04, 0.8])

        # Create scatter plot with time-based color mapping
        # scatter = ax.scatter(player_x, player_y, c=colors.flatten(), cmap='RdYlGn_r', s=100)

        # # Set color bar label
        # cbar = plt.colorbar(scatter)
        # cbar.set_label('Time (Min.Sec)')

        # Set pitch boundaries and labels
        # ax.set_xlim(0, pitch_length)
        # ax.set_ylim(0, pitch_width)
        ax.set_xlabel(r'$\mathrm{x}$ $\mathrm{position}$ $\mathrm{/m}$')
        ax.set_ylabel(r'$\mathrm{y}$ $\mathrm{position}$ $\mathrm{/m}$')
        ax.set_title(r'$\mathrm{{Player\ 10\ Position\ Over\ Time}}$')

        # Show the plot
        plt.show()








        # fig, ax = pitch_obj
        #
        # ax.plot(np.array(x_state_mean)[:, 0, 0], np.array(y_state_mean)[:, 0, 0], linestyle='--')
        # plt.show()




    def plot_var_convergence(self, var_estimate_list, mean_var_list):

        fig101, ax101 = plt.subplots()
        ax101.plot(np.linspace(1, len(var_estimate_list), len(var_estimate_list)), np.array(var_estimate_list).flatten(), label='Mode')
        ax101.plot(np.linspace(1, len(mean_var_list), len(mean_var_list)), np.array(mean_var_list).flatten(), label = 'Mean')

        fig101.suptitle('Var Estimate Evolution w/ timestep')
        plt.show()




    def display_pf_plots(self, state_mean, state_cov, known_real_states, pred_var, dimension, var=1, var_SF=1):

        pred_var = np.array(pred_var)

        x0lower, x0upper, x0mid = self.compute_CIs(pred_var, means=state_mean,
                                                   covs=np.array(pred_var) * state_cov, stdevs=3,
                                                   state='x0')
        x1lower, x1upper, x1mid = self.compute_CIs(pred_var, means=state_mean,
                                                   covs=np.array(pred_var) * state_cov, stdevs=3,
                                                   state='x1')
        x2lower, x2upper, x2mid = self.compute_CIs(pred_var, means=state_mean,
                                                   covs=np.array(pred_var) * state_cov, stdevs=3,
                                                   state='x2')

        x0_pf_data = [x0lower, x0mid, x0upper]
        x1_pf_data = [x1lower, x1mid, x1upper]
        x2_pf_data = [x2lower, x2mid, x2upper]

        # --------------------------------------------------------------------------------
        range_start=0

        if dimension == 'x':
            observation_data = np.array(self.x_observation_data).flatten()
        else:
            observation_data = np.array(self.y_observation_data).flatten()

        times = self.times.flatten()

        # Set style parameters
        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        # Adjust the font sizes
        plt.rcParams.update({
            'font.size': 15,  # Set axis label and title font size
            'xtick.labelsize': 13,  # Set x-axis tick label size
            'ytick.labelsize': 13  # Set y-axis tick label size
        })

        # Create subplots with shared x-axis
        # Create subplots with shared x-axis
        fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

        # Plot for ax2
        if known_real_states != False:
            ax2.scatter(times, self.true_x_evo[:, 0], color='r', s=4, zorder=2)
            ax2.plot(times, self.true_x_evo[:, 0], zorder=1, linestyle='--', color='r', alpha=0.7,
                     label=r'True State')

        ax2.plot(times[range_start:], x0mid[range_start:], linestyle='--', label='PF Mean')
        ax2.scatter(times[range_start:], observation_data[range_start:], marker='x', s=4, color='k',
                    label='Noisy Observations')
        ax2.fill_between(times[range_start:], x0upper[range_start:], x0lower[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax2.grid(True, linewidth=0.5)
        ax2.set_ylabel(r'$\mathrm{X_0}$')

        # Plot for ax3
        ax3.plot(times[range_start:], x1mid[range_start:], linestyle='--', label='PF Mean')
        if known_real_states != False:
            # ax3.scatter(times[:], self.true_x_evo[:, 1], marker='x', s=4, color='k',
            #             label='True State')
            ax3.plot(times, self.true_x_evo[:, 1], linestyle='--', alpha=0.7, color='r',
                     )
        ax3.fill_between(times[range_start:], x1upper[range_start:], x1lower[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax3.grid(True, linewidth=0.5)
        ax3.set_ylabel(r'$\mathrm{X_1}$')

        # Plot for ax4
        ax4.plot(times[range_start:], x2mid[range_start:], linestyle='--', label='PF Mean')
        if known_real_states != False:
            ax4.plot(times, self.true_x_evo[:, 2], linestyle='--', alpha=0.7, color='r',
                     label=r'True State')
        ax4.fill_between(times[range_start:], x2upper[range_start:], x2lower[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax4.grid(True, linewidth=0.5)
        ax4.set_ylabel(r'$\mathrm{X_2}$')

        # # Add legend below the last subplot
        # handles, labels = ax4.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3))

        lines_labels = [ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        legend = fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.055))

        # for text in legend.get_texts():
        #     text.set_fontfamily('Arial')

        # legend.set_bbox_to_anchor((0.5, -0.1 + 10 / fig.dpi))

        # Set common x-axis label
        fig.text(0.5, 0.04, '$\mathrm{Time,}$ $t$', ha='center')

        # Set common y-axis label
        # fig.text(0.04, 0.5, 'State', va='center', rotation='vertical')

        # Set overall title
        # fig.suptitle(r'$\mathrm{{Particle\ Filter\ State\ Estimation}}$, $N_p$: {:.1f}, $\beta$: {:.2f}, $\kappa_v$: {:.4f}, $\kappa_\mu$: {:.4f}, $\sigma^2$: {:.2f}'.format(self.Np, self.beta, self.kv, self.kmu, self.sigmasq), y=0.95, fontsize=15)
        fig.suptitle(r'$\mathrm{{Particle\ Filter\ State\ Estimation}},\ N_p: {}$'.format(self.Np) + '\n' +
                     r'$\beta: {:.2f},\ \kappa_v: {:.4f},\ \kappa_\mu: {:.4f},\ \sigma^2: {:.2f}, \ \kappa_w: {:.2f}$'.format(
                         self.beta, self.kv, self.kmu, self.sigmasq, self.kw), y=0.93, fontsize=15)

        # # Adjust spacing between subplots
        fig.subplots_adjust(hspace=0.05)

        plt.show()

        pass

    def compute_CIs(self, pred_var, means, covs, stdevs, state):
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
            x_stdev = np.sqrt(cov[idx][idx]) * np.sqrt(pred_var)

            mid.append(x_mean)
            lower.append(x_mean - stdevs * x_stdev)
            upper.append(x_mean + stdevs * x_stdev)
        return lower, upper, mid



if __name__ == "__main__":
    theta = -0.2778
    # ssmodel = ExtendedStateSpaceModel(beta=1, kv=0.01, kmu=0, sigmasq=1, p=1,
    #                                   initial_state=[0, 0, 0.5], flatterned_A=[0, 1, 0, theta])
    # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=200)
    # ssmodel.show_plots()

    tracking_df = load_tracking_data()
    sub_sample_rate = 10
    player_x, player_y, times, fig, ax = show_player_path(tracking_df, sub_sample_rate, 10, start='18:30', end='19:30')

    # times, noisy_data = return_data_and_time_series(load_finance_data())

    pf = ParticleFilter2D(beta=1, kv=2, sigmasq=1, kmu=0.01, p=1, kw=0.01,
                        initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
                        data=[times.flatten(), player_x.flatten(), player_y.flatten()],
                          Np=500, epsilon=0.5, x_init=40, y_init=48)
    pf.run_particle_filter(pitch_obj=(fig, ax), known_real_states=False)

    # 1830 - 2030 has a good jagged path characteristic
