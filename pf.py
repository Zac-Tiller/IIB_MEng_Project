import numpy as np
import random
import pandas as pd
import copy
from copy import deepcopy
from tqdm import tqdm
from sss import ExtendedStateSpaceModel
from processes import *
from tracking_data import *
from datahandler import *


def util_logsumexp(lw, h, x, axis=0, retlog=False):
	"""helper function to calculate the log of a sum of exponentiated values"""
	c = np.max(lw)
	broad_l = np.broadcast_to((lw-c).flatten(), x.T.shape).T
	if retlog:
		return c + np.log(np.sum(np.exp(broad_l) * h(x), axis=axis))
	else:
		return np.exp(c) * np.sum(np.exp(broad_l) * h(x), axis=axis)


class Particle(ExtendedStateSpaceModel):
    '''each particle is an instance of the Particle class - allows lightweight code'''

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
        # self.kmean_updt = self.kmean_updt + (self.kcov_updt @ np.random.randn(3)).reshape(-1, 1)
        self.kmean_updt = self.kmean_updt

        self.kmean_pred = 0
        self.kcov_pred = 0

        # particles have a (log)weight, count and an E
        self.logweight = 0.
        self.count = 0.
        self.E = 0.

        self.theta = flatterned_A[3]

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
                new_particles.append(copy.deepcopy(self.particles[idx]))

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
        # pred_mean_var_meth_two = util_logsumexp(lweights, lambda x: x, individual_particle_IG_means, retlog=False)
        # print('pred mean var meth 2: {}'.format(pred_mean_var_meth_two))

        Es = np.array([particle.E for particle in self.particles])
        E = util_logsumexp(lweights, lambda x: x, Es, retlog=False)

        if plot_density:
            xx = np.linspace(0.4, 2, 100000)
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

            # fig999, ax999 = plt.subplots()
            # ax999.plot(xx, np.array(inv_gamma_pdf).flatten(), 'r-', lw=1, alpha=0.6, label='invgamma pdf')
            # fig999.suptitle('Inv Gamma Distr For Estimated Var')

            # Set style parameters
            plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams.update({
                'font.size': 15,  # Set axis label and title font size
                'xtick.labelsize': 12,  # Set x-axis tick label size
                'ytick.labelsize': 12  # Set y-axis tick label size
            })

            fig999, ax999 = plt.subplots()
            ax999.plot(xx, np.array(inv_gamma_pdf).flatten(), 'r-', lw=1, alpha=0.6, label='invgamma pdf')
            # Add gridlines
            ax999.grid(True, linestyle='-', linewidth=0.5)
            # Set gridline weight
            ax999.xaxis.grid(linewidth=0.5)
            ax999.yaxis.grid(linewidth=0.5)

            # Set title and labels
            ax999.set_title(r'$\mathrm{{Inverse\ Gamma\ Distribution}}$', fontsize=15)
            ax999.set_xlabel(r'$\sigma^2$', fontsize=13)
            ax999.set_ylabel(r'$p(\sigma^2)$', fontsize=13)
            # Change resolution
            fig999.set_dpi(300)
            # Change font to 'cm'

            # Display the plot
            plt.show()

            x_lower = pred_mean_var - 3*pred_mean_var
            if x_lower < 0:
                x_lower = 0
            else:
                x_lower = x_lower

            # ax999.set_xlim(x_lower, pred_mean_var + 3*pred_mean_var)
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



    def run_particle_filter(self, FXD_SKEW, mse_calcs, known_real_states=False, show_plots=True):

        if known_real_states != False:
            self.true_x_evo = known_real_states
            self.true_x_evo = np.reshape(self.true_x_evo, (int(np.size(self.true_x_evo)/3), 3))

        init_mean, init_cov = self.compute_state_posterior()
        state_mean = [init_mean]
        state_cov = [init_cov]
        mode_var_estimates = []
        mean_var_estimates = []

        MSE_x0 = []
        MSE_x1 = []
        MSE_x2 = []

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
                    MSE_x0.append( (predictive_mean[0][0] - noisy_observation)**2)
                    if known_real_states != False:
                        MSE_x1.append((predictive_mean[1][0] - self.true_x_evo[:,1][t-1]) ** 2)
                        MSE_x2.append((predictive_mean[2][0] - self.true_x_evo[:,2][t-1]) ** 2)




                state_mean.append(mixed_mean)
                state_cov.append(mixed_cov)

                # mode_var, mean_var = self.estimate_marginalised_var()
                pred_mean_var, pred_mode_var = self.estimate_marginalised_var()
                mean_var_estimates.append(pred_mean_var)
                mode_var_estimates.append(pred_mode_var)



        pred_mean_var, pred_mode_var = self.estimate_marginalised_var(plot_density=False)
        print('Estimated Var: {} MEAN'.format(pred_mean_var))
        print('Estimated Var: {} MODE'.format(pred_mode_var))

        if mse_calcs:
            print('Total Predictive (y-hat - y) MSE: {}'.format(np.sum(MSE_x0)/len(MSE_x0)))
            if known_real_states != False:
                print('Total Predictive (pred-x1 - x1) MSE: {}'.format(np.sum(MSE_x1) / len(MSE_x1)))
                print('Total Predictive (pred-x2 - x2) MSE: {}'.format(np.sum(MSE_x2) / len(MSE_x2)))

        if show_plots:
            self.plot_var_convergence(mode_var_estimates[5:], mean_var_estimates[5:])
            self.display_pf_plots(FXD_SKEW, state_mean, state_cov, known_real_states, pred_mode_var)

        log_likelihood = self.log_marginal_likelihood

        if known_real_states:
            ret = (np.sum(MSE_x0)/len(MSE_x0), np.sum(MSE_x1)/len(MSE_x1), np.sum(MSE_x2)/len(MSE_x2), log_likelihood)
        else:
            ret = (np.sum(MSE_x0)/len(MSE_x0), log_likelihood)

        return ret


    def plot_var_convergence(self, var_estimate_list, mean_var_list):

        # Set style parameters
        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams.update({
            'font.size': 15,  # Set axis label and title font size
            'xtick.labelsize': 10,  # Set x-axis tick label size
            'ytick.labelsize': 10  # Set y-axis tick label size
        })

        fig101, ax101 = plt.subplots()
        ax101.plot(np.linspace(1, len(var_estimate_list), len(var_estimate_list)), np.array(var_estimate_list).flatten(), label='Mode')
        ax101.plot(np.linspace(1, len(mean_var_list), len(mean_var_list)), np.array(mean_var_list).flatten(), label = 'Mean')
        ax101.legend()
        ax101.set_title(r'$\mathrm{{Variance\ Estimate\ Evolution}}$', fontsize=15)
        ax101.set_xlabel(r'$\mathrm{{Timestep}}$', fontsize=13)
        ax101.set_ylabel(r'$\sigma^2$', fontsize=13)

        # fig101.suptitle('Var Estimate Evolution w/ timestep')
        plt.show()




    def display_pf_plots(self, FXD_SKEW, state_mean, state_cov, known_real_states, pred_var, var=1, var_SF=1):

        predicted_var = 1
        # note - we move scaling  into compute_CIs !!!!!!!!!

        x0lower, x0upper, x0mid = self.compute_CIs(means=state_mean,
                                                   covs=predicted_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x0', pred_var=pred_var)
        x1lower, x1upper, x1mid = self.compute_CIs(means=state_mean,
                                                   covs=predicted_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x1', pred_var=pred_var)
        x2lower, x2upper, x2mid = self.compute_CIs(means=state_mean,
                                                   covs=predicted_var * var_SF * var * state_cov, stdevs=3,
                                                   state='x2', pred_var=pred_var)

        x0_pf_data = [x0lower, x0mid, x0upper]
        x1_pf_data = [x1lower, x1mid, x1upper]
        x2_pf_data = [x2lower, x2mid, x2upper]

        # --------------------------------------------------------------------------------
        range_start=0

        observation_data = np.array(self.observation_data).flatten()
        times = self.times - np.array(self.times[0])

        # Set style parameters
        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        # Adjust the font sizes
        plt.rcParams.update({
            'font.size': 15,  # Set axis label and title font size
            'xtick.labelsize': 12,  # Set x-axis tick label size
            'ytick.labelsize': 12  # Set y-axis tick label size
        })

        # Create subplots with shared x-axis
        # Create subplots with shared x-axis
        fig, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

        # Plot for ax2
        if known_real_states != False:
            ax2.scatter(times, self.true_x_evo[:, 0], color='r', s=2, zorder=2)
            ax2.plot(times, self.true_x_evo[:, 0], zorder=1, linestyle='--', color='r', alpha=0.7,
                     label=r'True State')

        ax2.plot(times[range_start:], x0mid[range_start:], linestyle='--', label='PF Mean')
        ax2.scatter(times[range_start:], observation_data[range_start:], marker='x', s=2, color='k',
                    label='Noisy Observations')
        ax2.fill_between(times[range_start:], np.array(x0upper).flatten()[range_start:], np.array(x0lower).flatten()[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax2.grid(True, linewidth=0.5)
        ax2.set_ylabel(r'$\mathrm{X_0}$, $\mathrm{EUR/USD}$, $\mathrm{Bid}$')
        # if FXD_SKEW:
        #     ax2.set_xlim(0, 11.8)
        #     ax2.set_ylim(-1, 4)


        # Plot for ax3
        ax3.plot(times[range_start:], x1mid[range_start:], linestyle='--', label='PF Mean')
        if known_real_states != False:
            # ax3.scatter(times[:], self.true_x_evo[:, 1], marker='x', s=4, color='k',
            #             label='True State')
            ax3.plot(times, self.true_x_evo[:, 1], linestyle='--', alpha=0.7, color='r',
                     )
        ax3.fill_between(times[range_start:], np.array(x1upper).flatten()[range_start:], np.array(x1lower).flatten()[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax3.grid(True, linewidth=0.5)
        ax3.set_ylabel(r'$\mathrm{X_1}$, $\mathrm{Trend}$')
        # if FXD_SKEW:
        #     ax3.set_xlim(0, 11.8)
        #     ax3.set_ylim(-2.5, 4)
        # else:
        #     ax3.set_ylim(-2.75, 2.75)
        #     ax3.set_xlim(0, 15)

        # Plot for ax4
        ax4.plot(times[range_start:], x2mid[range_start:], linestyle='--', label='PF Mean')
        if known_real_states != False:
            ax4.plot(times, self.true_x_evo[:, 2], linestyle='--', alpha=0.7, color='r',
                     label=r'True State')
        ax4.fill_between(times[range_start:], np.array(x2upper).flatten()[range_start:], np.array(x2lower).flatten()[range_start:], alpha=0.3,
                         label='PF $\pm$99% CI')
        ax4.grid(True, linewidth=0.5)
        ax4.set_ylabel(r'$\mathrm{X_2}$, $\mathrm{Skew}$')

        # # Add legend below the last subplot
        # handles, labels = ax4.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.3))

        lines_labels = [ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        legend = fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.055))

        # for text in legend.get_texts():
        #     text.set_fontfamily('Arial')

        # legend.set_bbox_to_anchor((0.5, -0.1 + 10 / fig.dpi))

        # Set common x-axis label
        fig.text(0.5, 0.04, '$\mathrm{Time,}$ $t$ $\mathrm{/ day}$', ha='center')
        plt.ticklabel_format(style='plain', axis='y')


        # Set common y-axis label
        # fig.text(0.04, 0.5, 'State', va='center', rotation='vertical')

        # Set overall title
        # fig.suptitle(r'$\mathrm{{Particle\ Filter\ State\ Estimation}}$, $N_p$: {:.1f}, $\beta$: {:.2f}, $\kappa_v$: {:.4f}, $\kappa_\mu$: {:.4f}, $\sigma^2$: {:.2f}'.format(self.Np, self.beta, self.kv, self.kmu, self.sigmasq), y=0.95, fontsize=15)
        fig.suptitle(r'$\mathrm{{Particle\ Filter\ State\ Estimation}},\ N_p: {}$'.format(self.Np) + '\n' +
                     r'$\beta: {:.2f},\ \kappa_v: {:.4f},\ \kappa_\mu: {:.4f},\ \sigma^2: {:.2f}, \ \kappa_w: {:.2f}, \ \theta: {:.2f} $'.format(
                        self.beta, self.kv, self.kmu, self.sigmasq, self.kw, self.particles[0].theta), y=0.93, fontsize=15)

        # # Adjust spacing between subplots
        fig.subplots_adjust(hspace=0.05)

        # Show the stacked plots
        plt.show()




















        # fig2, ax2 = plt.subplots()
        # if known_real_states != False:
        #     ax2.scatter(times, self.true_x_evo[:,0], color='r', s=4, zorder=2)
        #     ax2.plot(times, self.true_x_evo[:,0], zorder=1, linestyle='--', color='r', alpha=0.7,
        #              label='True X0 State')  # HUHJKHK
        #
        # ax2.plot(times[range_start:], x0mid[range_start:], linestyle='--', label='PF Mean')
        # ax2.scatter(times[range_start:], observation_data[range_start:], marker='x', s=4, color='k',
        #             label='Noisy State Observations')
        # ax2.fill_between(times[range_start:], x0upper[range_start:], x0lower[range_start:], alpha=0.4,
        #                  label='PF +-99% CI')
        # ax2.legend()
        # fig2.suptitle(
        #     'X0 Evolution - Particle Filter - Updated - k_mu_BM = {}, kv = {}. Np = {} \n beta ={}, k_cov_init = {}'.format(self.kmu,
        #                                                                                                 self.kv,
        #                                                                                                 self.Np,
        #                                                                                                 self.beta,
        #                                                                                                 self.kw))
        # plt.xlabel('Time')
        # plt.ylabel('X0')
        # plt.show()
        #
        # # --------------------------------------------------------------------------------
        #
        # fig3, ax3 = plt.subplots()
        # # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK
        #
        # ax3.plot(times[range_start:], x1mid[range_start:], linestyle='--', label='PF Mean')
        # if known_real_states != False:
        #     ax3.scatter(times[:], self.true_x_evo[:,1], marker='x', s=4, color='k',
        #                 label='True X1 State Underlying')
        #     ax3.plot(times, self.true_x_evo[:,1], linestyle='--', alpha=0.7, color='r',
        #              label='True X1 State Underlying')
        # ax3.fill_between(times[range_start:], x1upper[range_start:], x1lower[range_start:], alpha=0.4,
        #                  label='PF +-99% CI')
        # ax3.legend()
        # fig3.suptitle(
        #     'X1 Evolution - Particle Filter - Updated - k_mu_BM = {}, kv = {}. Np = {}\n beta ={}, k_cov_init = {}'.format(self.kmu,
        #                                                                                                 self.kv,
        #                                                                                                 self.Np,
        #                                                                                                 self.beta,
        #                                                                                                 self.kw))
        # plt.xlabel('Time')
        # plt.ylabel('X1')
        # plt.show()
        #
        # # --------------------------------------------------------------------------------
        #
        # fig4, ax4 = plt.subplots()
        # # ax1.scatter(obs_times, x_evolution[0][:], color='r', s=4, zorder=2)
        # # ax2.plot(times, x_evolution[0][:], zorder=1, linestyle='--', color='r', alpha=0.7, label='True X0 State') # HUHJKHK
        #
        # ax4.plot(times[range_start:], x2mid[range_start:], linestyle='--', label='PF Mean')
        # # ax3.scatter(times, pf.true_x_evo[1][:], marker='x', s=4, color='k', label='True X1 State Underlying')
        # if known_real_states != False:
        #     ax4.plot(times, self.true_x_evo[:,2], linestyle='--', alpha=0.7, color='r',
        #              label='True X2 State Underlying')
        # ax4.fill_between(times[range_start:], x2upper[range_start:], x2lower[range_start:], alpha=0.4,
        #                  label='PF +-99% CI')
        # ax4.legend()
        # fig4.suptitle(
        #     'X2 - Particle Filter - k_mu_BM = {} kv = {}. Np = {} \n beta = {}, k_cov_init = {}'.format(self.kmu,
        #                                                                                                 self.kv,
        #                                                                                                 self.Np,
        #                                                                                                 self.beta,
        #                                                                                                 self.kw))
        # plt.xlabel('Time')
        # plt.ylabel('X2')
        # plt.show()
        #
        # pass






    def compute_CIs(self, means, covs, stdevs, state, pred_var):
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

        # pred_var = 1.065

        for mean, cov in zip(means, covs):
            x_mean = mean[idx][0]
            x_stdev = np.sqrt(cov[idx][idx]) * np.sqrt(pred_var)

            mid.append(x_mean)
            lower.append(x_mean - stdevs * x_stdev)
            upper.append(x_mean + stdevs * x_stdev)
        return lower, upper, mid


if __name__ == "__main__":
    # theta = -0.85


    # theta = -2
    # ssmodel = ExtendedStateSpaceModel(beta=1, kv=0.01, kmu=0.1, sigmasq=1, p=1,
    #                                   initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta])
    # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=200)
    # ssmodel.show_plots()
    #
    # # tracking_df = load_tracking_data()
    # # player_x, player_y, times = show_player_path(tracking_df, 10, start='18:30', end='18:45')
    #
    # # times, noisy_data = return_data_and_time_series(load_finance_data())
    # # times, noisy_data = load_simons_data()
    #
    # pf = ParticleFilter(beta=1, kv=0.01, sigmasq=1, kmu=0, p=1, kw=0.4,
    #                     initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                     data=[times, noisy_data], Np=20, epsilon=0.5, two_D_tracking=False)
    # # if known_real_states is True, pass true_states as the argument
    # x0mse, x1mse, x2mse = pf.run_particle_filter(mse_calcs=True, known_real_states=true_states)




    times, noisy_data = load_simons_data()

    # beta_range = np.arange(0.1, 3, 0.3)
    # theta_range = np.arange(-0.01, -10, -1)
    # # Initialize the grid
    # L_grid = np.zeros((len(beta_range), len(theta_range)))
    # plt.rcParams.update({
    #     'font.size': 15,  # Set axis label and title font size
    #     'xtick.labelsize': 12,  # Set x-axis tick label size
    #     'ytick.labelsize': 12  # Set y-axis tick label size
    # })
    # # Set style parameters
    # plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
    # plt.rcParams['figure.dpi'] = 300
    # count = 0
    # for i, beta in enumerate(beta_range):
    #     for j, theta in enumerate(theta_range):
    #         print()
    #         print(' ----- COUNT ------ {} ------'.format(count))
    #         print()
    #         pf = ParticleFilter(beta=beta, kv=0.01, sigmasq=1, kmu=0.01, p=1, kw=0.9,
    #                             initial_state=[1.195, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                             data=[times, noisy_data], Np=100, epsilon=0.5, two_D_tracking=False)
    #
    #         x0mse, L = pf.run_particle_filter(FXD_SKEW=False, mse_calcs=True, known_real_states=False, show_plots=False)
    #
    #
    #         # x0mse, x1mse, x2mse, L =
    #
    #         # Store the resulting L value in the grid
    #         L_grid[i, j] = L
    #         count +=1
    #     count += 1
    #
    #
    # # Plot the heatmap
    # plt.figure()
    # plt.imshow(L_grid, cmap='hot', extent=[theta_range[0], theta_range[-1], beta_range[0], beta_range[-1]],
    #            aspect='auto', origin='lower', interpolation='spline16')
    # plt.colorbar(label='L value')
    # plt.xlabel(r'$\theta$')
    # plt.ylabel(r'$\beta$')
    # plt.title(r'$\mathrm{{Grid\ Search\ Heatmap}}$')
    # plt.show()


    # GRID SEARCH FOUND VALUES::::
    beta = 1
    theta = -5.01

    # theta = -0.3

    # times, noisy_data = return_data_and_time_series(load_finance_data())

    pf = ParticleFilter(beta=beta, kv=0.0001, sigmasq=1, kmu=0.001, p=1, kw=0.01,
                        initial_state=[1.195, 0, 0], flatterned_A=[0, 1, 0, theta],
                        data=[times, noisy_data], Np=1000, epsilon=0.5, two_D_tracking=False)

    x0mse, L = pf.run_particle_filter(FXD_SKEW=False, mse_calcs=True, known_real_states=False, show_plots=True)
    print('xo mse: {}'.format(x0mse))


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    MSE_dict = {'fixed-skew-kmu-0': {'x0mse': [], 'x1mse': [], 'x2mse':[]},
                'fixed-skew-kmu-non-0':{'x0mse': [], 'x1mse': [], 'x2mse':[]},
                'large-kmu-NON-DS-MODEL':{'x0mse': [], 'x1mse': [], 'x2mse':[]},
                'large-kmu-DS-MODEL':{'x0mse': [], 'x1mse': [], 'x2mse':[]}}
    #
    # # ---------------------------------------------------------------------------
    # # ---------------------------------------------------------------------------
    #
    #
    # theta = -2
    # ssmodel = ExtendedStateSpaceModel(beta=1.5, kv=0.01, kmu=0, sigmasq=1, p=1,
    #                                   initial_state=[0, 0, 1], flatterned_A=[0, 1, 0, theta])
    # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=120)
    # ssmodel.show_plots()
    # pf = ParticleFilter(beta=1.5, kv=0.01, sigmasq=1, kmu=0, p=1, kw=0.5,
    #                     initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                     data=[times, noisy_data], Np=1000, epsilon=0.5, two_D_tracking=False)
    # # if known_real_states is True, pass true_states as the argument
    # x0mse, x1mse, x2mse, L = pf.run_particle_filter(FXD_SKEW=True, mse_calcs=True, known_real_states=true_states)
    # MSE_dict['fixed-skew-kmu-0']['x0mse'] = x0mse
    # MSE_dict['fixed-skew-kmu-0']['x1mse'] = x1mse
    # MSE_dict['fixed-skew-kmu-0']['x2mse'] = x2mse
    #
    #
    # # run 2nd PF with diff parameters
    # pf = ParticleFilter(beta=1.5, kv=0.01, sigmasq=1, kmu=0.1, p=1, kw=0.5,
    #                     initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                     data=[times, noisy_data], Np=1000, epsilon=0.5, two_D_tracking=False)
    # # if known_real_states is True, pass true_states as the argument
    # x0mse, x1mse, x2mse, L = pf.run_particle_filter(FXD_SKEW=True, mse_calcs=True, known_real_states=true_states)
    # MSE_dict['fixed-skew-kmu-non-0']['x0mse'] = x0mse
    # MSE_dict['fixed-skew-kmu-non-0']['x1mse'] = x1mse
    # MSE_dict['fixed-skew-kmu-non-0']['x2mse'] = x2mse






    # theta = -2
    # ssmodel = ExtendedStateSpaceModel(beta=1, kv=0.01, kmu=0.1, sigmasq=1, p=1,
    #                                   initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta])
    # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=145) # generate dyn-skew-data
    # ssmodel.show_plots()
    #
    # pf = ParticleFilter(beta=1, kv=0.01, sigmasq=1, kmu=0, p=1, kw=0.4,
    #                     initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                     data=[times, noisy_data], Np=1000, epsilon=0.5, two_D_tracking=False)
    # # if known_real_states is True, pass true_states as the argument
    # x0mse, x1mse, x2mse, L = pf.run_particle_filter(FXD_SKEW=False, mse_calcs=True, known_real_states=true_states)
    # MSE_dict['large-kmu-NON-DS-MODEL']['x0mse'] = x0mse
    # MSE_dict['large-kmu-NON-DS-MODEL']['x1mse'] = x1mse
    # MSE_dict['large-kmu-NON-DS-MODEL']['x2mse'] = x2mse
    # #
    # #
    # pf = ParticleFilter(beta=1, kv=0.01, sigmasq=1, kmu=0.1, p=1, kw=0.4,
    #                     initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta],
    #                     data=[times, noisy_data], Np=1000, epsilon=0.5, two_D_tracking=False)
    # # if known_real_states is True, pass true_states as the argument
    # x0mse, x1mse, x2mse, L = pf.run_particle_filter(FXD_SKEW=False, mse_calcs=True, known_real_states=true_states)
    # MSE_dict['large-kmu-DS-MODEL']['x0mse'] = x0mse
    # MSE_dict['large-kmu-DS-MODEL']['x1mse'] = x1mse
    # MSE_dict['large-kmu-DS-MODEL']['x2mse'] = x2mse
    #
    #
    # print(MSE_dict)
    #
    # for k,v in MSE_dict:
    #     print(k, v)
    #
    # print('end')

