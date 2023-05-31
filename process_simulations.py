from processes import *
from plotting_utils import plotter, ngplotter


# The longer the simulation, the more samples we need !
# JJs method of gsamps works well for intervals t = 0 -> 1, but, when T gets large, divergence occurs so need to increase
# sample size (which is why i x by 2/3 T in set_process_conditions)

alpha = 1
beta = 0.01
gamma_obj = GammaDistr(alpha=alpha, beta=beta)
gamma_obj.set_process_conditions(t0=0, T=1, END=None, sample_size=300)  # sample_size refers to number of random times we generate

gamma_sim = DistributionSimulator(gamma_obj) # create our simulation
path, time, jump_time_set = gamma_sim.process_simulation() #gamma_obj) # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
path2, time2, jump_time_set2 = gamma_sim.process_simulation()
path3, time3, jump_time_set3 = gamma_sim.process_simulation()
path4, time4, jump_time_set4 = gamma_sim.process_simulation()
path5, time5, jump_time_set5 = gamma_sim.process_simulation()
path6, time6, jump_time_set6 = gamma_sim.process_simulation()
path7, time7, jump_time_set7 = gamma_sim.process_simulation()
path8, time8, jump_time_set8 = gamma_sim.process_simulation()

times = [time, time2, time3, time4, time5, time6, time7, time8]
paths = [path, path2, path3, path4, path5, path6, path7, path8]

plt = plotter(times, paths, 'Gamma Process Path: ', 'Time, t', '\Gamma_t', alpha, beta)

# fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(20_000, gamma_obj))


gamma_for_ng_obj = GammaDistr(alpha=alpha, beta=beta)
gamma_for_ng_obj.set_process_conditions(t0=0, T=1, END=None, sample_size=300)
mu = 0
sigmasq=1
normal_obj = NormalDistr(mean=mu, std=np.sqrt(sigmasq), secondary_distr=gamma_for_ng_obj) # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use

normal_gamma_sim = DistributionSimulator(normal_obj)
path, time = normal_gamma_sim.process_simulation()
path2, time2 = normal_gamma_sim.process_simulation()
path3, time3 = normal_gamma_sim.process_simulation()
path4, time4 = normal_gamma_sim.process_simulation()
path5, time5 = normal_gamma_sim.process_simulation()
path6, time6 = normal_gamma_sim.process_simulation()
path7, time7 = normal_gamma_sim.process_simulation()
path8, time8 = normal_gamma_sim.process_simulation()

times = [time, time2, time3, time4, time5, time6, time7, time8]
paths = [path, path2, path3, path4, path5, path6, path7, path8]

plt = ngplotter(times, paths, 'VG Process Path', 'Time, t', '\mathcal{VG}', alpha, beta, mu, sigmasq)

fig, ax = normal_gamma_sim.plot_simulation_distribution(normal_gamma_sim.process_endpoint_sampler(20_000, normal_obj))


