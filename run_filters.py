from kalmanfilter import *
from particlefilter import *
from datahandler import *

k_mu_BM = 0

kf = FreeStandingKalmanFilter(kv=0.01, k_mu_BM=k_mu_BM, theta=-1, initial_state=[0,0,0.5], t0=0, T=10, num_obs=50)
# ---------------------------------------------------
# time_series, obs_data_series, true_states = kf.runKalmanFilter(skew=0, beta=1, dynamic_skew={'Dynamic':False, 'var':1})
time_series, obs_data_series, true_states = kf.runKalmanFilterDynamicClosedForm(skew=1, beta=1, dynamic_skew={'Dynamic':True, 'var':1})
# ---------------------------------------------------


# with skew != 0, need a longer time sequence to estimate the marginalised variance more accurately!

# save the x_state_evo to use as artificial data in our pf!

# time_series, obs_data_series = return_data_and_time_series(load_finance_data())


data = [time_series, obs_data_series]

#
# change initial cov of KFILTER TOO!!!! check if there is an ny.eye
# had to change it in PF as i was making all particles have np.eye when i initialsied them

if k_mu_BM >= 0.01:
    k_cov_init = 0.001
else:
    k_cov_init = k_mu_BM + 0.0001

k_cov_init = 0.7

pf = ParticleFilter(kv=0.01, k_mu_BM=k_mu_BM, theta=-1, initial_state=[0,0,0], t0=0, T=10, num_obs=100)
pf.run_particle_filter(Np=1000, beta=1, k_cov_init=k_cov_init, X0_INIT=0,
                       state_obs_and_times=data, show_particle_paths=True,
                       dynamic_skew={'Dynamic': True}, known_real_states=true_states)


# k_mu = k_cov init gives better results i think ... !!!


# IN PARTICLE FILTERS, SUBTRACTED 1 FROM TIMES (FOR LOOP INDICES)