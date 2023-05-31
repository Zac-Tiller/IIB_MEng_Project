from processes import *
from state_space_process import *

state_simulation = StateSpaceSimulator2(t0=0.0, T=10.0, num_obs=100, MatA=True)
# ^^ This T does not really control much... end time is approx num_obs * 0.1 (I have set the expon rate to be 0.1 when generating random obs times)
theta = -1
state_simulation.define_A([0,1,0,theta])
state_simulation.define_h([0, 1])

# --------------------------------
"NORMAL (GAMMA) DISTR PARAM"
skew = 0.3
var = 2

"GAMMA DISTR PARAM"
beta = 1
# --------------------------------

obs_times, x_evolution, ng_path = state_simulation.forward_simulate_step_by_step_gauss(skew='dynamic', beta=beta,
                                                                                       var=var, state_dim=3)
# Insight: our jump process (n) depends on theta - because ftVi = f(A) = f(theta)
# state_simulation.show_plots(obs_times, x_evolution, ng_path)