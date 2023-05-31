import streamlit as st
import matplotlib.pyplot as plt
from sampling_functions import *

header = st.container()

parameter_selector = st.container()

gamma_simulation_engine = st.container()

NormalGamma_param_selector = st.container()

normal_gamma_simulaton_engine = st.container()

st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    <style>
    """,
    unsafe_allow_html=True
)


def SimulateAndShowGammaProcessPath(alpha, beta, end_time, samples):
    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=end_time, sample_size=samples)

    gamma_sim = DistributionSimulator(gamma_obj)  # create our simulation
    path, time, jump_time_set = gamma_sim.process_simulation(
        gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
    fig = plotter(time, path, 'Gamma Process Simulation', 'Time', 'Value')

    path, time, jump_time_set = gamma_sim.process_simulation(
        gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
    fig = plotter(time, path, 'Gamma Process Simulation', 'Time', 'Value')

    for i in range(5):

        path, time, jump_time_set = gamma_sim.process_simulation(
            gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
        fig = plotter(time, path, 'Gamma Process Simulation', 'Time', 'Value')
    st.pyplot(fig)


def SimulateAndShowGammaProcessHistogram(alpha, beta, samples):
    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=1, sample_size=samples)

    gamma_sim = DistributionSimulator(gamma_obj)

    fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(10000, gamma_obj))

    st.pyplot(fig)
    # st.pyplot(ax)



def SimulateAndShowNormalGammaProcessPath(alpha, beta, mean, var, end_time, samples):
    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=end_time, sample_size=samples)

    gamma_sim = DistributionSimulator(gamma_obj)  # create our simulation

    normal_obj = NormalDistr(mean=mean, std=np.sqrt(var), secondary_distr=gamma_obj)
    normal_gamma_sim = DistributionSimulator(normal_obj)

    path, time, jump_time_set = gamma_sim.process_simulation(
        gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute

    path, time = normal_gamma_sim.process_simulation(normal_obj) #, gamma_sim.sorted_process_set)
    fig = plotter(time, path, 'Normal Gamma Process Simulation', 'Time', 'Value')

    path, time = normal_gamma_sim.process_simulation(normal_obj)  # , gamma_sim.sorted_process_set)
    fig = plotter(time, path, 'Normal Gamma Process Simulation', 'Time', 'Value')

    for i in range(5):

        path, time = normal_gamma_sim.process_simulation(normal_obj)  # , gamma_sim.sorted_process_set)
        fig = plotter(time, path, 'Normal Gamma Process Simulation', 'Time', 'Value')

    st.pyplot(fig)

    return gamma_obj, normal_obj, gamma_sim

def SimulateAndShowNormalGammaProcessHistogram(alpha, beta, mean, var, samples):
    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=1, sample_size=samples)

    gamma_sim = DistributionSimulator(gamma_obj)  # create our simulation
    # check if we need to run the gamma sim

    path, time, jump_time_set = gamma_sim.process_simulation(
        gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute

    normal_obj = NormalDistr(mean=mean, std=np.sqrt(var), secondary_distr=gamma_obj)
    normal_gamma_sim = DistributionSimulator(normal_obj)


    fig, ax = normal_gamma_sim.plot_simulation_distribution(
        normal_gamma_sim.process_endpoint_sampler(10000, normal_obj)) #, G_obj=gamma_obj, G_sim_obj=gamma_sim))

    st.pyplot(fig)
    #st.pyplot(ax)

def SimulateStateSpaceProcess():
    pass




def streamlit_plotter(x, y): #, title, x_lbl, y_lbl):
    fig, ax = plt.subplots()
    ax.step(x, y)

    # plt.title(title)
    # plt.xlabel(x_lbl)
    # plt.ylabel(y_lbl)
    return fig, ax


with header:
    st.title('Interactive Simulation of a Levy Process!')

with parameter_selector:
    st.header('Gamma Process')

    st.markdown('**Choose parameters of the distribution alpha and beta:**')
    st.markdown('**Note:** For **small Beta**, increase Exp. Distr samples')


    alpha_col, beta_col, end_time_col, sample_col = st.columns(4)

    alpha = alpha_col.slider('Alpha Value?', min_value=float(0.01), max_value=float(5), step=0.01)
    beta = beta_col.slider('Beta Value?', min_value=float(0.001), max_value=float(2), step=0.001)
    end_time = end_time_col.slider('Simulation Time?', min_value=float(1), max_value=float(10), step=0.1)
    samples = sample_col.slider('Number of Samples?', min_value=500, max_value=1000)

    # gamma_obj = GammaDistr(alpha, beta)
    # gamma_obj.set_process_conditions(T=1, sample_size=samples)
    # gamma_sim = DistributionSimulator(gamma_obj)

with gamma_simulation_engine:

    show_path_col, show_samples_col = st.columns(2)

    with show_path_col:
        if st.button('Show Process Path', key="GProcess"):
            SimulateAndShowGammaProcessPath(alpha, beta, end_time, samples)



            # fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(10000, gamma_obj))
            #
            # # process_endpoints = sample_gamma_process(alpha, beta, samples, iterations=10000)
            # # fig, ax = plot_process_endpoints_samples(process_endpoints, gamma_distribution=False)
            #
            # st.pyplot(fig)
            # st.pyplot(ax)

    with show_samples_col:
        text_container = st.container
        if st.button('Show Samples', key="GHist"):
            SimulateAndShowGammaProcessHistogram(alpha, beta, samples)

            # path, times, jumps = gamma_process_sim(alpha, beta, samples, T=1)
            # fig, ax = streamlit_plotter(times, path) #, 'Gamma Process Simulation', 'Time', 'Path')


            # path, time, jump_time_set = gamma_sim.process_simulation(gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
            # fig = plotter(time, path, 'Gamma Process Simulation TEST', 'Time', 'Value')
            # st.pyplot(fig)

with NormalGamma_param_selector:
    st.header('Normal-Gamma (VG) Process')
    st.markdown('**Choose extra parameters for the Normal-Gamma - mean and variance:**')

    mean_col, var_col = st.columns(2)

    mean = mean_col.slider('Mean Value?', min_value=float(0.0), max_value=float(5), step=0.01)
    var = var_col.slider('Variance Value?', min_value=float(0.01), max_value=float(5), step=0.01)

with normal_gamma_simulaton_engine:
    show_path_col, show_samples_col = st.columns(2)

    with show_path_col:
        if st.button('Show Process Path', key="NGProcess"):
            alpha = float(1) # we want the gamma subordinator process to measure 1 as the rate of passing of time
            SimulateAndShowNormalGammaProcessPath(alpha, beta, mean, var, end_time, samples)

    with show_samples_col:
        if st.button('Show Samples', key="NGHist"):
            alpha = float(1)
            SimulateAndShowNormalGammaProcessHistogram(alpha, beta, mean, var, samples)
