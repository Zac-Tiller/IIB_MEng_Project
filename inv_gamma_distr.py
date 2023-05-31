import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma

# Define the parameters of the inverse-gamma distribution

# from simple testing: think there is somethign wrong with my E calculations!

a = 50  # shape parameter
b = 9600  # scale parameter


a = 150
b = 0.01 #TODO: MUST investigatge the valye of E when we run on real (any synthetic) data

#TODO: must investigae real calue of E when we run on real data

a = 50
b = 50

a = 150
b = 0.01

# Generate some random data from the inverse-gamma distribution
data = invgamma.rvs(a, scale=b, size=1000)

# Create a range of x-values to plot the PDF and CDF
x = np.linspace(invgamma.ppf(0.001, a, scale=b), invgamma.ppf(0.999, a, scale=b), 100)

# Plot the PDF of the inverse-gamma distribution
plt.plot(x, np.log(invgamma.pdf(x, a, scale=b)), 'r-', lw=2, alpha=0.6, label='PDF')

# Plot the CDF of the inverse-gamma distribution
# plt.plot(x, invgamma.cdf(x, a, scale=b), 'b-', lw=2, alpha=0.6, label='CDF')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()

# Display the plot
plt.show()