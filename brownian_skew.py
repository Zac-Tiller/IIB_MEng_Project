import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


rng=np.random.default_rng(1154)
times = np.insert(np.cumsum(rng.exponential(scale=1, size=100)), 0, 0)
lin_times = np.arange(0,150,1)

skew_var_generating_process = 0.0001

skew_vec = [0]
skew = 0

t = np.cumsum(rng.exponential(scale=1 / 10, size=600))


for i in range(len(t)-1):
    start = t[i]
    end = t[i+1]

    skew = skew + np.sqrt(skew_var_generating_process * (end - start)) * rng.normal(loc=0, scale=1)
    skew_vec.append(skew)
    

plt.plot(t, skew_vec)
plt.show()
