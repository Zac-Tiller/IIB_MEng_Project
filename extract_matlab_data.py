import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

# Load .mat file and extract struct data
data = sio.loadmat('dataEurUS.mat')
struct_data = data['last_traded']
# t_l = struct_data['t_l'][0][0].flatten()[::100][26:740]
# z_l = struct_data['z_l'][0][0].flatten()[::100][26:740]

t_l = struct_data['t_l'][0][0].flatten()

print((t_l - t_l[0])[:100]*24*60*60)
print((t_l - t_l[0])[-100:]*24*60*60)
z_l = struct_data['z_l'][0][0].flatten()

# Create DataFrame from the extracted fields
df = pd.DataFrame({'t_l': t_l, 'z_l': z_l})

# Plot t_l on x-axis and z_l on y-axis
plt.scatter( (df['t_l'] - df['t_l'][0])*24*60*60, df['z_l'], s=0.5)
plt.xlabel('t_l / days')
plt.ylabel('z_l')
plt.title('Plot of t_l vs z_l')
plt.show()

# 1 unit = 24 hours? therefore * 24 * 60 * 60 to get s

# 1450:2200

# t_l = struct_data['t_l'][0][0].flatten()[::][35000:49000][::4]
# z_l = struct_data['z_l'][0][0].flatten()[::][35000:49000][::4]


t_l = struct_data['t_l'][0][0].flatten()[::100][50:740]
z_l = struct_data['z_l'][0][0].flatten()[::100][50:740]

# Create DataFrame from the extracted fields
df = pd.DataFrame({'t_l': t_l, 'z_l': z_l})

# Plot t_l on x-axis and z_l on y-axis
plt.scatter( (df['t_l'] - df['t_l'][0]), df['z_l'], s=0.5) # <--- thsi is right - in days!
plt.xlabel('t_l / days')
plt.ylabel('z_l')
plt.title('Plot of t_l vs z_l 2')
plt.show()

def extract_time_and_data_series():
    data = sio.loadmat('dataEurUS.mat')
    struct_data = data['last_traded']
    t_l = struct_data['t_l'][0][0].flatten()[::100][26:740]
    z_l = struct_data['z_l'][0][0].flatten()[::100][26:740]

    time = t_l
    price = z_l

    return time, price
