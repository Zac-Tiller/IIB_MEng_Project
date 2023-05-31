import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt


def load_simons_data():
    data = sio.loadmat('dataEurUS.mat')
    struct_data = data['last_traded']
    t_l = struct_data['t_l'][0][0].flatten()[::100][50:740]
    z_l = struct_data['z_l'][0][0].flatten()[::100][50:740]

    # t_l = struct_data['t_l'][0][0].flatten()[::][10000:13000][::4]
    # z_l = struct_data['z_l'][0][0].flatten()[::][10000:13000][::4]

    # t_l = struct_data['t_l'][0][0].flatten()[::][35000:49000][::4]
    # z_l = struct_data['z_l'][0][0].flatten()[::][35000:49000][::4]

    time = t_l
    price = z_l

    return time, price



def load_finance_data():

    finance_df = pd.read_csv(
                  '/Users/zactiller/Documents/IIB_MEng_Project/Finance_Data/USDJPY-2023-03.csv',
        skiprows=2)
    finance_df.columns = ['Pair', 'Date - Time', 'Bid', 'Ask']
    finance_df = finance_df.drop_duplicates(subset=['Date - Time'], keep="first")

    # finance_df = finance_df[35000:35500][::3] FOR GBPJPY-2023-01
    # a = 19000

    a = 1500
    finance_df = finance_df[a:a+300][::] # FOR CHFJPY, 300

    finance_df['Date - Time'] = pd.to_datetime(finance_df['Date - Time'], format='%Y%m%d %H:%M:%S.%f')
    print(finance_df)

    return finance_df

def return_data_and_time_series(finance_data):
    "Returns time series starting at 0 and going to t (eg 115 seconds)"

    time_series = finance_data['Date - Time'].diff().fillna(pd.Timedelta(seconds=0)).cumsum().dt.total_seconds().values
    # data_series = (finance_data['Bid'].values + finance_data['Ask'].values)/2

    data_series = finance_data['Bid'].values # for CHF/JPY

    return time_series, data_series
if __name__=="__main__":
    t, p = return_data_and_time_series(load_finance_data())

    plt.scatter(t, p, s=0.5)
    plt.show()