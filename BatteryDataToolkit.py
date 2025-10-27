import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
import math
import pywt

class DataInitializer:
    def __init__(self, cutoff=0.1, fs=2, order=2, window_size_f1=30, window_size_f2=30, SG_order1 = 5, SG_order2 = 5, C_init= None):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.window_size_f1 = window_size_f1
        self.window_size_f2 = window_size_f2
        self.window_size_f3 = 50
        self.SG_order1 = SG_order1
        self.SG_order2 = SG_order2
        self.SG_order3 = 5
        self.C_init = C_init
        


    def wavelet_denoising(self, data, wavelet='db4', level=2):
        data = np.array(data)
        coeff = pywt.wavedec(data, wavelet, mode="per")
        sigma = np.median(np.abs(coeff[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:])
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode='per')
        return reconstructed_signal[:len(data)]
    
    def low_pass_filter(self, data):
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    
    def filter_SG1(self, x):
        if len(x) >= self.window_size_f1:
            return savgol_filter(x, self.window_size_f1, self.SG_order1)
        else:
            return x

    def filter_SG2(self, x):
        if len(x) >= self.window_size_f2:
            return savgol_filter(x, self.window_size_f2, self.SG_order2)
        else:
            return x
        
    def filter_SG3(self, x):
        if len(x) >= self.window_size_f3:
            return savgol_filter(x, self.window_size_f3, self.SG_order3)
        else:
            return x
                
    def length_equalizer(self, input_list, len_eq = None):    
        if not input_list:
            return np.array([])
        if len_eq:
            max_length = len_eq
        else:
            valid_lengths = [len(signal) for signal in input_list if hasattr(signal, '__len__') and len(signal) > 0]
            if not valid_lengths:
                return np.array([[] for _ in input_list]) # Return list of empty arrays if all were empty
            max_length = max(valid_lengths)

        uniformed_signals = []
        for signal in input_list:
            if not hasattr(signal, '__len__') or len(signal) == 0:
                uniformed_signals.append(np.zeros(max_length))
                continue

            original_indices = np.linspace(0, len(signal) - 1, num=len(signal))
            target_indices = np.linspace(0, len(signal) - 1, num=max_length)
            interpolated_signal = np.interp(target_indices, original_indices, signal)
            uniformed_signals.append(interpolated_signal)
            
        return np.array(uniformed_signals)
    
    def load_Experimental(self, file_path, min_temp = None, max_temp = None):
        print("______________")
        alldata = pd.read_csv(file_path, header=None)
        alldata.columns = ['DataPoint', 'Cycle', 'Step Type', 'Time','Total Time', 'Current', 'Voltage',
                           'Capacity', 'Temperature']

        resample_interval = int(3)
        alldata = alldata.iloc[::resample_interval, :]
        alldata['Temperature'] = alldata.groupby('Step Type')['Temperature'].transform(self.low_pass_filter)
        alldata['Temperature'] = alldata.groupby('Step Type')['Temperature'].transform(self.filter_SG1)       
        alldata['dT/dV'] = alldata.groupby('Step Type')['Temperature'].diff() / alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dT/dV'] = alldata.groupby('Step Type')['dT/dV'].transform(self.filter_SG2)
        alldata['-dT/dV'] = -1 * alldata['dT/dV']
        alldata['dT/dt'] = alldata.groupby('Step Type')['Temperature'].diff()
        alldata['dT/dt'] = alldata.groupby('Step Type')['dT/dt'].transform(self.filter_SG2)
        alldata['dV/dt'] = alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dV/dQ'] = alldata.groupby('Step Type')['Voltage'].diff()/alldata.groupby('Step Type')['Capacity'].diff()
        alldata['dQ/dV'] = alldata.groupby('Step Type')['Capacity'].diff()/alldata.groupby('Step Type')['Voltage'].diff()
        alldata['dQ/dV'] = alldata.groupby('Step Type')['dQ/dV'].transform(self.filter_SG2)
        alldata['Time'] = pd.to_timedelta(alldata['Time']).dt.total_seconds()

        alldata = alldata.replace([np.inf, -np.inf], np.nan)
        alldata = alldata.bfill()

        data_dch_all = alldata.loc[alldata['Step Type'] == 'CC DChg'].copy()
        data_dch_filtered = data_dch_all.loc[data_dch_all['Current'] < -0.5].copy()

        valid_cycles = data_dch_filtered.groupby('Cycle')['Time'].max()
        time_limit = 2900 if file_path[-9:] == 'V42-C.csv' else 7600
        valid_cycles = valid_cycles[valid_cycles < time_limit].index

        data_dch = data_dch_filtered[data_dch_filtered['Cycle'].isin(valid_cycles)].copy()
        data_ch = alldata.loc[alldata['Step Type'] == 'CC Chg'].copy()

        if file_path[-9:] == 'V45-B.csv':
            cycles = sorted(data_dch['Cycle'].unique())[:-5]
        elif file_path[-12:] == 'V42-B-HT.csv' or file_path[-7:] == 'OD1.csv' or file_path[-7:] == 'OD2.csv':
            cycles = sorted(data_dch['Cycle'].unique())
        else:
            cycles = sorted(data_dch['Cycle'].unique())[:-1]

        dch_cap, ch_cap = [], []

        time_dch, voltage_dch, current_dch, temperature_dch, DTV_dch, DT_dch, DV_dch, ICA_dch, C_dch = [], [], [], [], [], [], [], [], []
        time_ch, voltage_ch, current_ch, temperature_ch, DTV_ch, DT_ch, DV_ch, ICA_ch, C_ch = [], [], [], [], [], [], [], [], []

        for cycle in cycles:
            if cycle not in []:
                cycle_data_dch = data_dch[data_dch['Cycle'] == cycle]
                cycle_data_ch = data_ch[data_ch['Cycle'] == cycle]

                # Append time-series data
                time_dch.append(cycle_data_dch['Time'].values)
                voltage_dch.append(cycle_data_dch['Voltage'].values)
                current_dch.append(cycle_data_dch['Current'].values)
                temperature_dch.append(cycle_data_dch['Temperature'].values)
                DTV_dch.append(cycle_data_dch['dT/dV'].values)
                DT_dch.append(cycle_data_dch['dT/dt'].values)
                DV_dch.append(cycle_data_dch['dV/dt'].values)
                ICA_dch.append(cycle_data_dch['dQ/dV'].values)
                C_dch.append(cycle_data_dch['Capacity'].values)

                time_ch.append(cycle_data_ch['Time'].values)
                voltage_ch.append(cycle_data_ch['Voltage'].values)
                current_ch.append(cycle_data_ch['Current'].values)
                temperature_ch.append(cycle_data_ch['Temperature'].values)
                DTV_ch.append(cycle_data_ch['dT/dV'].values)
                DT_ch.append(cycle_data_ch['dT/dt'].values)
                DV_ch.append(cycle_data_ch['dV/dt'].values)
                ICA_ch.append(cycle_data_ch['dQ/dV'].values)
                C_ch.append(cycle_data_ch['Capacity'].values)


                # Append capacity
                dch_cap.append(np.max(cycle_data_dch['Capacity'].dropna()))
                ch_cap.append(np.max(cycle_data_ch['Capacity'].dropna()))

        # --- Package original (unequalized) time-series data ---
        data_original_dch = {
            'Voltage': voltage_dch, 'Temperature': temperature_dch,
            'Time': time_dch, 'DTV': DTV_dch, 'DT': DT_dch, 'DV': DV_dch, 'ICA': ICA_dch, 'C_dch':C_dch
        }
        data_original_ch = {
            'Voltage': voltage_ch, 'Temperature': temperature_ch,
            'Time': time_ch, 'DTV': DTV_ch, 'DT': DT_ch, 'DV': DV_ch, 'ICA': ICA_ch, 'C_ch':C_ch
        }

        len_eq = 96
        data_eq_ch = {key: self.length_equalizer(val, len_eq = len_eq) for key, val in data_original_ch.items()}
        data_eq_dch = {key: self.length_equalizer(val, len_eq = len_eq) for key, val in data_original_dch.items()}
        
        EoL = {'cycle':None, 'capacity':None}
        if self.C_init is not None:
            for cycle, cap in zip(cycles, dch_cap):
                if cap < 0.8 * self.C_init:
                    EoL = {'cycle':cycle, 'capacity':cap}
                    print(f"Cycle {cycle} drops below 80% of initial capacity (C_init = {self.C_init:.3f}) with capacity = {cap:.3f}")
                    break

        return_data = {
            'All': alldata,
            'data_dch': data_dch,
            'data_ch': data_ch,
            'Original_dch': data_original_dch,
            'Original_ch': data_original_ch,
            'Equalized_dch': data_eq_dch,
            'Equalized_ch': data_eq_ch,
            'cycles': cycles,
            'dch_cap': dch_cap,
            'ch_cap': ch_cap,
            'EoL':EoL
        }

        print(f"Cell data loaded from {file_path}")
        return return_data

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def extract_axis_data(self, cycle_data, ax):
        if ax != 'Time':
            return cycle_data[ax].dropna()
        else:
            return cycle_data['Time'].dropna().to_numpy() - cycle_data['Time'].iloc[0]

    def plot_3D(self, ax1, ax2, ax3, start_cycle, end_cycle):
        cmap = plt.get_cmap('plasma')
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

        cycles = sorted(self.data['Cycle'].unique())
        filtered_cycles = [cycle for cycle in cycles if start_cycle <= cycle < end_cycle]

        for cycle in range(1,len(filtered_cycles)):
            if cycle not in []:
                cycle_data = self.data[(self.data['Cycle'] == cycle)]

                Ax1 = self.extract_axis_data(cycle_data, ax1)
                Ax2 = self.extract_axis_data(cycle_data, ax2)
                Ax3 = self.extract_axis_data(cycle_data, ax3)

                color = cmap(cycle / (end_cycle - 1))
                rgb_color = f'rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)})'
                num_ticks = 5
                tick_values = np.linspace(start_cycle, end_cycle, num_ticks + 2).astype(int)

                trace = go.Scatter3d(
                    x=Ax1,
                    y=Ax2,
                    z=Ax3,
                    mode='lines',
                    showlegend=False,
                    line=dict(
                        color=rgb_color,
                        colorbar=dict(
                            title='Cycle Number',
                            tickvals=tick_values,
                            ticks='inside'),
                        cmin=start_cycle,
                        cmax=end_cycle
                    ))
                fig.add_trace(trace)

        fig.update_layout(
            scene=dict(
                xaxis_title=ax1,
                yaxis_title=ax2,
                zaxis_title=ax3
            ))
        fig.show()

    def plot_degradation(self, mode, C_initial=None, marker='o', color='green', markersize=3, label=None, EoL=None):
        # mode : ['dch_cap'/'ch_cap', start cycle, reference capacity cycle]
        ch_dch = mode[0]
        start_cycle = mode[1]
        ref_cycle = mode[2]

        Capacity = self.data[ch_dch]

        if C_initial is not None:
            C_n = Capacity / C_initial
        else:
            C_n = Capacity / Capacity[ref_cycle]

        cycles = np.array(self.data['cycles']) 
        mask = np.where(cycles > start_cycle)[0]

        #plt.plot(cycles[mask], C_n[mask], marker=marker, color=color, markersize=markersize, label=label)
        if EoL == None:
            EoL = -1
        
        plt.plot(C_n[1:EoL], marker=marker, color=color, markersize=markersize, label=label)