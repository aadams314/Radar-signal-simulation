#Filename: radar_model_m.py
#Description: Main program - to simulate multiple radar signal returns to generate data for machine learning problem of EM spectrum classification
#Author: Adrian Adams
#Date: 30/09/2019
#Ver: 2.0

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import time
import random
import rc_mod # module containing radar pulse generation and pulse code modulation functions



from scipy.io.wavfile import write


# Config simulation variables section
Debug_flag_signal = 0  # 1 is debug notes, 0 is debug notes off
Graph_flag = 0  # 1 shows each signal independently, 0 shows a composite
Spec_graph_flag = 1 # 1 to show spectrogram plot, 0 to not show it
FFT_graph_flag = 1
Save_data_flag = 1 # 1 saves the data to file, 0 does not save the data
Radar_param_fixed_options = 1
# Debug variables section

start_time = time.time()

#Simulation parameters
f_sample = 1e6
t_sample = 1 / f_sample
t_simulation = .2 # simulation time
no_samples = int(t_simulation / t_sample)
print("Sampling frequency = %f MHz" % (f_sample/1e6))
print(f_sample/1e6)
print("Sampling period = %8.20f microseconds" % (t_sample/1e-6))
print(t_sample)
print(type(t_sample))
print("Simulation time = %f seconds" % (t_simulation))
print(t_simulation)
print("Number of samples = %f samples" % (no_samples))
print(no_samples)
# End Simulation parameters section

nse_val = 0.10 #noise value

Barker_code_min = 3
Barker_code_max = 7


NFFT = 1024
Fs = int(1.0/t_sample)
Nov_win = 900 # no overlap window


random.seed()
#Radar parameters



if Radar_param_fixed_options == 0:
    # random radar parameters
    print("Simulation: Random number of radars, random params")
    Min_pulse_width = t_sample*1
    Max_pulse_width = t_sample*no_samples-t_sample
    Min_freq = 1
    Max_freq = int(f_sample/2)
    Max_ampmod_freq = int(0.00001*f_sample)
    Min_PRI = 100*t_sample
    Max_PRI = t_sample*no_samples/2
    Number_of_radars = random.randint(50,100)
    Radar_pulse_width = [(Max_pulse_width*random.random()+Min_pulse_width) for pw_i in range(Number_of_radars)]   # seconds
    Radar_frequency = [(Max_freq*random.random()) for freq_i in range(Number_of_radars)]    # Hz
    #Radar_freq_mod_flag = [int(round(random.random())) for fm_i in range(Number_of_radars)]  #
    Radar_freq_mod_flag_sweep_up = [int(2*round(random.random()))-1 for fm_i in range(Number_of_radars)]  #
    Radar_freq_mod_param = [Max_freq*random.random()/4 for fmp_i in range(Number_of_radars)]
    Radar_amplitude = [random.random() for amp_i in range(Number_of_radars)]
    #Radar_amp_mod_flag = [int(round(random.random())) for famf_i in range(Number_of_radars)]
    Radar_amp_mod_param = [Max_ampmod_freq*random.random() for famp_i in range(Number_of_radars)]
    Radar_CW_Pulse = [1 for CWP_i in range(Number_of_radars)]  # 0 is CW, 1 is pulsed
    Radar_PRI = [random.random()*Max_PRI for PRI_i in range(Number_of_radars)]          # seconds
    Radar_mod_type = [random.randint(0,3) for mod_i in range(Number_of_radars)]
    Radar_barker_code = [[int(round(random.random())) for bc_i in range(random.randint(Barker_code_min,Barker_code_max))] for bc_k in range(Number_of_radars)]
    print(Radar_pulse_width)
    print(Radar_frequency)
    print(Radar_amplitude)
    print(Radar_CW_Pulse)
    print(Radar_PRI)
    print(Radar_mod_type)
    #end of random radar parameters
elif Radar_param_fixed_options == 1:
    #defined radar parameters
    print("Simulation: 5 radars, fixed params")
    Number_of_radars = 5
    Radar_pulse_width = [50e-3, 5e-3, 10e-3, 7e-3, 20e-3]   # seconds
    Radar_frequency = [100000, 250000, 400000, 450000, 2000]    # Hz
    # Radar_freq_mod_flag = [1, 1, 0, 1, 0]  # 1 equal linear modulatiom
    Radar_freq_mod_flag_sweep_up = [1, -1, 1, -1, 1]  #
    Radar_freq_mod_param = [0, 5000, 5000, 4900, 1100]
    Radar_amplitude = [.5, 1, 0.5, 0.3, 0.6]
    #Radar_amp_mod_flag = [1, 0, 1, 0, 1]
    Radar_amp_mod_param = [20, 2500, 40, 20000, 200]
    Radar_CW_Pulse = [1, 1, 1, 1, 1]  # 0 is CW, 1 is pulsed
    Radar_PRI = [50e-3, 66e-3, 100e-3, 10e-3, 40e-3]          # seconds
    Radar_mod_type = [0, 1, 2, 1, 3]
    Radar_barker_code = [[1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0,1, 0, 1]]
    #end of defined radar parameters
elif Radar_param_fixed_options == 2:
    #defined radar parameters
    print("Simulation: 1 radars, fixed params")
    Number_of_radars = 1
    Radar_pulse_width = [50e-3]   # seconds
    Radar_frequency = [100]    # Hz
    # Radar_freq_mod_flag = [1, 1, 0, 1, 0]  # 1 equal linear modulatiom
    Radar_freq_mod_flag_sweep_up = [1]  #
    Radar_freq_mod_param = [0]
    Radar_amplitude = [.5]
    #Radar_amp_mod_flag = [1, 0, 1, 0, 1]
    Radar_amp_mod_param = [20]
    Radar_CW_Pulse = [1]  # 0 is CW, 1 is pulsed
    Radar_PRI = [50e-3]          # seconds
    Radar_mod_type = [3]
    #end of defined radar parameters
    Radar_barker_code = [[1, 0, 1, 0]]
else:
    # defined radar parameters
    print("Simulation: Default 1 radars, default fixed params")
    Number_of_radars = 0
    Radar_pulse_width = []  # seconds
    Radar_frequency = []  # Hz
    # Radar_freq_mod_flag = [1, 1, 0, 1, 0]  # 1 equal linear modulatiom
    Radar_freq_mod_flag_sweep_up = []  #
    Radar_freq_mod_param = []
    Radar_amplitude = []
    # Radar_amp_mod_flag = [1, 0, 1, 0, 1]
    Radar_amp_mod_param = []
    Radar_CW_Pulse = []  # 0 is CW, 1 is pulsed
    Radar_PRI = []  # seconds
    Radar_mod_type = []
    # end of defined radar parameters
    # End Radar Parameters

temp_waveform_sum_tot=np.zeros(no_samples)
temp_pulse_vector_tot = np.zeros(no_samples)
temp_t_vec = np.zeros(no_samples)

waveform_sum_tot = np.zeros(no_samples)
pulse_vector_tot =np.zeros(no_samples)
t_vec = np.zeros(no_samples)
label_vec_fin_n = np.zeros([Number_of_radars,no_samples-1])


#pulse_vector = []
phi_vec = []
temp_sin = 0
temp_cos = 0
label_vec = []

#no_sample = 10000

#if Graph_flag == 0:
#    plt.figure(figsize=(12, 6))


for no_radar_count in range(0,Number_of_radars):
 #   f.append(no_radar_count)
    print("Start Radar sim: ", no_radar_count, Number_of_radars)
    pulse_flag = 0
    waveform_sum_tot, pulse_vector_tot,t_vec,phi_vec,label_vec = rc_mod.radar_pulse(Radar_frequency[no_radar_count], Radar_PRI[no_radar_count], Radar_pulse_width[no_radar_count], no_samples, t_sample, pulse_flag,no_radar_count,Radar_freq_mod_flag_sweep_up[no_radar_count], Radar_freq_mod_param[no_radar_count], Radar_amp_mod_param[no_radar_count], Radar_amplitude[no_radar_count],Radar_mod_type[no_radar_count],Debug_flag_signal, Radar_barker_code[no_radar_count])
    temp_waveform_sum_tot = [waveform_sum_tot[i] + temp_waveform_sum_tot[i] for i in range(len(waveform_sum_tot))]
    temp_pulse_vector_tot = [pulse_vector_tot[j] + temp_pulse_vector_tot[j] for j in range(len(pulse_vector_tot))]
    temp_t_vec = t_vec #+ temp_t_vec
    label_vec_ar=np.array(label_vec)
    pulse_vec_ar = np.array(pulse_vector_tot)
    label_vec_fin = np.multiply(label_vec_ar,pulse_vec_ar)
    print(label_vec)
    print(pulse_vector_tot)
    print(label_vec_fin)
    label_vec_fin_n[no_radar_count,:] = label_vec_fin
    print("Type label_vec_fin_n", type(label_vec_fin_n))
    print("Type label_vec_fin", type(label_vec_fin))
    print("Size of label_vec_fin_n: ", label_vec_fin_n.shape)
    print("Size of label_vec_fin: ", len(label_vec_fin))
    print("End of Radar sim: ", no_radar_count, Number_of_radars,"Pulse width: ",Radar_pulse_width[no_radar_count],"Modulation type: ", Radar_mod_type[no_radar_count])
    if Graph_flag == 1:
        plt.figure(figsize=(12,6))
        plt.plot(t_vec, pulse_vector_tot)
        plt.plot(t_vec, waveform_sum_tot)
        print(len(phi_vec))
        print(len(t_vec))
        plt.plot(t_vec, phi_vec)
        plt.plot(t_vec,label_vec_fin)
        plt.title('Wave vs time')
        plt.xlabel('time')
        plt.ylabel('amplitude')
        plt.show()

print(phi_vec)
print(max(phi_vec))
print(min(phi_vec))

#if Graph_flag == 0:
#    plt.show()

stop_time = time.time()
print("--- %s microseconds ---" % ((stop_time-start_time) / 1e-6))
print("--- %s milliseconds ---" % ((stop_time-start_time) / 1e-3))
print("--- %s seconds ---" % (stop_time-start_time) )

# Adds noise
noise = nse_val*np.random.random(size=len(temp_t_vec))
x = temp_waveform_sum_tot + noise

plt.figure(figsize=(12, 6))
plt.plot(temp_t_vec, temp_pulse_vector_tot)
plt.plot(temp_t_vec, x)
plt.title('Wave vs time + noise')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()
print("Generating time domain plot of waveforms")

#Save data to file
if Save_data_flag == 1:
    #data_vec_gen =  [label_vec_fin_n[nr3,:].tolist() for nr3 in range(Number_of_radars)]
    np.savetxt('Wave.dat',[temp_t_vec, x,temp_pulse_vector_tot, phi_vec] + [label_vec_fin_n[nr2,:].tolist() for nr2 in range(Number_of_radars)])
    print("temp_t_vec",len(temp_t_vec))
    print("x", len(x))
    print("pulse_vector_tot", len(pulse_vector_tot))
    print("phi_vec",len(phi_vec))
    data = np.column_stack([temp_t_vec, x, pulse_vector_tot, phi_vec] + [label_vec_fin_n[nr3,:].tolist() for nr3 in range(Number_of_radars)])
    np.savetxt('Wavec.dat', data)
    print("Saving data to file")

# Plot spectrogram
if Spec_graph_flag == 1:
    print("Plotting spectrogram")
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(temp_t_vec, x)
    #plot.xlabel('samples')
    ax1.set_ylabel('amplitude')
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=Nov_win)
    #ax2.xlabel('samples')
    ax2.set_ylabel('Frequency (Hz)')
    plt.show()

# plot FFT
sample_win_start = 0
sample_win_stop = 10000
win_len = sample_win_stop - sample_win_start
win_overlap_ratio = 0.8

fig = plt.figure()
plt.axis([0, win_len, 0, 1])
for fft_win_loop in range(0,no_samples-1,int(win_len*win_overlap_ratio)):
    print("Plotting FFT")
    print("fft_win_loop: ",fft_win_loop)
    print("no_samples - 1: ", no_samples - 1)
    print("win_len: ",win_len)
    print("win_overlap_ratio: ", win_overlap_ratio)
    sample_win_start = fft_win_loop
    sample_win_stop = fft_win_loop+win_len
    print("sample_win_start: ", sample_win_start)
    print("sample_win_stop: ", sample_win_stop)
    if FFT_graph_flag == 1:
        x_fft = np.fft.fft(x[sample_win_start:sample_win_stop])/win_len
        x_fft = x_fft[range(int(win_len/2))]
        #plt.figure(figsize=(12, 6))
        plt.plot(abs((x_fft)))
        plt.title('FFT')
        plt.xlabel('Frequency')
        plt.ylabel('amplitude')
        plt.pause(0.5)
        fig.clear()
plt.show()












