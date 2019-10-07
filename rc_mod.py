#Filename: rc_mod.py
#Description: radar signal modulation - module
#Author: Adrian Adams
#Version: 2.0
#Date: 30/09/2019
#List of functions: linear frequency modulation, barker, rect, pulse generation

#rc_mod.py
import math

# Function: Linear frequency modulation and Rectangular (No modulation)
def freq_mod( freq_sweep_up,freq_param, pulse_width_fm, pulse_width_counter_position, pulse_flag,ts,t_count):

    if pulse_flag == 1:
         f_mod_val = freq_sweep_up*freq_param/(pulse_width_fm*ts)/2
    else:
        f_mod_val = 0

    return f_mod_val

def amp_mod(amp_val,ampmod_param,index,ts):
    amp_mod_val = amp_val * math.sin(2 * math.pi * ampmod_param * index * ts)
    return amp_mod_val

def barker_mod(barker_index, barker_code):
    phi_mod_val = barker_code[barker_index]
    #print(barker_code,barker_index, barker_code[barker_index])
    return phi_mod_val


# Function: Generates radar pulse train with specified pulse code modulation
def radar_pulse(frequency, PRI, pulse_width, samples,t,pulse,radar_counter, radar_freq_sweep_up,radar_freq_mod_param, radar_ampmod_param, radar_amp_val,radar_mod, Debug_flag, radar_barker_val):
    if Debug_flag == 1:
        print("Radar counter: ", radar_counter)
        print("PRI: ", PRI)
        print("Pulse width: ", pulse_width)
        print("Pulse width samples:", int(pulse_width/t))
        print("Frequency: ", frequency)
        print("Samples: ", samples)
        print("T: ", t)
        print("Pulse: ", pulse)
        print("Radar freq sweep up", radar_freq_sweep_up)
        print("Radar freq mode param", radar_freq_mod_param)
        print("Radar amp mod param", radar_ampmod_param)
        print("Radar mod type",radar_mod)
        #radar_barker_val = [1, 0, 1, 0]
        print("Radar barker code val", radar_barker_val)
        print("Length of barker", len(radar_barker_val))

    phi = 0
    barker_code_index = 0
    phi_barker = []
    label_vec_signal = []
    pulse_width_samples = int(pulse_width/t)
    time_vec = []
    waveform_sum = []
    pulse_vector = []
    pulse_width_counter = 0
    for x in range(1, samples):
        if (pulse == 0) and (x % int(PRI/t) == 0) :
            pulse = 1
            pulse_width_counter = 0
            barker_code_index = 0
            #print("A: Barker code index", barker_code_index)
            #print("A - reset Barker index at start of pulse", barker_code_index)
        elif (pulse == 1) and ((x - int(pulse_width/t)) % int(PRI / t) == 0):
            pulse = 0
            pulse_width_counter = pulse_width_counter
            barker_code_index = 0
            #print("D - outside of pulse Barker index is back at 0")
            #print("D: Barker code index", barker_code_index)
        else:
            pulse = pulse
            if (pulse==1) and pulse_width_counter % (int(pulse_width_samples/len(radar_barker_val))) == 0 and (barker_code_index != len(radar_barker_val)-1):
                barker_code_index = barker_code_index + 1
                #print("B - increment Baker index at next cycle in pulse", pulse_width_samples,pulse_width_samples/ len(radar_barker_val), pulse_width_counter, pulse_width_counter/len(radar_barker_val), pulse_width_counter % len(radar_barker_val), barker_code_index)
                #print("B: Barker code index", barker_code_index)
            else:
                barker_code_index = barker_code_index
                #print("C: Barker code index", barker_code_index)
                #print("C - keep Barker index the same in pulse until next increment", barker_code_index, pulse_width_counter)
        if (pulse == 1):
            pulse_width_counter = pulse_width_counter + 1
        else:
            pulse_width_counter = 0

        pulse_vector.append(pulse)

   # for x in range(1, samples):
        if radar_mod == 0:   # Rectangular (Fixed frequency)
            temp_cos = pulse*math.cos(2*math.pi*(frequency*x*t + phi))
            temp_sin = pulse*math.sin(2*math.pi*(frequency*x*t + phi))
            phi_barker.append(0)
            label_vec_signal.append(0)
        elif radar_mod == 1: # Linear FM
            temp_cos = pulse*math.cos(2*math.pi*(frequency * x * t + freq_mod(radar_freq_sweep_up, radar_freq_mod_param, pulse_width_samples, pulse_width_counter, pulse, t, x)*((pulse_width_counter*t)**2) + phi))
            temp_sin = pulse*math.sin(2*math.pi*(frequency * x * t + freq_mod(radar_freq_sweep_up, radar_freq_mod_param, pulse_width_samples, pulse_width_counter, pulse, t, x)*((pulse_width_counter*t)**2) + phi))
            phi_barker.append(0)
            label_vec_signal.append(1)
        elif radar_mod == 2: # Amplitude Modulation
            temp_cos = amp_mod(radar_amp_val,radar_ampmod_param,x,t)*pulse*math.cos(2*math.pi*(frequency*x*t + phi))  # Need to figure out if this is correct for AM modulation
            temp_sin = amp_mod(radar_amp_val,radar_ampmod_param,x,t)*pulse*math.sin(2*math.pi*(frequency*x*t + phi))  # Need to figure out if this is correct for AM modulation
            phi_barker.append(0)
            label_vec_signal.append(2)
        elif radar_mod == 3: #Barker code
            temp_cos = pulse * math.cos(2 * math.pi * (frequency * x * t) + math.pi*barker_mod(barker_code_index, radar_barker_val))
            temp_sin = pulse * math.sin(2 * math.pi * (frequency * x * t) + math.pi*barker_mod(barker_code_index, radar_barker_val))
            phi_barker.append(barker_mod(barker_code_index, radar_barker_val))
            label_vec_signal.append(3)
        else:
            temp_cos = 0
            temp_sin = 0
            phi_barker.append(0)
            label_vec_signal.append(-1)

        waveform_sum.append(temp_sin + 0*temp_cos)
        time_vec.append(x)
    return waveform_sum, pulse_vector, time_vec, phi_barker, label_vec_signal