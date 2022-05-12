from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pyabf as abf
import protocol_params as prm
import sys
import glob

np.set_printoptions(threshold=sys.maxsize)


def read_data(abf_data_file, patterns_data_file):
    abf_data = abf.ABF(abf_data_file)
    raw_data = []
    time_points = []
    print(f"sweepCount {abf_data.sweepCount}")
    for sn in range(abf_data.sweepCount):
        abf_data.setSweep(sweepNumber=sn)
        time_points.append(abf_data.sweepX)
        raw_data.append(abf_data.sweepY)

    patterns = np.load(patterns_data_file)
    return time_points, raw_data, patterns


def get_baseline(trace):
    """
    Baseline =  mean of values from pre-stim period
    """
    return np.mean(
        trace[
            int(0.25 * prm.PRE_STIM_DURATION * prm.SAMPLING_RATE) : int(
                prm.PRE_STIM_DURATION * prm.SAMPLING_RATE
            )
        ]
    )


def calc_amplitude(trace, stim_type):
    """
    Extracts the amplitude of response
    """

    trace_baseline_sub = trace - get_baseline(trace)
    # print(trace_baseline_sub)

    if stim_type == "elec":
        stim_start = prm.ELEC_STIM_START_TIME
    elif stim_type == "opt-paired":
        stim_start = prm.PAIRED_OPT_START_TIME
    elif stim_type == "opt-ind":
        stim_start = prm.IND_OPT_START_TIME

    return np.min(
        trace_baseline_sub[
            int(stim_start * prm.SAMPLING_RATE) : int(
                (stim_start + prm.RESPONSE_DURATION) * prm.SAMPLING_RATE
            )
        ]
    )


def identify_ppf_patterns(time, data, patterns):
    unique_patterns, unique_indices, inverse = np.unique(
        patterns, axis=0, return_index=True, return_inverse=True
    )
    paired_amplitudes = []
    ind_amplitudes = []
    ppf_patterns = []
    no_ppf_patterns = []

    for p, pattern in enumerate(unique_patterns):
        paired_amplitudes.append([])
        ind_amplitudes.append([])
        occurrence = np.where(inverse == p)[0]
        paired_occurrence = occurrence[occurrence % 2 == 0]
        ind_occurrence = occurrence[occurrence % 2 == 1]
        paired_sweep_numbers = np.array([int(o / 2) for o in paired_occurrence])
        ind_sweep_numbers = np.array([int(o / 2) for o in ind_occurrence])
        for sn in paired_sweep_numbers:
            paired_amplitudes[p].append(calc_amplitude(data[sn], "opt-paired"))
        for sn in ind_sweep_numbers:
            ind_amplitudes[p].append(calc_amplitude(data[sn], "opt-ind"))
        
        ppf_ratio = np.mean(paired_amplitudes[p] / np.mean(ind_amplitudes[p]))
        if ppf_ratio > prm.PPF_RATIO_THRESHOLD:
           ppf_patterns.append(pattern) 
        elif ppf_ratio >= prm.NO_PPF_RATIO_THRESHOLD and ppf_ratio<=1.0:
            no_ppf_patterns.append(pattern)

        print(p, pattern, ppf_ratio)
        # plt.boxplot(ind_amplitudes[p], patch_artist=False, notch="True")
        # break
    # bp_paired = plt.boxplot(paired_amplitudes, patch_artist=True, notch="True", vert=0)
    # print(ind_amplitudes)
    # print(bp_ind["boxes"])
    return ppf_patterns, no_ppf_patterns
    


def plot_traces(time_list, data_list, sweep_num=None):
    plt.figure()
    if sweep_num:
        plt.plot(time_list[sweep_num], data_list[sweep_num])
        plt.title(f"Sweep #{sweep_num}")
    else:
        for sn in range(len(data_list)):
            plt.plot(time_list[sn], data_list[sn])
    plt.show()


# time_points, raw_data, patterns = read_data()
# plot_traces(time_points, raw_data)
# identify_ppf_patterns(time_points, raw_data, patterns)
