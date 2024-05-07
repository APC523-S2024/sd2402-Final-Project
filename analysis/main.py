"""
Script for analyzing outputs from Collective Coordinate Optimization runs
Extracts final quantities (energy, enthalpy, prox metric, etc) from output files and averages
See bottom for booleans to set to measure each thing
Also implements loop for converting long SLURM output to N text files each containing an interm config

Sam Dawley
5/2024
"""
import os
import sys
import glob
import pathlib
import numpy as np

ABSOLUTE_TOLERANCE = 1e-20

############################################################
# BOILERPLATE
############################################################
def readfile(inpfile: str):
    """ Iterate over slurm output files and yield lines """
    with open(inpfile, "r") as f:
        for line in f:
            yield line
            
############################################################
# READING-IN DATA
############################################################
def readinterms(inpdir: str) -> str:
    """ Read through intermediate directories contained within parent directory """
    threedigits = lambda x: f"{int(x):03d}"
    dir_sorter = lambda x: threedigits(x.split("/")[-1].lstrip("Interm"))
    parent = glob.glob("Interm*", root_dir=inpdir)
    parent = sorted(parent, key=dir_sorter)
    for configdir in parent:
        filename = glob.glob("*-*", root_dir=configdir)[0]
        yield configdir + "/" + filename

def read_iterations(pathname: str, verbose: bool=True) -> np.array:
    """ Read through slurm output and collect number of iterations required to optimize each configuration """
    out_iters = []
    iters, success_indices = avg_iterations(pathname)
    for sidx, num_iterations in zip(success_indices, iters):
        out_iters.append(num_iterations)
        if verbose:
            print(f"Config. {sidx:<3d} required  {num_iterations:<8d} iterations")
    
    if verbose:
        avg_iters = np.mean(iters)    
        print(f"Avg. No. Iterations = {avg_iters:0.4f}")
    return np.asarray(out_iters)

############################################################
# SINGLE FILE/DIRECTORY ANALYSIS
############################################################
def avg_iterations(inpfile: str) -> tuple:
    """
    Read ccoptimization slurm output and return list of number of iterations required for optimization in each of M configs
    Counts iterations of succesful runs ONLY
    Returns
        tracking_iterations = list of num iterations required to reach ground state for successful optimizations
        success_indices = list of indices of configurations which succesfully reached the grounds state
    """
    tracking_iterations = []
    counter = 0
    config_counter = 0
    success_indices = []
    for line in readfile(inpfile):
        if "MultiRun start." in line:
            counter = 0
            config_counter += 1
        elif "(TOTAL) Relaxation complete." in line:
            iterations_line = line.strip("\n").rstrip("(TOTAL) Relaxation complete.").split("=")
            current_iterations = iterations_line[1]
            num_iterations = int(current_iterations)
            counter += num_iterations
        elif "Add to success pack(s)" in line:
            tracking_iterations.append(counter)
            success_indices.append(config_counter-1) # configs start counting at 0
    return tracking_iterations, success_indices

def final_stats(inpfile: str) -> np.array:
    """ Read ccoptimization slurm outputs and return list of number of iterations required for optimization in each config """
    E, V, H = [], [], []
    nn_dists = []
    prox_metrics = []
    for line in readfile(inpfile):
        if "Final E=" in line:
            temp = line.strip("\n").split(", ")
            stats = [float(s.split("=")[-1]) for s in temp]
            E.append(stats[0]) # energy
            V.append(stats[1]) # volume
            H.append(stats[2]) # enthalpy
        elif "Mean nearest-neighbor distance of initial state:" in line:
            stat_line = line.strip("\n").split(": ")[-1]
            rNN = float(stat_line.split("=")[-1])
            nn_dists.append(rNN)
        elif "Proximity metric (unscaled) between initial and final states:" in line:
            stat_line = line.strip("\n").split(": ")[-1]
            p = float(stat_line.split("=")[-1])
            prox_metrics.append(p)
            print(f"   Current Config E={E[-1]:<8.4e}  V={V[-1]:<7.1f} H={H[-1]:<8.4e} | r_NN={nn_dists[-1]:<8.4e} p={prox_metrics[-1]:<8.4e}")
    return np.asarray(E), np.asarray(V), np.asarray(H), np.asarray(nn_dists), np.asarray(prox_metrics), inpfile

def all_final_stats(inpdir: str) -> np.array:
    """ Read all ccoptimization slurm outputs contained within Interm* directories from parent directory """
    sidx = 0
    mE, mV, mH, filenames = [], [], [], []
    for intermfile in readinterms(inpdir):
        energy, volume, enthalpy, inpfile = final_stats(intermfile)
        me, mv, mh = np.mean(energy), np.mean(volume), np.mean(enthalpy)
        mE.append(me)
        mV.append(mv)
        mH.append(mh)
        filenames.append(inpfile)
        sidx += 1
    return mE, mV, mH, filenames

############################################################
# SLURM OUTPUT ANALYSIS
############################################################
def compare_outputs(inpdir: str, intermfile: str) -> tuple:
    """
    Get member of ensemble idx for all of initial, final, and specified intermediate index
    Note that ndarrays errors and success_indices are not necessarily the same length
        errors.size = M (number of members within ensemble)
        success_indices.size <= errors.size
    Args
        inpdir = parent directory containing total optimization and all intermediate directories
        intermfile = SLURM output of some known intermediate directory
    Returns
        errors = L2 errors between total optimization and partial within intermfile for all M configurations
        success_indices = configurational index of patterns within intermfile which are optimized to within specified energy tolerance
    """
    configSuccess = glob.glob("*-*.log", root_dir=inpdir)[0]
    total, partial = final_stats(configSuccess), final_stats(intermfile)
    etotal, _, _, _ = total
    epartial, _, _, _ = partial
    errors, success_indices = [], []
    for idx in range(etotal.size): # assume etotal.size == epartial.size
        a, b = etotal[idx], epartial[idx]
        L2error = np.hypot(a, b)
        errors.append(L2error)
        if L2error <= ABSOLUTE_TOLERANCE:
            success_indices.append(idx)
            print(f"{idx:<3d} {a:<10.4e}, {b:<10.4e} (error = {L2error:0.4e})")
        
    return np.asarray(errors), np.asarray(success_indices)

if __name__ == "__main__":
    inpdir = sys.argv[1]
    filename = glob.glob("*-*", root_dir=inpdir)[0]
    print(f"Reading from {filename}")
    get_iters = True
    get_stats = True
    gather_logs = False

    ########################################
    # SINGLE FILE/DIRECTORY ANALYSIS
    ########################################
    if get_iters: # Average number of iterations required to complete optimization
        print(f"Reading ITERATIONS data...")
        iters, success_indices = avg_iterations(inpdir + filename)
        avg_iters = np.mean(iters)
        var_iters = np.var(iters)
        for sidx, num_iterations in zip(success_indices, iters):
            print(f"   Config {sidx:<3d} required  {num_iterations:<8d} iterations")
        print(f"** Avg. No. Iters.   =  {avg_iters:<8.2f} over {len(success_indices):d} Configs. (from {inpdir + filename})")
        print(f"** Var(No. Iters) = {var_iters:<8.2f} and Std. Dev. = {np.sqrt(var_iters):<8.2f}")
        print()

    if get_stats: # Final values of the objective function following total optimization
        print(f"Reading STATISTICAL data...")
        a_energy, a_volume, a_enthalpy, a_files = all_final_stats(inpdir)
        energy, volume, enthalpy, nearest_neighbor_dists, proximity_metrics, filename = final_stats(filename)

        me, mv, mh = np.mean(energy), np.mean(volume), np.mean(enthalpy)
        mrNN, mp = np.mean(nearest_neighbor_dists), np.mean(proximity_metrics)
        print(f"** MEAN OPTIMIZED E={me:0.4e}  V={mv:0.1f}  H={mh:0.4e} | r_NN={mrNN:0.4e} p={mp:0.4e} (from {inpdir + filename})")

        counter = 0
        for ae, av, ah, f in zip(a_energy, a_volume, a_enthalpy, a_files):
            counter += 5
            idx = f.split("/")[0].lstrip("Interm")
            print(f"({f:<33s} <=> {idx:3s}) - E={ae:<10.4e}, V={av:<6.1f}, H={ah:<10.4e}")