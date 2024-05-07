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

############################################################
# BOILERPLATE
############################################################
def readfile(inpfile: str):
    """ Iterate over slurm output files and yield lines """
    with open(inpfile, "r") as f:
        for line in f:
            yield line

############################################################
# CUSTOM CLASSES
############################################################
class slurmParser:
    def __init__(self, this_inpfile: str, outdir: str, new_timesteps: float=1e3) -> None:
        self.this_input_file = this_inpfile
        self.outdir = outdir
        # self.num_particles = N
        self.input_path = pathlib.Path("./" + self.this_input_file)
        # print(self.input_path.exists())
        self.time_steps = int(new_timesteps)

        # test that output directory exists, otherwise create it
        if not os.path.isdir(outdir):
            print(f"Directory for storing outputs does not exist... Creating directory ./{outdir}")
            os.mkdir(outdir)

        # create dictionary for storing members of ensemble
        self.configs = {}

    def __repr__(self):
        return f"Input file is {self.this_input_file}\nContaining N = {self.num_particles}\nStoring outputs in directory ./{self.outdir}"

    ##### BOILERPLATE #####
    def _wtf(self):
        """ Iterate over slurm output files and yield lines """
        with open(self.this_input_file, "r") as f:
            for line in f.readlines():
                yield line

    # def _padding_routine(self, arr: np.array, mode: str="constant", vals: list=[1e6]):
    #     """ Padding ~shenanigans~ to ensure each array is proper size for ensemble_* arrays """
    #     if arr.size < 2*self.num_particles:
    #         nbqs = abs(2*self.num_particles - arr.size)
    #         return np.pad(arr, (0, nbqs), mode, constant_values=vals)
    #     return arr
    
    ##### PARSING DATA #####
    def _collect_intermediates(self) -> None:
        """
        Collect intermediate states from a single configuration
        No return value but populates the dictionary self.configs = {(key: val)} with
        (key: val) pairs member_idx: (intermediate_arrays, number_snapshots, intermediate_statistics)
            intermediate_arrays     = (I, N, d) ndarray containing coordinates of each snapshot
            number_snapshots        = interger number of snapshots recorded over optimization = I
            intermediate_statistics = iteration, energy, and stepsizes for each I of this member
        """
        recording = False
        midx = -1 # ensemble member idx
        iidx = -1 # configurational snapshot idx
        temp_coords = []
        intermediates = []
        intermediate_stats = []

        for line in self._wtf():
            if recording:
                L = line.strip("\n")
                try:
                    coord = float(L)
                    temp_coords.append(coord)
                except ValueError:
                    print(f"(Member {midx}) Attempted to convert '{L}' to <float>. Ignoring...")      
            
            if ",CONFIG=" in line:
                recording = True

                # record intermediate values of iteration, energy, and step size
                temp_sl = []
                stat_line = line.strip().split(",")
                sl_iter, sl_objective, sl_ss, _ = stat_line # (iteration number, objective function val, step size, 'CONFIG=')
                for stat in [sl_iter, sl_objective, sl_ss]:
                    sl = float(stat.split("=")[-1])
                    temp_sl.append(sl)
                intermediate_stats.append(temp_sl)

                # record intermediate coordinates (the thing we care about most)
                if iidx >= 0:
                    temp_coords = np.asarray(temp_coords)
                    # coord_array = self._padding_routine(temp_coords)
                    # coord_array = coord_array.reshape((self.num_particles, 2))
                    coord_array = temp_coords.reshape((int(temp_coords.size/2), 2))

                    # if coord_array.shape == (self.num_particles, 2):
                    intermediates.append(coord_array)
                    temp_coords = []

                    # else:
                        # print(f"Failed to add coordinate array {iidx} to intermediate states")

                iidx += 1

            if "MultiRun start." in line:
                # record intermediate coordinates (the thing we care about most)
                if iidx >= 0:
                    temp_coords = np.asarray(temp_coords)
                    coord_array = temp_coords.reshape((int(temp_coords.size/2), 2))
                    intermediates.append(coord_array)
                    temp_coords = []

                if midx >= 0:                    
                    # self.configs[midx] = (np.asarray(intermediates), I, np.asarray(intermediate_stats))
                    self.configs[midx] = (intermediates, len(intermediates), np.asarray(intermediate_stats))

                    intermediates = []
                    intermediate_stats = []

                midx += 1
                iidx = -1
        
        return

    ##### WRITING DATA #####
    def _write_interm2txt(self, intermediates_array: np.array, member_idx: int) -> None:
        """
        Write intermediate states to individual text files
        Args
            intermediates_array = that returned by self._collect_intermediates
            member_idx = idx of member of ensemble
        """
        fix2digits = lambda x: f"{int(x):02d}"
        fix4digits = lambda x: f"{int(x):04d}"

        midx = fix2digits(member_idx)

        I = len(intermediates_array)
        # print(f"Shape of intermediate array is {intermediates_array.shape}")
        print(f"Intermediate array contains {I} snapshots")
        
        for snap_idx in range(I):
            interm_array = intermediates_array[snap_idx]
            iidx = fix4digits(snap_idx)
        
            with open(self.outdir + f"/config_{midx}-snap_{iidx}.txt", "w") as f:
                for coord in interm_array:
                    x, y = coord
                    f.write(f"{x:0.6e},{y:0.6e}\n")

        return

    def _write_stats2txt(self, intermediates_array: np.array, member_idx: int) -> None:
        """
        Write intermediate states to individual text files
        Args
            intermediates_array = that returned by self._collect_intermediates
            member_idx = idx of member of ensemble
        """
        fix2digits = lambda x: f"{int(x):02d}"

        midx = fix2digits(member_idx)

        I = len(intermediates_array)
        print(f"Intermediate array contains {I} snapshots")
        
        with open(self.outdir + f"/config_{midx}-statistics.txt", "w") as f:
            for snap_idx in range(I):
                iteration, objective_function, step_size = intermediates_array[snap_idx]

                f.write(f"{iteration:0.6e},{objective_function:0.6e},{step_size:0.6e}\n")
                
        return
        
    def _write_timescales2txt(self, num_snapshots: int, member_idx: int) -> None:
        """
        Write time normalized time scales for each member of ensemble to output file for reading later
        Args
            num_snapshots = integer number of snapshots recorded in optimizing this member of the ensemble
            timesteps = number of steps to discretize each snapshot to
                'timesteps' will be the same for all members of the ensemble and 
                therefore provides a method of corresponding snapshots from different members of this ensemble
            member_idx = integer representing which member of the ensemble this is
        """
        fix2digits = lambda x: f"{int(x):02d}"

        midx = fix2digits(member_idx)

        time_domain  = np.linspace(0, 1, self.time_steps)
        indexing_set = np.linspace(0, num_snapshots-1, self.time_steps, dtype=np.intc) # make sure these are always integers

        with open(self.outdir + f"/config_{midx}-timescales.txt", "w") as f:
            for tindex in range(time_domain.size):
                time = time_domain[tindex]
                snap = indexing_set[tindex]

                f.write(f"{snap:d},{time:0.6e}\n")

        return

    def write_all2text(self, write_coordinate_output: bool=True, write_statistical_output: bool=True, write_timescale_output: bool=True) -> None:
        """ Write all intermediate outputs to text files """
        self._collect_intermediates() # COLLECT DATA HERE

        for member_index, interm_statistics in self.configs.items():
            interm_array, interm_array_length, interm_stats = interm_statistics # IMPORTANT THAT WE UNPACK THE TUPLE FROM DICTIONARY

            if write_coordinate_output:
                print(f"Writing intermediate COORDINATE  results of member {member_index} to txt files...")
                self._write_interm2txt(interm_array, member_index)

            if write_statistical_output:
                print(f"Writing intermediate STATISTICS  results of member {member_index} to txt files...")
                self._write_stats2txt(interm_stats, member_index)

            if write_timescale_output:
                print(f"Writing intermediate TIME SCALES results of member {member_index} to txt files...")
                self._write_timescales2txt(interm_array_length, member_index)
    
        print("Done!")
        return


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
            iterations_line = line.strip("\n").rstrip(" iterations.").split()[-1]
            num_iterations = int(iterations_line)
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
        

############################################################
# ACOUSTIC KITTY
############################################################
def readfile(inpfile: str):
    """ Iterate over slurm output files and yield lines """
    with open(inpfile, "r") as f:
        for line in f:
            yield line

def coords_from_txt(inpfile: str) -> np.array:
    """ Read COORDINATES from intermediate configurational text files """
    coordinates = []
    for line in readfile(inpfile):
        L = line.strip("\n").split(",")
        coordinates.append([float(a) for a in L])
    return np.asarray(coordinates)

def stats_from_txt(inpfile: str) -> np.array:
    """ Read STATISTICS from intermediate statistical text files """
    stats = []
    for line in readfile(inpfile):
        L = line.strip("\n").split(",")
        iteration, objective_func, step_size = [float(a) for a in L]
        stats.append([iteration, objective_func, step_size])
    return np.asarray(stats)

def timescales_from_text(inpfile: str) -> np.array:
    """
    Read TIME SCALES from intermediate timescale files
    Returns
        ts = (I, 2) ndarray with elements [snapshot_index, normalized_time]
    """
    ts = []
    for line in readfile(inpfile):
        L = line.strip("\n").split(",")
        tidx, time = int(L[0]), float(L[1])
        ts.append([tidx, time])
    return np.asarray(ts)

def trajectory_arc_lengths(trajectories: np.array, dt: int=1) -> np.array:
    """
    Compute arc length of each trajectory in ensemble
    Args
        trajectories = (N, I, d) ndarray of all N particle trajectories across I snapshots in d dimensions
    Returns
        (N, 1) array with arc lengths of all N particles given specific simulation trajectory
    """
    arcs = []
    for t in trajectories:
        dxdy = np.diff(t, axis=0)
        dx = dxdy[:, 0] / dt
        dy = dxdy[:, -1] / dt
        dz = 0
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        arcs.append(sum(ds))
    return np.asarray(arcs)

def acoustic_kitty(inpdir: str, M: int, d: int) -> tuple:
    """ 
    yktv
    Args
        inpdir = input directory containing txt files to all intermediate configurations
        M = integer number of configurations to ensemble average within each directory (parent + intermediates)
        N = number of particles in each configuration, must be >= most particles within a single configuration
        d = dimensionality
    Returns
        All return values are dictionarys with (key: val) pairs (member_index: ndarray) : ndarray varies in shape
        ensemble_patterns     = ... ndarray.shape = (I, N, d) with configs for all I snapshots
        ensemble_trajectories = ... ndarray.shape = (N, I, d) with trajectories for all N particles through I snapshots
        ensemble_arcs         = ... ndarray.shape = (N, 1)    with (scalar) arc lengths for all N particles
            Note that I = number of snapshots over course of optimization for that member
    """    
    if inpdir[-1] != "/": inpdir += "/"
        
    # all used for sorting text files based on configuration index
    # MAKE SURE fixNdigits() WORKS FOR APPLICATION, i.e., CHECK DIGITS
    fix2digits = lambda x: f"{int(x):02d}" 
    fix4digits = lambda x: f"{int(x):04d}" 
    config_sorter = lambda name: fix4digits(name.rstrip(".txt").split("_")[-1])

    # Where the sausage gets made
    ensemble_statistics = {}
    ensemble_patterns = {}
    ensemble_trajectories = {}
    ensemble_arcs = {}
    
    # indexed by 'midx' : iterate over all members of the ensemble
    # indexed by 'cidx' : iterate over all intermediate configurations for specific member
    print(f"Reading from")
    for member_index in range(M):
        print(f"Member {member_index}")
        midx = fix2digits(member_index)
        
        # READING STATISTICAL DATA
        member_statistics = glob.glob(inpdir + f"config_{midx}-statistics.txt")
        if not len(member_statistics):
            print(f"Could not get statistics file corresponding to member {midx} of ensemble. Aborting member...")
            continue
        elif len(member_statistics) != 1:
            print(f"Found multiple statistics files corresponding to member {midx} of ensemble. Aborting member...")
            continue
        else:
            stats_file = member_statistics[-1]

        # READING TIMESCALE DATA
        member_timescale = glob.glob(inpdir + f"config_{midx}-timescales.txt")
        if not len(member_timescale):
            print(f"Could not get time scales file corresponding to member {midx} of ensemble. Aborting member...")
            continue
        elif len(member_timescale) != 1:
            print(f"Found multiple time scales files corresponding to member {midx} of ensemble. Aborting member...")
            continue
        else:
            timescale_file = member_timescale[-1]

        # READING COORDINATE DATA
        member_configurations = glob.glob(inpdir + f"config_{midx}-snap*")
        member_configurations = sorted(member_configurations, key=config_sorter) 
        
        # gather statistics first so we know how many intermediate configs each member has
        statistics = stats_from_txt(stats_file)
        # I = statistics.shape[0] - 1 # number of snapshots this member has EXCLUDING GROUND STATE

        # now gather relevant time scale
        timescale = timescales_from_text(timescale_file)
        I = timescale.shape[0] # NORMALIZED number of snapshots to include

        # now gather coordinates of intermediate configurations
        # the number of particles may vary between members (viz. saturated RSA) so we must get N for each member
        # once we have this member's N it's used as a dict key before converting dict to ndarray
        # followed by reshaping ndarray so we have well-defined trajectories again
        d_patterns = {}
        for cidx, _ in timescale:
            cidx = int(cidx) # this will always be an integer don't worry; gets changed to float when timescale is returned as ndarray

            interm_config_file = member_configurations[cidx]
            interm_coordinates = coords_from_txt(interm_config_file)
            num_particles = interm_coordinates.shape[0]

            d_patterns[cidx] = interm_coordinates
            
        patterns = np.zeros((I, num_particles, d))
        for cidx, _ in timescale:
            cidx = int(cidx) # this will always be an integer don't worry; gets changed to float when timescale is returned as ndarray
            patterns[cidx] = d_patterns[cidx]

        trajectories = patterns.transpose((1, 0, 2))

        # finish by computing arc lengths of each particle for this member of the ensemble
        arcs = trajectory_arc_lengths(trajectories, dt=1)

        ensemble_statistics[member_index] = statistics
        ensemble_patterns[member_index] = patterns
        ensemble_trajectories[member_index] = trajectories
        ensemble_arcs[member_index] = arcs

        print(ensemble_patterns[member_index].shape)

    print("Done!")
    return ensemble_patterns, ensemble_trajectories, ensemble_arcs

if __name__ == "__main__":
    try:
        inpdir = sys.argv[1]
    except IndexError:
        inpdir = f"/home/sd2402/scratch-sd2402/acoustic_kitty/rsa/RSA-2D-N_1000-rho_1.0_chi_0.4/"
    filename = glob.glob("*-*", root_dir=inpdir)[0]
    # filename = sys.argv[2]
    print(f"Reading from {filename}")
    get_iters = True
    get_stats = True
    gather_logs = False
    slurm2txts = True
    ak_testing = False

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

    ########################################
    # SLURM OUTPUT ANALYSIS
    ########################################
    if gather_logs:
        print(f"Gathering .log files...")
        for intermediate_output in readinterms(inpdir):
            intermdir = intermediate_output.split("/")[0]
            print(f"Reading from {inpdir + intermdir}")
            errors, successful = compare_outputs(inpdir, intermediate_output)

    ########################################
    # SINGLE FILE/DIRECTORY ANALYSIS
    ########################################
    if slurm2txts:
        print(f"Convering SLURM output to intermediate text files...")
        theParser = slurmParser(inpdir + filename, "intermediate_outputs", new_timesteps=5e2)
        theParser.write_all2text(write_coordinate_output=True, write_statistical_output=True, write_timescale_output=True)

    if ak_testing:
        packing = "poisson/POISSON-2D-N_1024-phi_0.54-R_0.4146-chi_0.02"
        inpdir = f"/home/sd2402/scratch-sd2402/acoustic_kitty/{packing}/TESTING/trial_2/intermediate_outputs"
        M = 49 
        N = 1024
        d = 2

        ep, et, ea = acoustic_kitty(inpdir, M, d)