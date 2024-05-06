#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=ccopt-testing
#SBATCH --output=./%A-%a.log
#SBATCH --mail-type=end
#SBATCH --mail-user=sd2402@princeton.edu

i=${SLURM_ARRAY_TASK_ID}
# #SBATCH --output=/home/sd2402/scratch-sd2402/acoustic_kitty/rsa/RSA-2D-N_1000-rho_1.0_chi_0.02/%A-%a.log

# COMMAND LINE ARGUMENTS
timelimit=48 # time limit in hours. The code will stop and save the progress at the time limit; very important if you simulate with MD
tolerance=1e-17	# tolerance for the ground-state energy
max_evl=1000000 # maximum number of evaluations
seed=2960015 # random seed for initial conditions.

# IMPORTANT COMMANDS
d=2 # dimensionality

K1a=0.0 # minimum wavenumber in the dimensionless exclusion region (0 implies hyperuniformity)

# (exclusion zone)x(individual particle radius) <=> K=1 corresponds to Ka=a
Ka=0.414593 # chi=0.02
# Ka=0.929516 # chi=0.1

packing_frac=0.54 # USED FOR CALCULATING PARTICLE RADIUS
N=1000 # particle number
Nc=50 # the number of configurations
S0=0.0 # value of the structure factor for K1a < ka < Ka (0 means stealthy hyperuniformity.)
val=1.0 # strength of the soft-core repulsion <=> hard core applies to val \geq \phi\times10^{\phi}, i.e., choose val=1 for hard cores (0 implies no repulsion <=> nonzero 'val' results in hard code particles)
sigma=1 # ${list_sigma[$i]} # exclusion diameter of the soft-core repulsion - RELEVANT ONLY WHEN 'val'>0
num_threads=20 # the number of threads in openmp parallelization - never really needs to be changed unless you're familiar with OpenMP
Nc_equi=0 # the number of configurations before sampling - THIS SHOULD ALWAYS BE 0
input=input # type of input data - takes values of 'random' (random initial conditions) or 'input' (pre-defined configuration)

# loadnames relevant only if input='input' (absolute path names a lot easier)
loadname="/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_1024-phi_0.54-R_0.4146/POISSON-2D-N_1024-phi_0.54-R_0.4146"
# loadname="/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_2025-phi_0.54-R_0.4146/POISSON-2D-N_2025-phi_0.54-R_0.4146"
# loadname="/home/sd2402/CONFIGURATIONS/POISSON/POISSON-2D-N_3025-phi_0.54-R_0.4146/POISSON-2D-N_3025-phi_0.54-R_0.4146"

# loadname="/home/sd2402/CONFIGURATIONS/SATURATED_RSA/RSA-2D-N_1000-rho_1.0/RSA-2D-N_1000-rho_1.0"

# loadname="/home/sd2402/CONFIGURATIONS/URL/URL-2D-N_1024-phi_0.0078-R_0.05/URL-2D-N_1024-phi_0.0078-R_0.05"

# loadname="/home/sd2402/CONFIGURATIONS/N1000_configs_GaussianPairStat/GAUSSIAN-2D-N_1000"

# loadname="/home/sd2402/CONFIGURATIONS/HYPERPLANE_INTERSECTION/HPP/HPP"

# loadname="/home/sd2402/CONFIGURATIONS/THOMAS_PP/THOMAS-2D-N_1024-phi_0.54-R_0.4146/THOMAS-2D-N_1024-phi_0.54-R_0.4146"

savename=ccoptimization # savename of ConfigPack files
algorithm="MINOP" # "LBFGS" or "ConjugateGradient" or 'MINOP' or 'SteepestDescent' or 'qnLBFGS'
sim_type=ground	# simulation type - takes values of 'ground' or 'MD' 
							# ground = find ground states (ground states are stored in ${savename}_Success.ConfigPack)
							# MD = molecular dynamics simulation - data are stored in ${savename}.ConfigPack

# ~/ccoptimization/cco/EXC/soft_core_stealthy2.out ${timelimit} ${tolerance} ${max_evl} $seed <<< "${d} ${K1a} ${Ka} ${S0} ${val} ${sigma} ${phi_fake} $num_threads ${N} ${Nc} ${Nc_equi} ${input} $savename ${sim_type}"
~/ccoptimization/cco/EXC/soft_core_stealthy2.out ${timelimit} ${tolerance} ${max_evl} $seed <<< "${d} ${K1a} ${Ka} ${S0} ${val} ${sigma} ${packing_frac} $num_threads ${N} ${Nc} ${Nc_equi} ${input} ${algorithm} $loadname $savename ${sim_type}"
