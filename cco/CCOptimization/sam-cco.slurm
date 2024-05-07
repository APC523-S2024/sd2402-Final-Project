#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G
#SBATCH --time=24:00:00
#SBATCH --job-name=ccopt-testing
#SBATCH --output=./%A-%a.log

# COMMAND LINE ARGUMENTS
timelimit=48 # time limit in hours. The code will stop and save the progress at the time limit; very important if you simulate with MD
tolerance=1e-17	# tolerance for the ground-state energy
max_evl=1000000 # maximum number of evaluations
seed=824352 # random seed for initial conditions.

# IMPORTANT COMMANDS
d=2 # dimensionality

K1a=0.0 # minimum wavenumber in the dimensionless exclusion region (0 implies hyperuniformity)

# (exclusion zone)x(individual particle radius) <=> K=1 corresponds to Ka=a
Ka=0.414593 # chi=0.02
# Ka=0.929516 # chi=0.1

packing_frac=0.0078 # USED FOR CALCULATING PARTICLE RADIUS
N=1024 # particle number
Nc=50 # the number of configurations
S0=0.0 # value of the structure factor for K1a < ka < Ka (0 means stealthy hyperuniformity.)
val=1.0 # strength of the soft-core repulsion <=> hard core applies to val \geq \phi\times10^{\phi}, i.e., choose val=1 for hard cores (0 implies no repulsion <=> nonzero 'val' results in hard code particles)
sigma=1 # ${list_sigma[$i]} # exclusion diameter of the soft-core repulsion - RELEVANT ONLY WHEN 'val'>0
num_threads=20 # the number of threads in openmp parallelization - never really needs to be changed unless you're familiar with OpenMP
Nc_equi=0 # the number of configurations before sampling - THIS SHOULD ALWAYS BE 0
input=input # type of input data - takes values of 'random' (random initial conditions) or 'input' (pre-defined configuration)

# loadnames relevant only if input='input' (absolute path names a lot easier)
# loadname="/home/sd2402/SP24/APC523/sd2402-Final-Project/CONFIGURATIONS/POISSON/POISSON-2D-N_1024/POISSON-2D-N_1024"
# loadname="/home/sd2402/SP24/APC523/sd2402-Final-Project/CONFIGURATIONS/RSA/RSA-2D-N_1024/RSA-2D-N_1000-rho_1.0"
loadname="/home/sd2402/SP24/APC523/sd2402-Final-Project/CONFIGURATIONS/URL/URL-2D-N_1024/URL-2D-N_1024-phi_0.0078-R_0.05"

savename=ccoptimization # savename of ConfigPack files
algorithm="qnLBFGS" # "LBFGS" or "ConjugateGradient" or 'MINOP' or 'SteepestDescent' or 'qnLBFGS'
sim_type=ground	# simulation type - takes values of 'ground' or 'MD' 
							# ground = find ground states (ground states are stored in ${savename}_Success.ConfigPack)
							# MD = molecular dynamics simulation - data are stored in ${savename}.ConfigPack

# ~/ccoptimization/cco/EXC/soft_core_stealthy2.out ${timelimit} ${tolerance} ${max_evl} $seed <<< "${d} ${K1a} ${Ka} ${S0} ${val} ${sigma} ${phi_fake} $num_threads ${N} ${Nc} ${Nc_equi} ${input} $savename ${sim_type}"
~/SP24/APC523/sd2402-Final-Project/cco/EXC/soft_core_stealthy2.out ${timelimit} ${tolerance} ${max_evl} $seed <<< "${d} ${K1a} ${Ka} ${S0} ${val} ${sigma} ${packing_frac} $num_threads ${N} ${Nc} ${Nc_equi} ${input} ${algorithm} $loadname $savename ${sim_type}"
