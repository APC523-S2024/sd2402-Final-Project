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
tolerance=1e-17
max_evl=1000000
seed=824352 # random seed for initial conditions.

# IMPORTANT COMMANDS
d=2 # dimensionality

K1a=0.0 # minimum wavenumber in the dimensionless exclusion region (0 implies hyperuniformity)

# (exclusion zone)x(individual particle radius) <=> K=1 corresponds to Ka=a
Ka=0.414593 # chi=0.02

packing_frac=0.0078 # USED FOR CALCULATING PARTICLE RADIUS
N=1024 # particle number
Nc=50 # the number of configurations
S0=0.0
val=1.0
sigma=1
num_threads=20
Nc_equi=0
input=input

# loadname="./CONFIGURATIONS/POISSON/POISSON-2D-N_1024/POISSON-2D-N_1024"
# loadname="./CONFIGURATIONS/RSA/RSA-2D-N_1024/RSA-2D-N_1000-rho_1.0"
loadname="./CONFIGURATIONS/URL/URL-2D-N_1024/URL-2D-N_1024-phi_0.0078-R_0.05"

savename=ccoptimization # savename of ConfigPack files
algorithm="qnLBFGS"
sim_type=ground

./EXC/soft_core_stealthy2.out ${timelimit} ${tolerance} ${max_evl} $seed <<< "${d} ${K1a} ${Ka} ${S0} ${val} ${sigma} ${packing_frac} $num_threads ${N} ${Nc} ${Nc_equi} ${input} ${algorithm} $loadname $savename ${sim_type}"
