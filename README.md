# sd2402-Final-Project

Implementation of quasi-Newton method for large-scale unconstrained optimization applied to generation of stealthy hyperuniform point patterns. The collective coordinate procedure algorithm used herein is credited to [Prof. Ge Zhang](https://scholar.google.com/citations?user=DsdZPxcAAAAJ). This repository is meant to demonstrate application of the Limited-Memory BFGS update step for quasi-Newton methods applied to the collective coordinate optimization algorithm. Any questions regarding the implementation should be directed to me at sd2402[at]princeton.edu.

This project was created in part for the course APC 523 Numerical Algorithms for Scientific Computing with Prof. Romain Teyssier.

# Usage
From the root directory navigate to `./cco/CCOptimization/makefile` and change the value of `EXEdir` to the preferred path on your machine. Then, running `make` from within this directory will create the executable `soft_core_stealthy2.out` in the specified executable directory.

Similarly, we must make the file manager. From the root directory navigate to `./cco/EXC_formats/makefile` and change the variable `EXEdir` to match that from above. Running `make` from within this directory will lead to the executable `manager.out` in the prescribed directory; the `manager` is responsible for reading and writing outputs created during optimization.

## Submitting Batch Jobs
After the executables above are created the file can be run on Adroit using the SLURM submission script `sam-cco.slurm` within `./cco/CCOptimization`. There are some default `SBATCH` directives already but these should be changed as necessary. Also, in the final line of this file `soft_core_stealthy2.out` is called from the executable directory `EXEdir` from above. If you changed the location of this directory when altering the makefiles ensure that the path to this directory in the SLURM submission script is correct.