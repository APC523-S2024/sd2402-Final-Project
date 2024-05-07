"""
Script for generating 2D Uniformly Randomized Lattice (URL) packings of hard spheres
We njit the heck out of this procedure so it'll be pretty fast up to 100**2 particles
Sam Dawley
2/28/24

Refs
- M. A. Klatt, J. Kim, and S. Torquato, Cloaking the Underlying Long-Range Order of Randomly Perturbed Lattices, Physical Review E, 101 032118 (2020).
"""
import sys
import numpy as np
import numba as nb
import numpy.linalg as la

##################################################
# MISCELLANEOUS
##################################################
def stringify(V: np.array, new_d: int=4) -> list:
    """
    Stringify V lattice vectors to 4D (a la CCOptimization) for writing to output files
    Args
        V = (n, d) lattice vectors
        new_d = 'dimension' of output basis vectors
    Returns
        res = (1, d) list of strings containing lattice vectors
    """
    res = []
    for v in V:
        line = ""
        svector = [str(norm) for norm in v]
        if len(svector) < new_d:
            svector += ["0.0" for _ in range(new_d-len(svector))]
        svector = np.asarray(svector).ravel()
        for sv in svector:
            line += sv + "\t"
        res.append(line)
    return res

@nb.njit(fastmath=True)
def main(primitive: np.array, num_particles: int) -> np.array:
    """
    Primary subroutine for creating Poisson point patterns in 2 dimensions
    Args
        primitive = primitive lattice vectors of simulation box
        num_particles = number of spheres to include in packing
    Returns
        particles = (num_particles**2, 2) ndarray containing 2D positions of all num_particles**2 particles
    """
    d = primitive.shape[0]
    e1norm = la.norm(primitive[0])
    e2norm = la.norm(primitive[1])
    
    return e1norm*np.random.rand(num_particles, d)

if __name__ == "__main__":
    localtime = True
    slurmtime = False
    twodigits = lambda x: f"{int(x):02d}" # yktv

    # SLURM QUEUING
    if slurmtime:
        seed_range = 1e6
        uu = np.random.randint(seed_range)
        np.random.seed(uu)
        print(f"Initialized as np.random.seed({uu})")

        dimension = int(sys.argv[1])
        lattice_constant = float(sys.argv[2])
        radius = 0.4145929793656026 # achieves packing fraction ~0.54 <=> saturated RSA in 2D
        N = int(sys.argv[3])
        total_particles = N**2
        outfile_idx = int(sys.argv[4])
        idx = twodigits(outfile_idx)
        
        side = lattice_constant*N
        phi = total_particles*(np.pi*radius**2)/side**2
        lv = np.array([[side, 0], [0, side]])

        pts = main(lv, total_particles)

        outfile = f"POISSON-2D-N_{total_particles:d}-config__00{idx}.txt"
        print(f"Writing to ./patterns/{outfile}")
        with open(f"./patterns/{outfile}", "w") as f:
            f.write(f"{dimension}\n")
            sv = stringify(lv)
            for v in sv:
                f.write(f"{v}\n")
            for pt in pts:
                spt = stringify([pt])[0]
                f.write(f"{spt}\n")

    # INDIVIDUAL CONFIGURATION GENERATION
    if localtime:
        dimension = 2
        lattice_constant = 1.0
        N = 512**2 # total number of particles
        lv = np.array([[lattice_constant*np.sqrt(N), 0], [0, lattice_constant*np.sqrt(N)]]) # make number density == 1
        write_output = True

        M = 50 # number of samples to generate
        seed_range = 1e8 # range [0, seed_range) to generate random seed for sampling
        for outfile_idx in range(M):    
            uu = np.random.randint(seed_range)
            np.random.seed(uu)

            pts = main(lv, N)
            
            idx = twodigits(outfile_idx)
            outfile = f"POISSON-2D-N_{N:d}-config__00{idx}.txt"

            if write_output:
                print(f"Writing to ./{outfile} (random seed={uu})")
                with open("./patterns/" + outfile, "w") as f:
                    f.write(f"{dimension}\n")
                    sv = stringify(lv)
                    for v in sv:
                        f.write(f"{v}\n")
                    for pt in pts:
                        spt = stringify([pt])[0]
                        f.write(f"{spt}\n")
