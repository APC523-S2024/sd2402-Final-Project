"""
Script for random sequential addition of hard spheres in 2D with variable penetrabilities

Sam Dawley
5/2024
"""
import os
import sys
import numpy as np
import numba as nb
import numpy.linalg as la

SQRT_PI = 1.7724538509055159

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

@nb.jit(nopython=True)
def nearest_image(primitive: np.array, p1: np.array, p2: np.array=None) -> list:
    """
    Compute d-dimensional Euclidean distance between points p1, p2 using nearest-image convention
    Args
        p1 = (1, d) coordinates of tagged point
        p2 = (1, d) coordinates of other point
        primitive = (d, d) primitive lattice vectors of simulation box
    Returns
        (1, d) coordinates of distance from origin (or distance between points) accounting for periodic boundaries
    """
    d = p1.size # cheeky dimension
    primitive_norms = np.array([la.norm(k) for k in primitive])
    # Nearest-image position of a single particle
    if p2 is None:
        for i, k in zip(range(d), primitive_norms): 
            if   p1[i] >=  k: p1[i] -= k
            elif p1[i] <= -k: p1[i] += k
        return np.abs(p1)
    # Nearest-image distance between two particles
    else:
        dr = p1 - p2
        for i, k in zip(range(d), primitive_norms): 
            if   dr[i] >=  k/2: dr[i] -= k
            elif dr[i] <= -k/2: dr[i] += k
        return np.abs(dr)

############################################################
# POTPOURRI
############################################################
def fix2digits(x: int) -> str:
    return f"{int(x):02d}"

def phi2radius(p2: float, w: float) -> float:
    """
    Convert particle volume fraction to radius for 2D spheres of penetrability w
    Only supports fully penetrating (w=0) and fully impenetrating (w=1) as of now
    Args
        p2 = particle volume fraction
        w = lambda parameter defining penetrability of spheres; w must be on [0, 1]
    """
    if p2 < 0 or p2 > 1:
        print(f"Particle volume fraction outside of range [0, 1]; aborting...")
        return 0
    elif w < 0 or w > 1:
        print(f"Penetrability factor outside of range [0, 1]; aborting...")
        return 0
    
    if w:
        return np.sqrt(p2)/SQRT_PI # impenetrable
    return np.sqrt(-np.log(1-p2))/SQRT_PI # fully penetrating

############################################################
# READING/WRITING DATA
############################################################
def writeCCO(data: tuple, seed: int, fidx: int=0) -> None:
    """
    Write point pattern to CCO-style output file
    Args
        d = dimensionality of point pattern
        data = tuple containing all relevant data for simulation. Must take the form
            data[0] = integer dimensionality of pattern
            data[1] = (N, d) ndarray containing d-dimensional positions of all N particles
            data[2] = (d, d) ndarray of primitive lattice vectors which define box containing all particles
        fidx = integer representing which configuration out of all M members of the ensemble we are writing down
    """
    idx = fix2digits(fidx)
    d, pts, primitive = data
    N = pts.shape[0]

    outdir = "./patterns/"
    if not os.path.isdir(outdir):
        print(f"Output directory does not exist. Creating {outdir}")
        os.mkdir(outdir)

    outfile = f"RSA-N_{N:d}-config__00{idx}.txt"
    print(f"Writing to {outdir + outfile} (random seed={seed})")

    with open(outdir + outfile, "w") as f:
        f.write(f"{d}\n")
        sv = stringify(primitive)
        for v in sv:
            f.write(f"{v}\n")
        for pt in pts:
            spt = stringify([pt])[0]
            f.write(f"{spt}\n")

############################################################
# PLACING PARTICLES 
############################################################
@nb.njit(fastmath=True)
def random2Daddition(simulation: np.array, num_particles: int, primitive: np.array, R: float, w: float) -> np.array:
    """
    Randomly and sequentially place hard spheres of penetrability defined by w into simulation box defined by lv
    Args
        simulation = (N, d) array of positions of all particles included in simulation
        num_particles = integer number of spheres to include in simulation
        primitive = (d, d) ndarray of primitive lattice vectors of the cell
        R = radius of particles to be included in simulation (monodisperse)
        w = float on interval [0, 1] representing penetrability constraints of spheres
    """
    # if no penetrability constraint just return a poisson pattern
    dimension = primitive.shape[0]
    effective_radius = w*R

    if w == 0:
        for d in range(dimension):
            simulation[:, d] = la.norm(primitive[d]) * np.random.rand(num_particles)
        return

    Lx = la.norm(primitive[0])
    Ly = la.norm(primitive[1])

    # otherwise go through testing of overlap at each addition
    n_counter = 0
    while n_counter < num_particles:
        # print("Attemping to place particle " + str(n_counter+1) + " of " + str(num_particles))
        x = Lx*np.random.rand()
        y = Ly*np.random.rand()
        new_particle = np.array([x, y])

        # addition attempt by testing all interparticle distances
        # if new_particle and test overlap, break out of loop
        # if new_particle does not overlap with any existing particles, add to simulation array
        overlapping = False
        for particle in simulation:
            rdiff = nearest_image(primitive, new_particle, particle)
            r = la.norm(rdiff)

            if r <= 2*effective_radius:
                overlapping = True

        if not overlapping:
            simulation[n_counter] = new_particle
            n_counter += 1

    return


if __name__ == "__main__":
    rng = np.random.default_rng()
    ensemble = np.arange(0, 10, 1) # number of configurations
    
    # metadata
    d = 2
    N = 32**2
    exclusion = 1.0 # lambda value controlling penetrability of spheres
    particle_vol_fraction = 0.2

    particle_radius = phi2radius(particle_vol_fraction, exclusion)

    if particle_radius:    
        a10 = 1.0 # lattice constant
        lv = np.array([[a10*np.sqrt(N), 0], [0, a10*np.sqrt(N)]], dtype=np.float32) # ensures unit number density
    
        for midx in ensemble:
            Omega = np.zeros((N, d))
            uu = rng.integers(1e8)
            np.random.seed(uu)

            random2Daddition(Omega, N, lv, particle_radius, exclusion)

            res = (d, Omega, lv)
            writeCCO(res, uu, fidx=midx)

    else:
        print("Vanishing particle radius; aborting...")
