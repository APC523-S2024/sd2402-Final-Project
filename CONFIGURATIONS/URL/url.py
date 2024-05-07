"""
Script for generating 2D Uniformly Randomized Lattice (URL) packings of hard spheres
We njit the heck out of this procedure so it'll be pretty fast up to 100**2 particles

Sam Dawley
5/2024

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

@nb.njit
def spherical_autointersections(primitive: np.array, D: float, particles: np.array) -> bool:
    """
    Check if any pts within particles are intersection
    Assumes uniform diameter D for all particles and npt
    Primarily used for rejection sampling of particles ndarray 
    Args
        primitive = (d, d) primitive lattice vectors of cell containing both points
        D = particle diameters (all pts within particles and of npt)
        particles = (N, d) array containing Cartesian coordinates of all points
    Returns
        True (False) if ANY (NONE) intersections of the points contained within particles ndarray
    """
    for pt in particles:
        for npt in particles:
            if _test_spherical_intersection(primitive, D, pt, npt):
                return True
    return False

@nb.njit
def spherical_intersections(primitive: np.array, D: float, particles: np.array, npt: np.array) -> bool:
    """
    Check if hard sphere npt intersects any other hard sphere within ndarray particles
    Assumes uniform diameter D for all particles and npt
    Args
        primitive = (d, d) primitive lattice vectors of cell containing both points
        D = particle diameters (all pts within particles and of npt)
        particles = (N, d) array containing Cartesian coordinates of all points
        npt = (1, d) array of point which we are testing intersections against
    Returns
        True (False) if npt intersects ANY (NONE) of the points contained within particles ndarray
    """
    for pt in particles:
        if _test_spherical_intersection(primitive, D, pt, npt):
            return True
    return False

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

@nb.njit
def _test_spherical_intersection(primitive: np.array, D: float, pt1: np.array, pt2: np.array) -> bool:
    """ 
    Test if two spherical particles with uniform diameters D pt1 and pt2 overlap
    Used primarily as a subroutine for spherical_intersection() immediately below
    Args
        primitive = (d, d) primitive lattice vectors of cell containing both points
        D = particle diameters (should be the same)
        pt1, pt2 = (1, d) arrays of Cartesian coordinates of either point
    Returns
        True (False) if pt1 and pt2 (don't) intersect in Euclidean space
    """
    interparticle_dist = la.norm(nearest_image(primitive, pt1, pt2))
    if interparticle_dist <= D:
        return True
    return False

##################################################
# PRIMARY SUBROUTINES
##################################################
@nb.njit
def rejection_sample(M: int, primitive: np.array, a: float, num_particles: int) -> None:
    """
    Generates a TON of URL configurations without regard for hard core before dismissing those with overlapping particles
    If using significantly large radius a, M should on the order of millions/billions
    """
    mdot = int(np.log10(M))
    seed_range = 1e6 # range [0, seed_range) to generate random seed for sampling

    for m in range(M): 
        idx = str(int(m)).zfill(mdot)
        uu = np.random.randint(int(seed_range))
        np.random.seed(uu)

        pts = main(primitive, a, num_particles, hard_core=False)

        # test overlaps for this config
        if not spherical_autointersections(primitive, 2*a, pts):
            print(f"({m}) Configuration {idx} succeeded with random seed {uu}.")
            
            # outfile = f"testing2__00{idx}.txt"
            # print(f"Writing to ./temp_rejection/{outfile} (random seed={uu})")
            # print()

            # with open("./temp_rejection/" + outfile, "w") as f:
            #     f.write(f"{dimension}\n")
            #     sv = stringify(lv)
            #     for v in sv:
            #         f.write(f"{v}\n")
            #     for pt in pts:
            #         spt = stringify([pt])[0]
            #         f.write(f"{spt}\n")
        else:
            # print(f"({m}) Configuration {idx} failed.")
            pass


@nb.njit
def main(primitive: np.array, a: float, num_particles: int, hard_core: bool=True) -> np.array:
    """
    Primary subroutine for creating URL sphere packings in 2D
    Note that we iterate over num_particles-1 because we associate zeroth particle with Nth particle under perioidic boundaries
    -> Also, we don't really check for particle overlaps so make sure particle radius/box size aren't problematic
    Random perturbations to particles are U(-l/2, l/2) with l = lattice_constant
    Args
        primitive = primitive lattice vectors of simulation box
        a = radius of individual spheres
        num_particles = number of spheres to include in packing
        hard_core = whether to test for spherical overlaps. Should be true unless performing rejection sampling (above)
    Returns
        particles = (num_particles**2, 2) ndarray containing 2D positions of all num_particles**2 particles
    """
    nx, ny = num_particles-1, num_particles-1
    particles = np.zeros((nx*ny, 2))

    ##### INITIALIZE LATTICE #####
    e1norm = la.norm(primitive[0])
    e2norm = la.norm(primitive[1])
    X = np.linspace(0, e1norm, num_particles)
    Y = np.linspace(0, e2norm, num_particles)
    lattice_constant = np.ediff1d(X)[0] # box has equal side lengths

    ptidx = 0
    for i in range(nx):
        for j in range(ny):
            particles[ptidx] = np.array([X[i], Y[j]])
            ptidx += 1

    ##### RANDOM PERTURBATIONS #####
    if hard_core:
        # perturbed = np.zeros((1, particles.shape[0]))
        for idx, pt in enumerate(particles):
            u = np.random.uniform(low=-lattice_constant/2, high=lattice_constant/2, size=2)
            npt = nearest_image(primitive, pt + u)
            # if not spherical_intersections(primitive, 2*a, particles, npt):
                # particles[idx] = npt
            intersection = spherical_intersections(primitive, 2*a, particles, npt)
            while intersection:
                # print("INTERSECTING")    
                u = np.random.uniform(low=-lattice_constant/2, high=lattice_constant/2, size=2)
                npt = nearest_image(primitive, pt + u)
                intersection = spherical_intersections(primitive, 2*a, particles, npt)
            particles[idx] = npt
            # perturbed[idx] = 1
            print(f"particle {idx} ({npt-pt}) added")
    
    return particles

if __name__ == "__main__":
    individual = False
    slurmtime = False
    rejecttime = True
    twodigits = lambda x: f"{int(x):02d}" # yktv

    # SLURM QUEUING
    if slurmtime:
        dimension = int(sys.argv[1])
        lattice_constant = float(sys.argv[2])
        radius = float(sys.argv[3])
        N1 = int(sys.argv[4])
        outfile_idx = int(sys.argv[5])
        idx = twodigits(outfile_idx)
        
        side = lattice_constant*N1
        phi = (N1**2)*(np.pi*radius**2)/side**2
        lv = np.array([[side, 0], [0, side]])

        pts = main(lv, radius, N1+1)

        outfile = f"URL-2D-N_{N1**2:d}-phi_{phi:0.3f}-R_{radius:0.4f}__00{idx}.txt"
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
    if individual:
        dimension = 2
        lattice_constant = 1.0
        radius = 4e-1
        N1 = 32 # number of particles in a single direction
        lv = np.array([[lattice_constant*N1, 0], [0, lattice_constant*N1]])
        write_output = True

        M = 1 # number of samples to generate
        seed_range = 1e6 # range [0, seed_range) to generate random seed for sampling
        for outfile_idx in range(M):    
            uu = np.random.randint(seed_range)
            np.random.seed(uu)

            pts = main(lv, radius, N1+1)
            
            idx = twodigits(outfile_idx)
            outfile = f"testing2__00{idx}.txt"

            if write_output:
                print(f"Writing to ./{outfile} (random seed={uu})")
                with open("./" + outfile, "w") as f:
                    f.write(f"{dimension}\n")
                    sv = stringify(lv)
                    for v in sv:
                        f.write(f"{v}\n")
                    for pt in pts:
                        spt = stringify([pt])[0]
                        f.write(f"{spt}\n")
        
    if rejecttime:
        dimension = 2
        lattice_constant = 1.0
        radius = 4e-1
        N1 = 32 # number of particles in a single direction
        lv = np.array([[lattice_constant*N1, 0], [0, lattice_constant*N1]])

        M = 1e12 # number of samples to generate. Should be multiple of 10^x 
        rejection_sample(M, lv, radius, N1)
