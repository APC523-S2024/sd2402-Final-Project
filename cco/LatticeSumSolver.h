/*
Header file for lattice structures

Sam Dawley
5/2024

References
...
*/

#ifndef LATTICESUMSOLVER_INCLUDED
#define LATTICESUMSOLVER_INCLUDED

//#if defined(POTENTIAL_INCLUDED)
#include "Potential.h"
//#endif

#include <functional>
#include <vector>
#include <fstream>
#include <iostream>

#include "PeriodicCellList.h"

struct LatticeTerm
{
	unsigned long number;
	double distance;

	LatticeTerm() {
		this->number = 0;
		this->distance = 0;
	}

	LatticeTerm(double x, unsigned long y) {
		this->distance = x;
		this->number = y;
	}
};

class LatticeSumSolver
{
public:
	std::vector<LatticeTerm> Terms; // Terms per unit cell
	double CurrentRc;
	double OriginalDensity; // the density of the crystal that generates Terms
	size_t NumParticle; // Number of atoms per unit cell. 
	DimensionType Dimension;

	LatticeSumSolver(void);

	virtual const char * Tag(void) =0;
	virtual ~LatticeSumSolver();
	virtual Configuration GetStructure(void) =0;
	virtual void UpdateTerms(double NewRc);

#if defined POTENTIAL_INCLUDED
	double LatticeSum(double Volume, IsotropicPairPotential & potential);
#endif

};

bool SameSolver(LatticeSumSolver * a, LatticeSumSolver * b);
double SolverDistance(LatticeSumSolver * a, LatticeSumSolver * b);


#endif