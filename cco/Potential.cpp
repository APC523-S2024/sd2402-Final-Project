/*
Header file for computing potential energy of point particle systems
Contains classes for general pairwise anisotropic and isotropic potentials
Also explicitly contains a class for the hard sphere potential

Sam Dawley
5/2024

References
...
*/

#include <cassert>
#include <cmath>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

#include "Potential.h"

const double Potential_Check_Delta = 1e-7;
void Potential::Check(Configuration c, size_t Nbr)
{
	this->SetConfiguration(c);
	GeometryVector temp;
	GeometryVector force;
	this->Force(force, Nbr);
	double E0 = this->Energy();
	temp = c.GetCartesianCoordinates(Nbr);
	temp.x[0] += Potential_Check_Delta;
	c.MoveParticle(Nbr, c.CartesianCoord2RelativeCoord(temp));
	this->SetConfiguration(c);
	double E1 = this->Energy();
	std::cout << (E0 - E1) / Potential_Check_Delta / force.x[0] << '\n';
}
//this function will print a number very close to 1 when EnergyDerivativeToBasisVectors() is consistent with Energy(), and c is not a local minimum of this potential
void Potential::Check2(Configuration c, size_t Nbr, double Pressure)
{
	size_t d2 = Dimension*Dimension;
	Nbr = Nbr%d2;
	double * der = new double[d2];
	this->SetConfiguration(c);
	this->EnergyDerivativeToBasisVectors(der, Pressure);
	double E0 = this->Energy() + Pressure*c.PeriodicVolume();

	GeometryVector bas[::MaxDimension];
	for (DimensionType i = 0; i < Dimension; i++)
		bas[i] = c.GetBasisVector(i);
	bas[Nbr / Dimension].x[Nbr%Dimension] += Potential_Check_Delta;
	c.ChangeBasisVector(bas);
	this->SetConfiguration(c);
	double E1 = this->Energy() + Pressure*c.PeriodicVolume();
	std::cout << (E1 - E0) / Potential_Check_Delta / der[Nbr] << '\n';

	delete[] der;
}
Potential::~Potential()
{
}
PairPotential::PairPotential(DimensionType dimension, double Rcut):Potential(dimension)
{
	if(dimension<=0) throw "error in PairPotential::PairPotential : unsupported dimension";
	if (Rcut < 0)
	{
		std::cerr<< "error in PairPotential::PairPotential : unsupported Rcut="<<Rcut<<'\n';
	}

	this->Dimension=dimension;
	this->Rcut2=Rcut*Rcut;
	this->Rcut=Rcut;
	this->pConfig=nullptr;
}
PairPotential::~PairPotential()
{
}
void PairPotential::SetConfiguration(const Configuration & Position)
{
	this->pConfig = & Position;
}
void PairPotential::PairForce(const GeometryVector & offset, GeometryVector & result)
{
	double distance2=offset.Modulus2();
	if(distance2>this->Rcut2)
	{
		result=GeometryVector(this->Dimension);
		return;
	}
	this->ForceFormula(offset, result);
}
double PairPotential::PairEnergy(const GeometryVector & offset)
{
	double distance2=offset.Modulus2();
	if(distance2>this->Rcut2)
		return 0;
	return this->EnergyFormula(offset);
}
double PairPotential::Energy()
{
	assert(this->pConfig!=nullptr);
	const Configuration & conf=*this->pConfig;
	double Energy=0;
	//pre-allocation of internal neighbor-cell cache in Configuration
	this->pConfig->PrepareIterateThroughNeighbors(this->Rcut);
#pragma omp parallel for num_threads(this->ParallelNumThread)
	for(long i=0; i<conf.NumParticle(); i++)
	{
		PairPotential & backup = ( *this);
		//Configuration::particle * trial = this->pConfig->GetParticle(i);
		conf.IterateThroughNeighbors(conf.GetRelativeCoordinates(i), this->Rcut, [&Energy, &backup, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) ->void 
		{
			if(SourceAtom!=i || LatticeShift.Modulus2()!=0) 
			{
				double temp=backup.PairEnergy(shift);
#pragma omp atomic
				Energy+=temp;
			}
		});
	}
	Energy/=2;
	return Energy;
}
double PairPotential::Energy(const Configuration & conf)
{
	double Energy=0;
	for(Configuration::Index i=0; i<conf.NumParticle(); i++)
	{
		PairPotential & backup = ( *this);
		//Configuration::particle * trial = this->pConfig->GetParticle(i);
		conf.IterateThroughNeighbors(conf.GetRelativeCoordinates(i), this->Rcut, [&Energy, &backup, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) ->void 
		{
			if(SourceAtom!=i || LatticeShift.Modulus2()!=0) 
			{
				double temp=backup.PairEnergy(shift);
				Energy+=temp;
			}
		});
	}
	Energy/=2;
	return Energy;
}
void PairPotential::Force(GeometryVector & result, size_t i)
{
	assert(this->pConfig!=nullptr);
	result=GeometryVector(this->Dimension);

	PairPotential & backup = ( *this);
	//Configuration::particle * trial = this->pConfig->GetParticle(i);
	this->pConfig->IterateThroughNeighbors(this->pConfig->GetRelativeCoordinates(i), this->Rcut, [&result, &backup, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) ->void 
	{
		if(SourceAtom==i)
			return;
		GeometryVector temp=SameDimensionZeroVector(shift);
		if(shift.Modulus2()> ::LengthPrecision* ::LengthPrecision) backup.PairForce((-1.0)*shift, temp);
		result.AddFrom(temp);
	}
	);
	return;
}
void PairPotential::Force(const Configuration & conf, GeometryVector & result, size_t i)
{
	result=GeometryVector(this->Dimension);

	PairPotential & backup = ( *this);
	conf.IterateThroughNeighbors(conf.GetRelativeCoordinates(i), this->Rcut, [&result, &backup, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) ->void 
	{
		if(SourceAtom==i)
			return;
		GeometryVector temp=SameDimensionZeroVector(shift);
		if(shift.Modulus2()> ::LengthPrecision* ::LengthPrecision) backup.PairForce((-1.0)*shift, temp);
		result.AddFrom(temp);
	}
	);
	return;
}
void PairPotential::AllForce(std::vector<GeometryVector> & results)
{
	size_t Np=this->pConfig->NumParticle();
	results.resize(Np);
	//pre-allocation of internal neighbor-cell cache in Configuration
	this->pConfig->PrepareIterateThroughNeighbors(this->Rcut);
#pragma omp parallel for num_threads(this->ParallelNumThread)
	for(long i=0; i<Np; i++)
		this->Force(results[i], i);
}
double PairPotential::SecondDerivative(const GeometryVector & offset, DimensionType i, DimensionType j)//(d^2V)/(dxi*dxj), recommended to rewrite it
{
	GeometryVector off(offset);
	GeometryVector f1(this->Dimension);
	this->ForceFormula(off, f1);

	off.x[i]+= ::LengthPrecision;
	GeometryVector f2(this->Dimension);
	this->ForceFormula(off, f2);

	return (f1.x[j]-f2.x[j])/ ::LengthPrecision;
}


void IsotropicPairPotential::SetEcut(void)
{		
	this->Ecut=this->IsotropicEnergyFormula(std::sqrt(this->Rcut2));
	this->EcutUnset=false;
}
double IsotropicPairPotential::EnergyFormula(const GeometryVector & offset)
{
	return this->IsotropicPairEnergy(std::sqrt(offset.Modulus2()));
}

void IsotropicPairPotential::ForceFormula(const GeometryVector & offset, GeometryVector & result)
{
	double dist=std::sqrt(offset.Modulus2());
	if(dist > this->Rcut)
	{
		result = GeometryVector(this->Dimension);
		return;
	}
	double coeff=this->IsotropicForceFormula(dist)/dist;
	result=coeff*offset;
}

IsotropicPairPotential::IsotropicPairPotential(DimensionType dimension, double Rcut):PairPotential(dimension, Rcut)
{
	this->EcutUnset=true;
}
IsotropicPairPotential::~IsotropicPairPotential()
{
}

const double deltaR=1e-10;
double IsotropicPairPotential::IsotropicForceFormula(double distance)
{
	double e1=this->IsotropicEnergyFormula(distance-deltaR);
	double e2=this->IsotropicEnergyFormula(distance+deltaR);
	return (e1-e2)/(2*deltaR);
}



//Energy derivative respect to basis vectors
//grad[n*dim+m] is the energy derivative of mth (coordinate) component of nth basis vector
void PairPotential::EnergyDerivativeToBasisVectors(double * grad, double Pressure)
{
	assert(this->pConfig!=nullptr);

	DimensionType dim=this->pConfig->GetDimension();
	for(size_t i=0; i<dim*dim; i++)
		grad[i]=0.0;
	double Volume=this->pConfig->PeriodicVolume();
	//pre-allocation of internal neighbor-cell cache in Configuration
	this->pConfig->PrepareIterateThroughNeighbors(this->Rcut);
#pragma omp parallel for num_threads(this->ParallelNumThread)
	for(long i=0; i<this->pConfig->NumParticle(); i++)
	{
		//Configuration::particle * pA=this->pConfig->GetParticle(i);
		const Configuration * pTemp=this->pConfig;
		//derivatives of cell basis vectors
		this->pConfig->IterateThroughNeighbors(this->pConfig->GetRelativeCoordinates(i), this->Rcut, 
			[&] (const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t srcAtom) ->void
		{
			if(shift.Modulus2()< LengthPrecision*LengthPrecision)
				return;
			GeometryVector RelativeShift = pTemp->GetRelativeCoordinates(srcAtom)-pTemp->GetRelativeCoordinates(i);
			for(DimensionType i=0; i<dim; i++)
				RelativeShift.x[i]+=PeriodicShift[i];
			GeometryVector pairForce(dim);
			this->PairForce(shift, pairForce);
			for(DimensionType m=0; m<dim; m++)
				for(DimensionType n=0; n<dim; n++)
				{
					double temp=(-0.5)*(pairForce.x[m])*RelativeShift.x[n];
#pragma omp atomic
					grad[n*dim+m] += temp;
				}
		}
		);
	}
	//get the inverse basis vector matrix
	gsl_matrix * minvbasis=nullptr;
	gsl_matrix * mbasis = gsl_matrix_alloc(dim, dim);
	minvbasis = gsl_matrix_alloc(dim, dim);
	for(DimensionType i=0; i<dim; i++)
	{
		GeometryVector temp=this->pConfig->GetBasisVector(i);
		for(DimensionType j=0; j<dim; j++)
			mbasis->data[i*dim+j]=temp.x[j];
	}
	gsl_permutation * p = gsl_permutation_calloc(dim);
	int signum = 0;
	gsl_linalg_LU_decomp(mbasis, p, &signum);
	gsl_linalg_LU_invert(mbasis, p, minvbasis);
	gsl_permutation_free(p);
	gsl_matrix_free(mbasis);
	for(size_t j=0; j<dim; j++)
	{
		for(size_t k=0; k<dim; k++)
		{
			grad[j*dim+k]+=Pressure*Volume*gsl_matrix_get(minvbasis, k, j);
		}
	}
	gsl_matrix_free(minvbasis);
}
void PairPotential::EnergyDerivativeToBasisVectors(const Configuration & conf, double * grad, double Pressure)
{
	this->SetConfiguration(conf);
	return this->EnergyDerivativeToBasisVectors(grad, Pressure);
}

