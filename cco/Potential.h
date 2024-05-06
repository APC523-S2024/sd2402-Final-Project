/*
File for computing potential energy of point particle systems
Contains classes for general pairwise anisotropic and isotropic potentials
Also explicitly contains a class for the hard sphere potential

Sam Dawley
5/2024

References
...
*/

#ifndef POTENTIAL_INCLUDED
#define POTENTIAL_INCLUDED

#include <exception>
#include <vector>

#include "GeometryVector.h"
#include "etc.h"
#include "PeriodicCellList.h"

class EnergyDerivativeToBasisVectors_NotImplemented : public std::exception
{
public:
	EnergyDerivativeToBasisVectors_NotImplemented() : exception()
	{}
	virtual const char* what() const throw ()
	{
		return "EnergyDerivativeToBasisVectors is not implemented for this class!\n";
	}
};

/** \brief A virtual class to define potential energy. */
class Potential
{
public:
	DimensionType Dimension;	//< Space dimension.

	//some potentials might allow parallelization.
	//if so, this is the number of threads
	unsigned long ParallelNumThread;	//< The number of threads in openMP.

	/** A default constructor. */
	Potential(DimensionType dimension):Dimension(dimension)
	{
		this->ParallelNumThread=1;
	}
	virtual ~Potential();

	/** \brief A virtual function to compute the total potential energy. 
	 @return the total potential energy of the given configuration.*/
	virtual double Energy() =0;

	/** \brief A virtual function to compute force at the ith particle.
		@param[out] result	Force.
		@param[in]	i		Particle index.*/
	virtual void Force(GeometryVector & result, size_t i)
	{
		std::cerr<<"Error : virtual void Force is not implemented in this Potential!\n";
	}

	/**resize the vector to the number of particles, and calculate the force for all Particles.
	 @param[out] results	A list of forces. */
	virtual void AllForce(std::vector<GeometryVector> & results)
	{
		//not implemented, because
		//1. Don't know how many particles are there
		//2. Implementations, if parallelized, should ensure thread safety
		std::cerr<<"Error : virtual void AllForce is not implemented in this Potential!\n";
	}
	//Energy derivative respect to basis vectors
	//grad[n*dim+m] is the energy derivative of mth (coordinate) component of nth basis vector
	virtual void EnergyDerivativeToBasisVectors(double * grad, double Pressure)
	{
		throw EnergyDerivativeToBasisVectors_NotImplemented();
	}

	//set the potential to calculate energy and/or force for a specific configuration
	//the configuration must exist when calculating energy and force
	virtual void SetConfiguration(const Configuration & Position) =0;

	//this function will print a number very close to 1 when Force() is consistent with Energy(), and c is not a local minimum of this potential
	void Check(Configuration c, size_t Nbr = 0);

	//this function will print a number very close to 1 when EnergyDerivativeToBasisVectors() is consistent with Energy(), and c is not a local minimum of this potential
	void Check2(Configuration c, size_t Nbr = 0, double Pressure = 1.0);

};

//a potential that have good performance for Monte Carlo should inherent from this class and implement these two virtual functions
class forMCPotential
{
public:
	/** \brief A virtual member function to compute the change in potential energy by a trial move.
		The potential must have been set to the correct configuration before this function is called.
		@param pervCart	Current position in Cartesian coordinates.
		@param afterCart	A trial position in Cartesian coordinates.
		@return change in energy. */
	virtual double TryMove(GeometryVector & prevCart, GeometryVector & afterCart) = 0;

	/** \brief A virtual member function to accept the trial move and change the current state accordingly. */
	virtual void AcceptMove(void) = 0;
};

//add additional particles to the configuration in a potential
//useful in simulations with some particles fixed.
//this is a quick-and-dirty implementation
template<typename T> class AdditionalParticlesPotentialAdaptor : public T
{
private:
	std::vector<GeometryVector> AdditionalParticleRelatives;
	Configuration ConfigurationWithAdditionalParticles;
	//size_t NParticles;
public:
	AdditionalParticlesPotentialAdaptor(const std::vector<GeometryVector> & AdditionalParticleRelatives, const T & src): T(src), AdditionalParticleRelatives(AdditionalParticleRelatives)
		//, NParticles(0)
	{
	}
	virtual void SetConfiguration(const Configuration & Position)
	{
		//NParticles = Position.NumParticles();
		ConfigurationWithAdditionalParticles = Position;
		for (auto iter = AdditionalParticleRelatives.begin(); iter != AdditionalParticleRelatives.end(); ++iter)
			ConfigurationWithAdditionalParticles.Insert("Z", *iter);
		this->T::SetConfiguration(ConfigurationWithAdditionalParticles);
	}
	//virtual void AllForce() is not re-implemented because does not know how to parallelize
};

class PairPotential: public Potential
{
private:
	const Configuration * pConfig;
public:
	double Rcut2;//Rcut^2
	double Rcut;


	PairPotential(DimensionType dimension, double Rcut);
	void PairForce(const GeometryVector & offset, GeometryVector & result);
	double PairEnergy(const GeometryVector & offset);
	virtual ~PairPotential();
	virtual void SetConfiguration(const Configuration & Position);
	virtual double Energy();
	virtual void Force(GeometryVector & result, size_t i);//calculate the force for the ith Particle

	//contrary to setting a configuration and then using Energy() or Force()
	//this version does not change the state of the object unless EnergyFormula() or ForceFormula() changes the state of the object.
	//can be handy in threads
	double Energy(const Configuration & conf);//return the total potential energy of the given configuration
	virtual void Force(const Configuration & conf, GeometryVector & result, size_t i);//calculate the force for the ith Particle
	virtual void AllForce(std::vector<GeometryVector> & results);
	virtual double EnergyFormula(const GeometryVector &) =0;
	virtual void ForceFormula(const GeometryVector & offset, GeometryVector & result) =0;
	virtual double SecondDerivative(const GeometryVector & offset, DimensionType i, DimensionType j);//(dFi)/(dxj), recommended to rewrite it

	//Energy derivative respect to basis vectors
	//grad[n*dim+m] is the energy derivative of mth (coordinate) component of nth basis vector
	virtual void EnergyDerivativeToBasisVectors(double * grad, double Pressure);
	void EnergyDerivativeToBasisVectors(const Configuration & conf, double * grad, double Pressure);
};

/** \brief A class derived from PairPotential class. It is useful when pair potential is isotropic. */
class IsotropicPairPotential: public PairPotential
{
public:
	double Ecut;	//< A shift in energy such that energy vanishes as r goes to infinity.
	bool EcutUnset; //< true = Ecut is not initialized.

	/** A deafult constructor. This function do NOT initialize this->Ecut. */
	IsotropicPairPotential(DimensionType dimension, double Rcut);
	/** Set this->Ecut and this->EcutUnset = false.*/
	void SetEcut(void);
	virtual ~IsotropicPairPotential();

	virtual double EnergyFormula(const GeometryVector & offset);
	virtual void ForceFormula(const GeometryVector & offset, GeometryVector & result);

	/** Define pair potetial. Rcut and Ecut are not implemented.
		@return potential.	 */
	virtual double IsotropicEnergyFormula(double distance) =0;
	/** Compute the force. By default, it performes numerical differentiation. 
		It is strongly recommanded to redefine this function!.
		@return the magnitude of force. */
	virtual double IsotropicForceFormula(double distance);//\frac{- d \Psi(r)}{d r}, recommended to rewrite it

	//cutoff and shifted
	inline double IsotropicPairEnergy(double distance)
	{
		if(distance>this->Rcut) return 0;
		if(this->EcutUnset)
			this->SetEcut();
		return this->IsotropicEnergyFormula(distance)-this->Ecut;
	}
};

class LinearInterpolationIsotropicPairPotential : public IsotropicPairPotential
{
	std::vector<double> data;
	double Start, stepsize;
public:
	LinearInterpolationIsotropicPairPotential(IsotropicPairPotential & src, double Start, size_t pieces) : IsotropicPairPotential(src.Dimension, src.Rcut)
	{
		this->Start=Start;
		this->data.resize(pieces+2);
		this->stepsize=(src.Rcut-Start)/pieces;
		for(size_t i=0; i<this->data.size(); i++)
			this->data[i]=src.IsotropicPairEnergy(Start+i*this->stepsize);
	}
	virtual double IsotropicEnergyFormula(double distance)
	{
		if(distance<this->Start)
			return this->data[0];
		double dindex=(distance-this->Start)/this->stepsize;
		size_t iindex=(size_t)(std::floor(dindex));
		double rindex=dindex-iindex;//remaining index
		return this->data[iindex]+(this->data[iindex+1]-this->data[iindex])*rindex;
	}
};

namespace
{
	const double AspectRatioSquared=4;
	const signed MaxI=6;
	const signed MaxJ=6;
	const double CorrectionCenter=1.0;
	const double CorrectionWidth=0.1;
	class Correction : public IsotropicPairPotential
	{
		IsotropicPairPotential * orig;
		double V0;
	public:
		Correction(IsotropicPairPotential * original) : IsotropicPairPotential(original->Dimension, CorrectionCenter+2*CorrectionWidth)
		{
			this->orig=original;
			this->V0=0;
			for(signed i=(-1)*MaxI; i<=MaxI; i++)
			{
				for(signed j=(-1)*MaxJ; j<=MaxJ; j++)
				{
					double ii=i*i;
					double jj=j*j;
					if(i==0 && j==0)
						continue;
					V0-=orig->IsotropicPairEnergy(std::sqrt(ii+AspectRatioSquared*jj))*(ii-AspectRatioSquared*jj)/(ii+AspectRatioSquared*jj);
				}
			}
		}
		virtual double IsotropicEnergyFormula(double distance)
		{
			if(distance<CorrectionCenter-2*CorrectionWidth)
				return 0;
			if (distance>CorrectionCenter+2*CorrectionWidth)
				return 0;

			double V1=(-1)*this->V0;
			for(signed i=(-1)*MaxI; i<=MaxI; i++)
			{
				for(signed j=(-1)*MaxJ; j<=MaxJ; j++)
				{
					double ii=i*i;
					double jj=j*j;
					if(i==0 && j==0)
						continue;
					V1-=orig->IsotropicPairEnergy(distance*std::sqrt(ii+AspectRatioSquared*jj))*(ii-AspectRatioSquared*jj)/(ii+AspectRatioSquared*jj);
				}
			}
			V1/=2;
			double CorrectionCoeff;
			if(distance<CorrectionCenter-CorrectionWidth)
				CorrectionCoeff=( distance-(CorrectionCenter-2*CorrectionWidth) )/CorrectionWidth;
			else if(distance>CorrectionCenter+CorrectionWidth)
				CorrectionCoeff=( (CorrectionCenter+2*CorrectionWidth)-distance )/CorrectionWidth;
			else
				CorrectionCoeff=1;

			return V1*CorrectionCoeff;
		}
	};
};
class CorrectedIsotropicPairPotential : public IsotropicPairPotential
{
	IsotropicPairPotential * orig;
	LinearInterpolationIsotropicPairPotential * correction;

public:
	//the object (*original) will be freed when this class destruct
	CorrectedIsotropicPairPotential(IsotropicPairPotential * original) : IsotropicPairPotential(original->Dimension, original->Rcut)
	{
		this->orig=original;
		Correction corr(original);
		this->correction = new LinearInterpolationIsotropicPairPotential(corr, CorrectionCenter-2*CorrectionWidth, 500);
	}
	virtual double IsotropicEnergyFormula(double distance)
	{
		double result=orig->IsotropicEnergyFormula(distance);
		if(distance>CorrectionCenter-2*CorrectionWidth && distance<CorrectionCenter+2*CorrectionWidth)
			result+=this->correction->IsotropicEnergyFormula(distance);

		return result;
	}
	~CorrectedIsotropicPairPotential()
	{
		delete this->correction;
		delete this->orig;
	}
};

class CombinedIsotropicPairPotential : public IsotropicPairPotential
{
private:
	IsotropicPairPotential * p1;
	IsotropicPairPotential * p2;
public:
	CombinedIsotropicPairPotential(IsotropicPairPotential * p1, IsotropicPairPotential * p2) : IsotropicPairPotential(p1->Dimension, p1->Rcut>p2->Rcut?p1->Rcut:p2->Rcut)
	{
		assert(p1->Dimension==p2->Dimension);
		this->p1=p1;
		this->p2=p2;
	}
	virtual double IsotropicEnergyFormula(double distance)
	{
		return p1->IsotropicPairEnergy(distance)+p2->IsotropicPairEnergy(distance);
	}
	~CombinedIsotropicPairPotential()
	{
		delete this->p1;
		delete this->p2;
	}
};
class CombinedPotential : public Potential
{
public:
	Potential * p1;
	Potential * p2;
	double Weight1, Weight2;
	DimensionType Dimension;
	CombinedPotential(Potential * p1, Potential * p2): Potential(p1->Dimension)
	{
		assert(p1->Dimension==p2->Dimension);
		this->p1=p1;
		this->p2=p2;
		this->Weight1=1.0;
		this->Weight2=1.0;
	}
	virtual ~CombinedPotential()
	{
		//delete this->p1;
		//delete this->p2;
	}
	virtual double Energy()//return the total potential energy of the given configuration
	{
		return this->p1->Energy()*this->Weight1 + this->p2->Energy()*this->Weight2;
	}
	virtual void Force(GeometryVector & result, size_t i)//calculate the force for the ith Particle
	{
		GeometryVector f1, f2;
		this->p1->Force(f1, i);
		this->p2->Force(f2, i);
		result= f1*this->Weight1 + f2*this->Weight2;
	}
	virtual void AllForce(std::vector<GeometryVector> & results)//calculate the force for all Particles
	{
		std::vector<GeometryVector> temp;
		this->p1->AllForce(results);
		this->p2->AllForce(temp);
		for(int i=0; i<results.size(); i++)
			results[i]=results[i]*this->Weight1+temp[i]*this->Weight2;
	}
	//Energy derivative respect to basis vectors
	//grad[n*dim+m] is the energy derivative of mth (coordinate) component of nth basis vector
	virtual void EnergyDerivativeToBasisVectors(double * grad, double Pressure)
	{
		size_t dd=this->Dimension*this->Dimension;
		double * temp = new double[dd];
		this->p1->EnergyDerivativeToBasisVectors(grad, Pressure);
		this->p2->EnergyDerivativeToBasisVectors(temp, Pressure);
		for(size_t i=0; i<dd; i++)
			grad[i]=grad[i]*this->Weight1+temp[i]*this->Weight2;
		delete [] temp;
	}
	virtual void SetConfiguration(const Configuration & Position)
	{
		this->p1->SetConfiguration(Position);
		this->p2->SetConfiguration(Position);
	}
};

class RnPotential : public IsotropicPairPotential
{
public:
	double c, power;
	RnPotential(DimensionType dimension, double c, double power, double Cutoff) : IsotropicPairPotential(dimension, Cutoff), c(c), power(power)
	{}
	virtual double IsotropicEnergyFormula(double Distance)
	{
		return c*std::pow(Distance, power);
	}
	virtual double IsotropicForceFormula(double distance)
	{
		return (-1)*c*power*std::pow(distance, power-1.0);
	}
};





class HS_pot : public Potential
{
private:
	double maxR;
	double dx_max;
	bool monoDISPESE = true; //maximum displacement is identical for all particles
	RandomGenerator rng;
public:
	SpherePacking config;
	double Acceptance = 0.5;
	HS_pot() : Potential(1)
	{
		this->maxR = 0; this->dx_max = 0; 
	}
	HS_pot(DimensionType d) : Potential(d) { this->rng.seed(std::time(NULL)); }
	HS_pot(DimensionType d, double rMax) :Potential(d)
	{
		this->maxR = rMax;
		this->rng.seed(std::time(NULL));
	}
	HS_pot(DimensionType d, size_t s) :Potential(d)
	{
		this->rng.seed(s + std::time(NULL));
	}

	virtual void SetConfiguration(const SpherePacking & syst) {
		if (this->Dimension == syst.GetDimension()) {
			this->config = SpherePacking(syst);
			this->config.UpdateMaxRadius();
			this->maxR = this->config.GetMaxRadius();
			config.SetCellSize(2.0*this->maxR);
			this->dx_max = pow(this->config.PeriodicVolume() / (double)this->config.NumParticle(), 1.0 / this->config.GetDimension()) - this->maxR;
		}
		else {
			std::cout << "wrong dimension!!!" << std::endl;
		}
	}
	virtual void SetConfiguration(const Configuration & c) {
		std::cout << "wrong type of systems \n";
	}


	virtual double Energy() {
		if (config.CheckOverlap())
			return 1.0; //spheres are overlapped
		else
			return 0;
	}
	void SetDispersity(bool yes) { this->monoDISPESE = yes;	}
	void SetMaxDisplacement(double dx) { this->dx_max = dx; }

	double Energy_trial(size_t i, const GeometryVector & x) {//check whether the ith particle can move to x, x: relative coordinates
		bool overlap = false;
		std::vector<GeometryVector> data; //data[i].x[0]: index, data[i].x[1]:displace
		double r_i = config.GetCharacteristics(i);
		double D2 = this->maxR* this->maxR*4.0;
		double D = this->maxR * 2.0;
		//obtain indices of particles that can overlap with the test particle
		if (monoDISPESE) {
			config.IterateThroughNeighbors(x, D,
				[i, D2, &overlap](const GeometryVector &r, const GeometryVector &LatticeShift, const signed long * PeriodicShift, const size_t SourceParticle)
				->void {double dist2 = r.Modulus2();	if (SourceParticle != i && dist2 < D2) { overlap = true; return; }});

			if (overlap)
				return 1.0;
			else
				return 0;
		}

		else {
			config.IterateThroughNeighbors(x, this->maxR,
				[i, D2, &data](const GeometryVector &r, const GeometryVector &LatticeShift, const signed long * PeriodicShift, const size_t SourceParticle)
				->void {double dist2 = r.Modulus2();	if (SourceParticle != i && dist2 < D2) { data.push_back(GeometryVector(SourceParticle, sqrt(dist2))); }});

			for (int j = 0; j < data.size(); j++) {
				if (data[j].x[1] < r_i + config.GetCharacteristics(data[j].x[0]))
					return 1;
			}
			return 0;
		}
	}
	void UpdateConfig(size_t i, const GeometryVector & x) {
		this->config.MoveParticle(i, x);
	}

	double Evolve() {
		double count = 0;
		DimensionType d = this->Dimension;
		GeometryVector u(d);
		GeometryVector x0(d);
		for (int i = 0; i < this->config.NumParticle(); i++) {
			u = (this->dx_max)*RandomUnitVector(d, this->rng);
			x0 = this->config.CartesianCoord2RelativeCoord(u + this->config.GetCartesianCoordinates(i));
			if (Energy_trial(i, x0) < 0.5) {
				this->config.MoveParticle(i, x0);
				count++;
			}
		}
		return count / (double)this->config.NumParticle();
	}

	double Evolve(int numOfCycles) {//execute Evole() successively for a given numbers
		double result = 0;
		for (int i = 0; i < numOfCycles; i++)
			result += this->Evolve();

		return result / (double)numOfCycles;
	}

	void AdjustDisplacement(size_t numtrials) {
		double acc = 0;
		//int numtrials = 10;
		//----equilibrate----
		for (int i = 0; i < 5 * numtrials; i++)
			acc += Evolve();

		acc /= 5.0*numtrials;
		double factor = 0.1;
		double dev = fabs(acc - this->Acceptance);

		//int l = 0;
		//char name[20] = {};
		while (dev>0.02) {
			if (dev > 0.1)
				factor = 0.1;
			else
				factor = 0.01;

			if (acc > this->Acceptance)
				this->dx_max *= 1 + factor;
			else if (acc < this->Acceptance)
				this->dx_max *= 1 - factor;

			acc = 0;
			for (int i = 0; i < numtrials; i++)
				acc += Evolve();

			acc /= (double)numtrials;
			//std::cout << "Acceptance = " << acc << "\t maximum displacements = " << this->dx_max << "\n";
			dev = abs(acc - this->Acceptance);
			//	l++;
			//	sprintf(name, "%d", l);
			//	PlotPacking2D(name, this->config);
		}
	}

	SpherePacking Output() { return this->config; }
	double GetMaxDisplacement() {		return this->dx_max;	}
	
};

#endif
