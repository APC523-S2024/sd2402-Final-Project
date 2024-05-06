/*
Function implementations for setting parameters of optimization runs prior to job submission

Sam Dawley
5/2024

References
...
*/

#ifndef PARAMETERSET_INCLUDED
#define PARAMETERSET_INCLUDED

#include "Potential.h"
#include "RandomGenerator.h"

#include <iostream>

class ParameterSet
{
public:
	double * Parameters;
	ParameterSet()
	{
		this->Parameters = nullptr;
	}

	ParameterSet(const ParameterSet & src)
	{
		if (src.Parameters == nullptr) {
			return;
		}
		size_t nbr = src.NumParameters();
		this->Parameters = new double [nbr];
		std::memcpy(this->Parameters, src.Parameters, sizeof(double)*nbr);
	}

	ParameterSet(ParameterSet && src)
	{
		std::swap(this->Parameters, src.Parameters);
	}

	virtual ~ParameterSet()
	{
		if (this->Parameters != nullptr)
			delete [] this->Parameters;
	}
	virtual IsotropicPairPotential * GetPotential(void) const =0;

	//calculate the penalty from OptimizableParameters
	//get range of each parameters
	virtual double Penalty(void) const = 0;
	virtual void GetBound(double * LowerBound, double * UpperBound) const = 0;

	// evolve to a more complex Parameterization, return true if succed, return false if can't evolve any more
	// for some non-continuous parameter, iterate through it. Intended to be used when it has never evolved
	virtual bool Evolve(void) = 0;
	virtual bool Iterate(void) = 0;
	virtual ParameterSet * clone() const = 0;
	virtual void Write(std::ostream & out) const = 0;
	virtual void Read(std::istream & in) = 0;

	// when r<MinDistance, the potential should have no interesting behavior, guaranteed to fix unchanged during the life of this object
	// the Rcut of the potential should not exceed this, guaranteed to fix unchanged during the life of this object
	virtual double MinDistance(void) const = 0;
	virtual double MaxDistance(void) const = 0;

	virtual size_t NumParameters(void) const = 0;
	virtual void SignalGlobalOptimization(void)
	{
	}

	virtual void SignalLocalOptimization(void)
	{
	}

	virtual double SharpestRelativeSize(void) = 0;

	void Randomize(RandomGenerator & gen)
	{
		size_t n = this->NumParameters();
		double * lb = new double[n];
		double * ub = new double[n];
		this->GetBound(lb, ub);

		for (size_t i = 0; i < n; i++) {
			this->Parameters[i] = lb[i]+(ub[i]-lb[i])*gen.RandomDouble();
		}

		delete [] ub;
		delete [] lb;
	}
};
	
class CombinedParameterization : public ParameterSet
{
private:
	ParameterSet * p1;
	ParameterSet * p2;
	double * pParameters1;
	double * pParameters2;

public:
	CombinedParameterization(ParameterSet * p1, ParameterSet * p2) : ParameterSet()
	{
		this->p1 = p1;
		this->p2 = p2;
		this->Allocate();
	}
	~CombinedParameterization()
	{
		this->DeAllocate();
		delete this->p1;
		delete this->p2;
	}

	virtual IsotropicPairPotential * GetPotential(void) const
	{
		IsotropicPairPotential * result = new CombinedIsotropicPairPotential(this->p1->GetPotential(), this->p2->GetPotential());
		return result;
	}

	virtual double Penalty(void) const
	{
		double result = this->p1->Penalty()+this->p2->Penalty();
		return result;
	}

	virtual void GetBound(double * LowerBound, double * UpperBound) const
	{
		size_t Num1 = p1->NumParameters();

		this->p1->GetBound(LowerBound, UpperBound);
		this->p2->GetBound(LowerBound+Num1, UpperBound+Num1);
	}

	virtual bool Evolve(void)
	{
		this->DeAllocate();
		double result = this->p2->Evolve();
		this->Allocate();
		return result;
	}

	virtual bool Iterate(void)
	{
		this->DeAllocate();
		double result = this->p2->Iterate();
		this->Allocate();
		return result;
	}

	virtual void Write(std::ostream & out) const
	{
		this->p1->Write(out);
		this->p2->Write(out);
	}

	virtual void Read(std::istream & in)
	{
		this->DeAllocate();
		this->p1->Read(in);
		this->p2->Read(in);
		this->Allocate();
	}

	virtual double MinDistance(void) const
	{
		double m1 = this->p1->MinDistance();
		double m2 = this->p2->MinDistance();

		return m1<m2?m1:m2;
	}

	virtual double MaxDistance(void) const
	{
		double m1 = this->p1->MaxDistance();
		double m2 = this->p2->MaxDistance();

		return m1 > m2 ? m1 : m2;
	}

	virtual size_t NumParameters(void) const
	{
		return this->p1->NumParameters()+this->p2->NumParameters();
	}

	virtual double SharpestRelativeSize(void)
	{
		double s1 = this->p1->SharpestRelativeSize();
		double s2 = this->p2->SharpestRelativeSize();

		return s1<s2?s1:s2;
	}

	virtual ParameterSet * clone() const
	{
		return new CombinedParameterization(this->p1->clone(), this->p2->clone());
	}
	
	void Allocate()
	{
		size_t Num1 = p1->NumParameters();
		size_t Num2 = p2->NumParameters();

		this->Parameters = new double [Num1+Num2];
		this->pParameters1 = this->Parameters;
		this->pParameters2 = this->Parameters+Num1;

		std::memcpy(pParameters1, p1->Parameters, Num1*sizeof(double));
		std::memcpy(pParameters2, p2->Parameters, Num2*sizeof(double));

		std::swap(this->pParameters1, this->p1->Parameters);
		std::swap(this->pParameters2, this->p2->Parameters);
	}

	void DeAllocate()
	{
		size_t Num1=p1->NumParameters();
		size_t Num2=p2->NumParameters();

		std::memcpy(pParameters1, p1->Parameters, Num1*sizeof(double));
		std::memcpy(pParameters2, p2->Parameters, Num2*sizeof(double));

		std::swap(this->pParameters1, this->p1->Parameters);
		std::swap(this->pParameters2, this->p2->Parameters);

		delete this->Parameters;

		this->Parameters = nullptr;
		this->pParameters1 = nullptr;
		this->pParameters2 = nullptr;
	}
};

#endif