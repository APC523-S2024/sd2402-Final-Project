/*
Header file for collection of pseudo-random number generation functions

Sam Dawley
5/2024

References
...
*/

#include <cmath>
#include <ctime>

#include "RandomGenerator.h"

void RandomGenerator::initialize(void)
	{
		this->min=gen.min();
		this->max=gen.max();
	}
void RandomGenerator::seed(int s)
	{
		this->gen.seed(s);
	}
RandomGenerator::RandomGenerator()
	{
		this->initialize();
		this->seed(std::time(NULL));
	}
RandomGenerator::RandomGenerator(int Seed)
	{
		this->initialize();
		this->seed(Seed);
	}
double RandomGenerator::RandomDouble(void)//[0,1)
	{
		return (double)(gen()-min)/((double)(max-min)+1);
	}
double RandomGenerator::RandomDouble(const std::function<double(double)> &inverseCp)
	{
		return inverseCp(RandomDouble());
	}

int RandomGenerator::PoissonRNG(double mean){
	//static boost::random::poisson_distribution<int> rgn(mean);
	//static boost::random::variate_generator<boost::mt19937, boost::random::poisson_distribution<int>> rvt(this->gen, rgn);
	boost::random::poisson_distribution<int> rgn(mean);
	boost::random::variate_generator<boost::mt19937, boost::random::poisson_distribution<int>> rvt(this->gen, rgn);
	return rvt();
}
void GetRandomVector(int Dimension, RandomGenerator & gen, double * result)
{
	double r;
	do
	{
		r=0;
		for(int i=0; i<Dimension; i++)
		{
			result[i]=2*gen.RandomDouble()-1;
			r+=result[i]*result[i];
		}
	}
	while(r>1);
	for(int i=0; i<Dimension; i++)
		result[i]/=std::sqrt(r);

	return;
}
