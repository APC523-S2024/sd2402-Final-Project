/*
Header file for collection of pseudo-random number generation functions

Sam Dawley
5/2024

References
...
*/

#ifndef RANDOMGENERATOR_INCLUDED
#define RANDOMGENERATOR_INCLUDED

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>

class RandomGenerator
{
private:
	boost::mt19937 gen;
	boost::mt19937::result_type min, max;
	void initialize(void);
public:
	void seed(int s);
	RandomGenerator();
	RandomGenerator(int Seed);
	double RandomDouble(void);
	
	//generator a random number following a certain distribution, where inverseCP is the inverse of the cumulative distribution function.
	double RandomDouble(const std::function<double(double)> &inverseCp);

	//generate a Gaussian random number of mean = 0 and standard deviation = std
	double RandomDouble_normal(double std) {
		boost::normal_distribution<> nd(0.0, std);
		boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(this->gen, nd);
		return var_nor();
	}
	int PoissonRNG(double mean);

	inline int RandomInt(void) {return this->gen();}	
};

void GetRandomVector(int Dimension, RandomGenerator & gen, double * result);


#endif
