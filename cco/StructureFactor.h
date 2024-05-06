/*
Functions for computing (an)isotropic structure factor and spectral density of point patterns/packings

Sam Dawley
5/2024

References
...
*/

#ifndef STRUCTUREFACTOR_INCLUDED
#define STRUCTUREFACTOR_INCLUDED

#include <cstdio> // writing output vectors
#include <functional>
#include <vector>

#include "GeometryVector.h"
#include "PeriodicCellList.h"

/** \file StructureFactor.h
 *	\brief Header file for computing the structure factor of point configurations,
	spectral density of sphere packings, and spectral density for packings of nonspherical particles (optional). */

/** Find all reciprocal lattice vectors (wavevectors) of a given periodic simulation box.
 *  It is modified to deal with errors.
 *	@param tempList	A periodic simulation box.
 *	@param (CircularKMax, LinearKMax)	Find all reciprocal lattice vectors (excluding parity pairs) whose magnitude is smaller than CircularKMax,
										and then fina all integral multiples of the aforementioned vectors whose magnitudes are smaller than LinearKMax.
 *	@param SampleProbability	When <1, randomly choose wavevectors within CircularKMax by the prescribed probability.*/
std::vector<GeometryVector> GetKs(const PeriodicCellList<Empty> & tempList, double CircularKMax, double LinearKMax, double SampleProbability = 1.0);

//calculate the Structure factor of certain k
double StructureFactor(const Configuration & Config, const GeometryVector & k);
//calculate the Structure factof with weights; here config.getCharacteristics(i) = the weight on the i-th particle
double WeightedStructureFactor(const PeriodicCellList<std::complex<double>> & Config, const GeometryVector & k);
//calculate structure factor of all ks in the range CircularKMax (i.e. for all ks that abs(k)<CircularKMax) and all of their multiples in the range LinearKMax
//Results has the following form:
//( abs(k), S(k), KPrecision, \delta S(k) )
//GetConfigsFunction should return configurations of index i when called
//if SampleProbability is <1, then for any k point satisfying the above condition there is SampleProbability probability that it will be used to calculate S(k) and (1-SampleProbability) probability that it will not be used.
void IsotropicStructureFactor(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision=0.01, double SampleProbability=1.0);
void IsotropicStructureFactor_weighted(std::function<const PeriodicCellList<std::complex<double>>(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision = 0.01, double SampleProbability = 1.0);
//calculate structure factor of all ks in the cubic box [-limit, limit]^d.
//dk : structure factors at ks in [k,k+dk]^d will be averaged: if dk = 1, structure factors are not averaged
//return coordinates of the Grid
//For contourplot in Matlab
std::vector<GeometryVector> DirectionalStructureFactor(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, long limit, std::vector<GeometryVector> & Results, int dk=1, double SampleProbability = 1.0);

//fourier transform of the indicator function of a hypersphere of radius R
double Mtilde(double k, double R, DimensionType d);

double SpectralDensity(const SpherePacking & Config, const GeometryVector & k);
inline double SpectralDensity_Lstorage(const SpherePacking & Config, const GeometryVector & k) {
	std::complex<double> rho;
	size_t NumParticle = Config.NumParticle();
	double kl = std::sqrt(k.Modulus2());
	DimensionType d = Config.GetDimension();
	double mt;
	for (size_t i = 0; i<NumParticle; i++)
	{
		std::complex<double> a = std::exp(std::complex<double>(0, k.Dot(Config.GetCartesianCoordinates(i))));
		double R = Config.GetCharacteristics(i);
		mt = Mtilde(kl, R, d);
		rho += a * mt;
	}
	return (rho.real()*rho.real() + rho.imag()*rho.imag()) / Config.PeriodicVolume();
}
void IsotropicSpectralDensity(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision);
//For extremely large sphere packings of high polydispersity
void IsotropicSpectralDensity_Lstorage(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision);

//For contourplot in Matlab
//Return Coordinates of Grid
//std::vector<GeometryVector> DirectionalSpectralDensity(const std::vector<SpherePacking> config, long limit, std::vector<GeometryVector> & Results, double dk = 1, double SampleProbability = 1.0);
std::vector<GeometryVector> DirectionalSpectralDensity(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, long limit, std::vector<GeometryVector> & Results, double dk = 1, double SampleProbability = 1.0);


//Compute data of spectral densities in both directional and isotropic manners. 
std::vector<GeometryVector> DirectionalSpectralDensity(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, long KMax, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & results_directional, std::vector<GeometryVector> & results_isotropic, double KPrecision = 0.01, int PixelPrecision = 1, double SampleProbability = 1.0);

#if defined USE_SHAPES
#include "NonsphericalParticles/Dispersions.h"
#include "NonsphericalParticles/ParticleShapes.h"
/* Author	: Jaeuk Kim
* Email	: phy000.kim@gmail.com
* Data		: April 16th 2019
* This small section is a collection of fnctions
to compute spectral density for packings of nonspherical particles.
Please use a preprocessor "USE_SHAPE" to activate this part. */

/** A function to compute spectral density of a dispersion at a wavevector k.
* It assumes that particles are not overlapping.
* It follows the convention (Fourier transform  = \int d{\vect{r}} e^{i k}    )
* @param config	A Dispersion object that describes a packing of (presumably) nonoverlapping nonspherical particles.
* @param k	A wavevector at which we compute the spectral density.
* @return A value of spectral density at k. */
double SpectralDensity_nonspherical(const Dispersion & config, const GeometryVector &k);

/** A function to compute an orientationally averaged spectral density.
 * We assume that all dispersions are nonoverlapping. @see SpectralDensity_nonspherical().
 * It computes values of the spectral density at all wavevectors searched by two parameters (CircularKmax, LinearKmax), and
	then computes the averages of spectral density at wavevectors whose magnitude is between [n*KPrecision, (n+1)*KPrecision].
 * @param[in] GetConfiguration	A Lambda function to generate Dispersion objects that we have interest in.
 * @param[in] NumConfigs	The number of configurations used in computation.
 * @param[in] (CircularKmax, LinearKmax)	Searching radii for wavevectors.
				Search all wavevectors (except for their parity pairs) up to k=CircularKmax,
				and search all wavevectors that are scalar multiplications of the aforementioned wavevectors up to k=LinearKmax.
 * @param[out] result		A table of spectral density.
				result[i].x[0] = wavenumber.
				result[i].x[1] = spectral density.
				result[i].x[2] = standard error of k
				result[i].x[3] = standard error of spectral density.
 * @param[in] KPrecision	Bin witdth of wavenumbers.*/
void IsotropicSpectralDensity_nonspherical(std::function<const Dispersion(size_t)> GetConfiguration, size_t NumConfigs,
	double CircularKmax, double LinearKMax, std::vector<GeometryVector> & result, double KPrecision = 0.01);

/** A function to compute spectral density to draw a contour plot.
 * @param[in] GetConfiguration	A Lambda function to generate Dispersion objects that we have interest in.
 * @param[in] NumConfigs	The number of configurations used in computation.
 * @param[in] limit	The number of wavevectors to consider. Wavevectors are k = i*k_1 + j*k_2 + ..., where i, j, ... are integers between -limit and limit.
 * @param[out] (ks, chik)	ks[i] is the ith wavevector, and chik[i] is the spectral density at this wavevector.
 * @param[in] dk	The number of pixels over which the spectral density is averaged.*/
void DirectionalSpectralDensity_nonspherical(std::function<const Dispersion(size_t)> GetConfigsFunction, size_t NumConfigs,
	long limit, std::vector<GeometryVector> & ks, std::vector<GeometryVector> & chik, double dk = 1);


#endif





#endif