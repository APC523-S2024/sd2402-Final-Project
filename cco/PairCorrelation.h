/*
Header file for computing pair correlation function (point patterns) and 2-point probability function (sphere packings)

Sam Dawley
5/2024

References
...
*/



#ifndef PAIRCORRELATION_INCLUDED
#define PAIRCORRELATION_INCLUDED

#include <omp.h>
#include <vector>

#include "PeriodicCellList.h"

// Class to generate histogram from data
class HistogramGenerator
{
public:

    // structure to store individual bins for histogram
	struct bin
	{
		size_t Count; //	The frequency in this bin.
		double Start; //	smallest value of the range of this bin.

        // Generate a bin whose smallest value in the range is 'Start' and frequency is 'Count'
		bin(size_t Count, double Start) : Count(Count), Start(Start){}

		// default constructor for an empy bin
		bin() : Count(0), Start(0){}
	};

    // histogram is just a series of bins
	std::vector<bin> bins;

private:

	void Iteration(double MinDist, double MaxDist, std::vector<double>::iterator begin, std::vector<double>::iterator end, double MoreData, double MinBinWidth, double TotalRange)
	{
		double count=static_cast<double>(end-begin);
		double yError = 1.0/std::sqrt(count*MoreData); // estimated relative error in y direction
		double xError = (MaxDist-MinDist)/TotalRange/2.0;

		if (yError > xError || count < 3 || MaxDist - MinDist < MinBinWidth) {
			if (bins.empty()) { // create new bin and insert empty bins before first nonempty
				size_t EmptyBinCount = 100;

				if (EmptyBinCount > 0) {
					double EmptyBinWidth = 0.2*TotalRange/EmptyBinCount;

					for (size_t i = 0; i < EmptyBinCount; i++) {
						bin temp;
						temp.Count=0;
						temp.Start=MinDist-0.2*TotalRange+i*EmptyBinWidth;
						bins.push_back(temp);
					}

				} else if (MinDist > 0) { // always have a bin at 0
					bin temp;
					temp.Count=0;
					temp.Start=0.0;
					bins.push_back(temp);
				}

				bin temp;
				temp.Count=end-begin;
				temp.Start=MinDist;
				bins.push_back(temp);

			} else if (MinDist == bins.back().Start) {
                // merge this into previous bin because
                // bin is identical to previous || count is too small and bin cannot be precise
				bins.back().Count += end - begin;

			} else { // create new bin
				bin temp;
				temp.Count=end-begin;
				temp.Start=MinDist;
				bins.push_back(temp);
			}

		} else { // divide this chunk
			std::vector<double>::iterator mid=begin+(end-begin)/2;
			std::nth_element(begin, mid, end);
			Iteration(MinDist, *mid, begin, mid, MoreData, MinBinWidth, TotalRange);
			Iteration(*mid, MaxDist, mid, end, MoreData, MinBinWidth, TotalRange);
		} // if (yError > xError || count < 3 || MaxDist - MinDist < MinBinWidth)
	}

public:
    /*
    Count all data and generate bins using adaptive method
    Set 'Count' of each bin to their frequency in measured data
    'data' is changed and returned
    'MoreData' = X indicates that X more data needs to be sorted into bins
    'ResolutionPreference' >1 to improve x-direction (bins) resolution, <1 to improve y-direction resolution (counts)
    */
	void GenerateBins(std::vector<double> & data, double MoreData, double ResolutionPreference, double MinBinWidth=0)
	{
		if (data.empty()) {
			std::cerr<<"Error in HistogramGenerator::GenerateBins : no data!\n";
			return;
		}

		double Max = *std::max_element(data.begin(), data.end());
		double Min = *std::min_element(data.begin(), data.end());
		Iteration(Min, Max, data.begin(), data.end(), MoreData*ResolutionPreference, MinBinWidth, Max-Min); 

        // bins generated above often too small; merge here
		std::vector<bin> newBins;
		for (auto iter = this->bins.begin();;) {
			if(iter==bins.end()) {
                break;

            } else if (iter == bins.end()-1) {
				newBins.push_back(*iter);
				break;

			} else {
				bin temp;
				temp.Count=iter->Count+(iter+1)->Count;
				temp.Start=iter->Start;
				newBins.push_back(temp);
				iter+=2;
			}
		}

		double dist = 0.2*(Max - Min) / 50.0;

		for(int i = 0; i < 50; i++) {
			bin temp;
			temp.Start=Max+i*dist;
			temp.Count=0;
			newBins.push_back(temp);
		}

		std::swap(newBins, this->bins);
	}

    /*
    Report new data and increase corresponding bin count by 1
    Call after calling GenerateBins()
    THIS FUNCTION IS THREAD SAFE
    NOTE: if call to Report() done before calling GenerateBins(), assertion will fail
    */
	void Report(const double & Data)
	{
		assert(this->bins.empty() == false);

		if (Data < bins[0].Start) {
			std::cerr<<"Warning in HistogramGenerator::Report : Data is out of range!\n";
			return;
		}

		bin temp;
		temp.Start = Data;
		std::vector<bin>::iterator loc = std::upper_bound(bins.begin(), bins.end(), temp, [](const bin & left, const bin & right) -> bool
        {
            return left.Start<right.Start;
        });

		assert(loc != bins.begin());
		loc--; // decrease it by one because std::upper_bound returns the first iterator which is GREATER. However, our bin::Start is the start location of the bin

#pragma omp atomic
		loc->Count++;
	}
};

/* 
Compute isotropic pair correlation function for point pattern
MaxDistance is max distance to evaluate g_2(r)
Result is cleared before being repopulated with pair correlation data
*/
void IsotropicTwoPairCorrelation(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<GeometryVector> & Result, size_t SampleDistanceSize=5000000, double ResolutionPreference=1.0);
void IsotropicTwoPairCorrelation(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<GeometryVector> & Result, HistogramGenerator & HGen);

/*
Compute anisotropic pair correlation function
'Result' is cleared before being populated by 3D GeometryVectors with elements
    r, theta, g_2(r)
Bins are generated adaptively
'ResolutionPreference' defined as usual
Theta bins are fixed
*/
void TwoPairCorrelation_2DAnisotropic(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<std::vector<GeometryVector> > & Result, size_t NumDirectionalBins, std::vector<std::vector<GeometryVector> > * pError=nullptr, size_t SampleDistanceSize=2000000, double ResolutionPreference=1.0);

/*
Calculate particle nearest-neighbor probability density
'Result' is cleared before being repopulated by probability density calculated from configuration; elements are
    r, p(r), \delta r, \delta g_2(r)
latter two quantities are uncertainty
'ResolutionPreference' >1 to improve x-direction (r) resolution, <1 to improve y-direction resolution (g_2)
*/
void NearestNeighborDistrubution(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, std::vector<GeometryVector> & Result, size_t SampleDistanceSize=500000, double ResolutionPreference=1.0);

// Calculate void nearest-neighbor probability density H_V(r)
// Each configuration will be sampled OverSampling*NumParticle() times
void HvDistrubution(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, std::vector<GeometryVector> & Result, size_t SampleDistanceSize=500000, double ResolutionPreference=1.0, double OverSampling=1.0);

// Returns minimum pair distance within configuration
// If pi and pj are not nullptr, *pi and *pj will be set to the particles that are closest to each other
double MinDistance(const Configuration & a, size_t * pi=nullptr, size_t * pj=nullptr);
double MeanNearestNeighborDistance(const Configuration & a);

// Convert histogram to PDF of CDF
// Returns total count HistogramGenerator has
size_t HistogramToPDF(const HistogramGenerator & HGen, std::vector<GeometryVector> & result);
size_t HistogramToCDF(const HistogramGenerator & HGen, std::vector<GeometryVector> & result);

// Compute 2-pt probability function of prescribed sphere packing at given separation x
// Intersection volumes of spheres are computed exactly 
// NOTE: packing defined by 'pConfig' SHOULD NOT CONTAIN PARTICLE OVERLAPS
// x is cartesian coordinates of seperation vector x
inline double getS2_exact(const SpherePacking & pConfig, const GeometryVector & x);

/*
Compute isotropic autocovariance for sphere packings
Rmax is max distance to evaluate autocovariance function
'Result' defined as usual
'dR' is bin width; if not specified default is half the smallest sphere radius
*/
void IsotropicAutocovariance(std::function<const SpherePacking(size_t i)> GetConfig, size_t NumConfigs, double Rmax, std::vector<GeometryVector> & Result, double dR = 0.1L, double sampling_density = 4.0);

// Structure for conveying extra parameters to IsotropicAutocovariance_hybrid() below
struct Parameters_Autocovariance {
	double Rmin; // the minimal radius, from which autocovariance function is computed.
	size_t TotalSamplingPts, // For each configuration, "TotalSamplingPts/ NumConfigs" random sampling points are thrown. 
		NumThreads;	// The number of threads.

	RandomGenerator rng;
	Parameters_Autocovariance() { 
		Rmin = -1.0;
        TotalSamplingPts = 10000;
        rng.seed(0);
        NumThreads = 1;
	};
};

// Isotropic autocovariance with extra parameters, e.g., minimal radius and total sampling number
void IsotropicAutocovariance_hybrid(std::function <const SpherePacking(size_t i)> GetConfig, size_t NumConfigs, double Rmax, std::vector<GeometryVector> & Result, double dR = 0.1L, const Parameters_Autocovariance & extra_parameters= Parameters_Autocovariance());

#endif