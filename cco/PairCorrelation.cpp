/*
File for computing pair correlation function (point patterns) and 2-point probability function (sphere packings)

Sam Dawley
5/2024

References
...
*/

#include <omp.h>

#include "PairCorrelation.h"

// Generates adaptive bins calculated after partial pair distances known, uses less memory
void IsotropicTwoPairCorrelation(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<GeometryVector> & Result, size_t SampleDistanceSize, double ResolutionPreference)
{
	if (!(MaxDistance > 0.0)) {
        return;
    }

	// list of pairwise distances
	std::cout << "Computing Pair Corrletation Function g2(r)";
	progress_display pd(NumConfigs);

	std::vector<double> distances;
	DimensionType dim;

	size_t NumConfigsSampled=0;
	size_t TotalAtomCount=0;
	double AverageDensity=0;
	signed long StartParticle=0;
	signed long NumParticle=1;

	for (size_t j = 0; j < NumConfigs; j++) {
		Configuration Config = GetConfigsFunction(j);
		Config.SetCellSize(0.35*MaxDistance);
		NumParticle = Config.NumParticle();

		if (j==0) {
            dim = Config.GetDimension();
        }

		size_t NumParticle = Config.NumParticle();

		// Pre-allocate
		Config.IterateThroughNeighbors(GeometryVector(Config.GetDimension()), MaxDistance, [](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
		{
		});

		for (signed long i = 0; i < NumParticle; i++) {
			Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(i), MaxDistance, [&distances, &i](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
			{
				if (SourceAtom > i || (SourceAtom == i && LatticeShift.Modulus2() != 0) ) {
                    distances.push_back(std::sqrt(shift.Modulus2()));
                }
			});

			if (distances.size() > SampleDistanceSize) {
				StartParticle=i+1;
				break;
			}
		}

		if (distances.size() > SampleDistanceSize) {
            break;
        }

		NumConfigsSampled++;
		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();
		pd++;
	}

	// Find the lowest pair distance
	if (distances.empty()) {
		std::cerr<<"Warning in IsotropicTwoPairCorrelation : Pair distance not found in specified cutoff!\n";
		Result.clear();
		return;
	}

	Result.clear();
	
	HistogramGenerator HGen;
	HGen.GenerateBins(distances, static_cast<double>(NumConfigs)/((double)(NumConfigsSampled)+(double)(StartParticle)/NumParticle), ResolutionPreference);

	// discard empty bins at the end; such bins are useful when we don't have an upper bound of data (e.g. nearest neighbor distance binning)
	// here, they are useless
	while (HGen.bins.size() > 1 && HGen.bins.back().Count == 0) {
        HGen.bins.pop_back();
    }

	// the bin counts
	for (size_t j = NumConfigsSampled; j < NumConfigs; j++) {
		Configuration Config = GetConfigsFunction(j);
		Config.SetCellSize(0.35*MaxDistance);
		Config.RefineCellList();
		size_t NumParticle=Config.NumParticle();
		Config.PrepareIterateThroughNeighbors(MaxDistance);

#pragma omp parallel
		{
#pragma omp for schedule(guided)
			for (signed long i = StartParticle; i < NumParticle; i++) {
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(i), MaxDistance, [&HGen, &i](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
				{
					if (SourceAtom > i || (SourceAtom == i && LatticeShift.Modulus2() != 0) ) {
						double dist=std::sqrt(shift.Modulus2());
						HGen.Report(dist);
					}
				});
			}
		}

		StartParticle = 0;
		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle() / Config.PeriodicVolume();
		pd++;
	}

	AverageDensity /= NumConfigs;

	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++) {
		double BinEnd;
		if (iter->Start >= MaxDistance) {
            continue;

        } else if (iter != HGen.bins.end()-1) {
			BinEnd = std::min(iter[1].Start, MaxDistance);

        } else {
			BinEnd = MaxDistance;
        }
		
		GeometryVector temp(4);
		temp.x[0] = (BinEnd+iter->Start)/2;
		temp.x[1] = (double)iter->Count / ((HyperSphere_Volume(dim, BinEnd)-HyperSphere_Volume(dim, iter[0].Start))*0.5*AverageDensity*TotalAtomCount);
		temp.x[2] = (temp.x[0]-iter[0].Start); 

		if (iter->Count != 0) {
            temp.x[3] = temp.x[1]/std::sqrt(static_cast<double>(iter->Count));
        } else {
			temp.x[3] = 0.0;
        }

		// if bin is too small, inaccurate and discard
		if (temp.x[0] > 0) {
			Result.push_back(temp);
        }
	}
    std::cout<<"done!\n";

}

void IsotropicTwoPairCorrelation(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<GeometryVector> & Result, HistogramGenerator & HGen)
{
	if (!(MaxDistance > 0)) {
        return;
    }

	// get list of pairwise distances
    std::cout << "computing g_2";
	progress_display pd(NumConfigs);

	std::vector<double> distances;

	DimensionType dim;

	size_t TotalAtomCount = 0;
	double AverageDensity = 0;
	
	Result.clear();

	// bin counts
	for (size_t j = 0; j < NumConfigs; j++) {
		Configuration Config = GetConfigsFunction(j);

		if (j == 0) {
            dim = Config.GetDimension();
        }

		Config.SetCellSize(0.35*MaxDistance);
		Config.RefineCellList();
		size_t NumParticle = Config.NumParticle();
		
		Config.IterateThroughNeighbors(GeometryVector(Config.GetDimension()), MaxDistance, [](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
		{
		});

		//Configuration::particle * p0=Config.GetParticle(0);
#pragma omp parallel
		{
#pragma omp for schedule(guided)
			for (signed long i = 0; i < NumParticle; i++) {
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(i), MaxDistance, [&HGen, &i](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) ->void
				{
					if (SourceAtom > i || (SourceAtom == i && LatticeShift.Modulus2() != 0)) {
						double dist = std::sqrt(shift.Modulus2());
						HGen.Report(dist);
					}
				});
			}
		}

		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle() / Config.PeriodicVolume();
		pd++;
	}

	AverageDensity /= NumConfigs;

	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++) {
		double BinEnd;
		if (iter->Start >= MaxDistance) {
            continue;

        } else if (iter != HGen.bins.end() - 1) {
			BinEnd = std::min(iter[1].Start, MaxDistance);

        } else {
			BinEnd = MaxDistance;
        }

		GeometryVector temp(4);
		temp.x[0] = (BinEnd + iter->Start) / 2;
		temp.x[1] = iter->Count / ((HyperSphere_Volume(dim, BinEnd) - HyperSphere_Volume(dim, iter[0].Start))*0.5*AverageDensity*TotalAtomCount);
		temp.x[2] = (temp.x[0] - iter[0].Start);

		if (iter->Count != 0)
			temp.x[3] = temp.x[1] / std::sqrt(static_cast<double>(iter->Count));//\delta g_2(r)
		else
			temp.x[3] = 0.0;

		// if bin is too small, inaccurate and discard
		if (temp.x[0]>0)
			Result.push_back(temp);
	}
    std::cout << "done!\n";
}

void TwoPairCorrelation_2DAnisotropic(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double MaxDistance, std::vector<std::vector<GeometryVector> > & Result, size_t NumDirectionalBins, std::vector<std::vector<GeometryVector> > * pError, size_t SampleDistanceSize, double ResolutionPreference)
{
	if (!(MaxDistance > 0)) {
        return;
    }
	
    // get list of pairwise distances
    std::cout<<"Generate g_2, sampling pair distances...";
	std::vector<double> distances;

	DimensionType dim;

	size_t NumConfigsSampled=0;
	size_t TotalAtomCount=0;
	double AverageDensity=0;
    
	for (size_t j = 0; j < NumConfigs; j++) {
		if (Verbosity > 3 || (Verbosity > 2 && j % 100==0) ) {
            std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
        }

		Configuration Config = GetConfigsFunction(j);

		if (j == 0) {
			dim = Config.GetDimension();

			if (dim != 2) {
				std::cerr<<"Error in TwoPairCorrelation_2DAnisotropic : Configuration is not two-dimensional!\n";
				std::cerr<<"This function only support two dimensional configurations!\n";
				assert(false);
			}
		}
		size_t NumParticle=Config.NumParticle();
		
		Config.IterateThroughNeighbors(GeometryVector(Config.GetDimension()), MaxDistance, [](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) ->void
		{
		});

		for (signed long i = 0; i < NumParticle; i++) {
			Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(i), MaxDistance, [&distances, &i](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) ->void
			{
				if (SourceAtom > i || (SourceAtom == i && LatticeShift.Modulus2() != 0) ) {
                    distances.push_back(std::sqrt(shift.Modulus2()));
                }
			});
		}

		NumConfigsSampled++;
		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();

		if (distances.size() > SampleDistanceSize) {
            break;
        }
	}

    std::cout<<"generating bins...";

	// find smallest pairwise distances
	if (distances.empty()) {
		std::cerr<<"Warning in IsotropicTwoPairCorrelation : Pair distance not found in specified cutoff!\n";
		Result.clear();
		return;
	}

	
	Result.clear();
	
	HistogramGenerator HGen;
	HGen.GenerateBins(distances, static_cast<double>(NumConfigs)/NumConfigsSampled/NumDirectionalBins, ResolutionPreference);

	// discard empty bins at the end. Such bins are useful when we don't have an upper bound of data (e.g. nearest neighbor distance binning)
	// here, they are useless
	while (HGen.bins.size() > 1 && HGen.bins.back().Count == 0) {
        HGen.bins.pop_back();
    }

	// clear the counts, we need directional g2(r), so the previous isotropic counts are useless
	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++) {
        iter->Count=0;
    }

	// generate directional bins
	std::vector<HistogramGenerator> DirectionalHGen(NumDirectionalBins, HGen);

	// redo the bin counts
    std::cout<<"counting pair distances...";
	TotalAtomCount = 0;
	AverageDensity = 0.0;

	for (size_t j = 0; j < NumConfigs; j++) {
		if (Verbosity > 3 || (Verbosity > 2 && j % 100 ==0 )) {
            std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
        }

		Configuration Config = GetConfigsFunction(j);
		size_t NumParticle = Config.NumParticle();
		
		Config.IterateThroughNeighbors(GeometryVector(Config.GetDimension()), MaxDistance, [](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
		{
		});

//#pragma omp parallel
		{
//#pragma omp for schedule(guided)
			for (signed long i = 0; i < NumParticle; i++) {
				
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(i), MaxDistance, [&DirectionalHGen, &i, &NumDirectionalBins](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) -> void
				{
					if (SourceAtom > i || (SourceAtom == i && LatticeShift.Modulus2() != 0) ) {
						double dist = std::sqrt(shift.Modulus2());

						// angle within interval [0, 1]
						double angle = 0.5 - std::atan(shift.x[0]/shift.x[1])/::pi;

						// turns 1 into 0 for binning
						while (!(angle < 1) ) {
                            angle -= 1;
                        }

						size_t AngleBin = std::floor(angle*NumDirectionalBins);
						DirectionalHGen[AngleBin].Report(dist);
					}
				});
			}
		}

		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle() / Config.PeriodicVolume();
	}
	AverageDensity /= NumConfigs;

	Result.resize(NumDirectionalBins, std::vector<GeometryVector>() );
	if (pError != nullptr) {
        pError->resize(NumDirectionalBins, std::vector<GeometryVector>() );
    }

	for (size_t i = 0; i < NumDirectionalBins; i++) {
		for (auto iter = DirectionalHGen[i].bins.begin(); iter != DirectionalHGen[i].bins.end(); iter++) {
			double BinEnd;

			if(iter->Start >= MaxDistance) {
                continue;

            } else if (iter != DirectionalHGen[i].bins.end()-1) {
				BinEnd = std::min(iter[1].Start, MaxDistance);

            } else {
				BinEnd = MaxDistance;
            }

			GeometryVector temp(4);
			temp.x[0]=(BinEnd+iter->Start) / 2;
			temp.x[1]=( (double)(i) / NumDirectionalBins + 0.5 )* ::pi;
			temp.x[2]=iter->Count / ((HyperSphere_Volume(dim, BinEnd)-HyperSphere_Volume(dim, iter->Start))*0.5*AverageDensity*TotalAtomCount)*NumDirectionalBins;
			Result[i].push_back(temp);

			if (pError != nullptr) {
				temp.x[0] = (temp.x[0]-iter[0].Start);
				temp.x[1] = ::pi/(2*NumDirectionalBins);
				if (iter->Count != 0) {
					temp.x[2] = temp.x[2]/std::sqrt(static_cast<double>(iter->Count));

                } else {
					temp.x[2]=0.0;
                }

				pError->at(i).push_back(temp);
			}
		}
	}

    std::cout<<"done!\n";
}

void NearestNeighborDistrubution(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, std::vector<GeometryVector> & Result, size_t SampleDistanceSize, double ResolutionPreference)
{
	// get list of pairwise distances
	std::cout << "Computing Particle Nearest-Neighbor Density H_p(r)";
	progress_display pd(NumConfigs);

	std::vector<double> distances;
	DimensionType dim;

	size_t NumConfigsSampled = 0;
	size_t TotalAtomCount = 0;
	double AverageDensity = 0;

	for (size_t j = 0; j < NumConfigs; j++) {
		pd++;
		Configuration Config = GetConfigsFunction(j);

		if (j == 0) {
            dim = Config.GetDimension();
        }

		size_t NumParticle = Config.NumParticle();
		double TypicalLength = std::pow(Config.PeriodicVolume()/Config.NumParticle(), 1.0/Config.GetDimension())*1.5;

		for (size_t j = 0; j < Config.NumParticle(); j++) {
			std::vector<GeometryVector> neighbors;
			double l = TypicalLength;

			while (neighbors.size() < 2) {
				neighbors.clear();
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(j), l, [&neighbors](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t Sourceparticle) -> void
                {
					neighbors.push_back(shift);
				});

				l *= 2;
			}

			std::partial_sort(neighbors.begin(), neighbors.begin()+2, neighbors.end(), [](const GeometryVector & left, const GeometryVector & right)->bool{return left.Modulus2()<right.Modulus2();});
			distances.push_back(std::sqrt(neighbors[1].Modulus2()));
		}

		NumConfigsSampled++;
		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();

		if (distances.size() > SampleDistanceSize) {
            break;
        }
	}

	Result.clear();
	
	HistogramGenerator HGen;
	HGen.GenerateBins(distances, static_cast<double>(NumConfigs)/NumConfigsSampled, ResolutionPreference);

	// redo bin counts
	for (size_t j = NumConfigsSampled; j < NumConfigs; j++) {
		pd++;
		Configuration Config = GetConfigsFunction(j);
		size_t NumParticle = Config.NumParticle();

		for (size_t j = 0; j < Config.NumParticle(); j++) {
			std::vector<GeometryVector> neighbors;
			double TypicalLength = std::pow(Config.PeriodicVolume()/Config.NumParticle(), 1.0/Config.GetDimension())*1.5;
			double l = TypicalLength;

			while(neighbors.size() < 2) {
				neighbors.clear();
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(j), l, [&neighbors](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t Sourceparticle) -> void
				{
					neighbors.push_back(shift);
				});
				l *= 2;
			}

			std::partial_sort(neighbors.begin(), neighbors.begin()+2, neighbors.end(), [](const GeometryVector & left, const GeometryVector & right)->bool{return left.Modulus2()<right.Modulus2();});
			HGen.Report(std::sqrt(neighbors[1].Modulus2()));
		}

		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();
	}

	AverageDensity /= NumConfigs;

	size_t TotalCount=0;
	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++)
		TotalCount += iter->Count;

	for (auto iter = HGen.bins.begin(); iter!=HGen.bins.end(); iter++) {
		GeometryVector temp(4);

		if (iter != HGen.bins.end()-1) {
            temp.x[0]=((iter[0].Start)+(iter[1].Start))/2.0;

        } else {
			continue;
        }

		if (iter == HGen.bins.begin()) {
			if (iter->Start == 0.0 && (iter[2].Start-iter[1].Start) > iter[1].Start && iter->Count == 0) {
				continue;
            }
		}

		temp.x[2] = (temp.x[0]-iter[0].Start);
		temp.x[1] = static_cast<double>(iter->Count)/TotalCount/(2*temp.x[2]);

		if (iter->Count != 0) {
			temp.x[3] = temp.x[1]/std::sqrt(static_cast<double>(iter->Count));

        } else {
			temp.x[3] = 0.0;
        }

		Result.push_back(temp);
	}
}

void HvDistrubution(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, std::vector<GeometryVector> & Result, size_t SampleDistanceSize, double ResolutionPreference, double OverSampling)
{
	RandomGenerator gen(81479058);
	// list of pairwise distances
    std::cout<<"Computing H_v(r)";
	progress_display pd(NumConfigs);

	std::vector<double> distances;

	DimensionType dim;

	size_t NumConfigsSampled = 0;
	size_t TotalAtomCount = 0;
	double AverageDensity = 0;

	for (size_t j = 0; j < NumConfigs; j++) {
		pd++;
		Configuration Config = GetConfigsFunction(j);

		if (j==0) {
            dim=Config.GetDimension();
        }

		size_t NumParticle = Config.NumParticle();
		double TypicalLength = std::pow(Config.PeriodicVolume()/Config.NumParticle(), 1.0/Config.GetDimension())*1.5;

		for (size_t j = 0; j < Config.NumParticle()*OverSampling; j++) {
			std::vector<GeometryVector> neighbors;
			double l=TypicalLength;
			GeometryVector temp(dim);

			for (int i = 0; i < dim; i++) {
                temp.x[i]=gen.RandomDouble();
            }
			
            while (neighbors.size() < 2) {
				neighbors.clear();
				Config.IterateThroughNeighbors(temp, l, [&neighbors](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t Sourceparticle) -> void
				{
					neighbors.push_back(shift);
				});
				l *= 2;
			}

			std::partial_sort(neighbors.begin(), neighbors.begin()+1, neighbors.end(), [](const GeometryVector & left, const GeometryVector & right)->bool{return left.Modulus2()<right.Modulus2();});
			distances.push_back(std::sqrt(neighbors[0].Modulus2()));
		}

		NumConfigsSampled++;
		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();

		if (distances.size() > SampleDistanceSize) {
            break;
        }
	}

	Result.clear();
	
	HistogramGenerator HGen;
	HGen.GenerateBins(distances, static_cast<double>(NumConfigs)/NumConfigsSampled, ResolutionPreference);

	// redo bin counts
	for (size_t j = NumConfigsSampled; j<NumConfigs; j++) {
		pd++;
		Configuration Config = GetConfigsFunction(j);
		size_t NumParticle=Config.NumParticle();

		for (size_t j = 0; j < Config.NumParticle(); j++) {
			std::vector<GeometryVector> neighbors;
			double TypicalLength = std::pow(Config.PeriodicVolume()/Config.NumParticle(), 1.0/Config.GetDimension())*1.5;
			double l = TypicalLength;

			while (neighbors.size() < 2) {
				neighbors.clear();
				Config.IterateThroughNeighbors(Config.GetRelativeCoordinates(j), l, [&neighbors](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t Sourceparticle) -> void
				{
					neighbors.push_back(shift);
				});
				l *= 2;
			}

			std::partial_sort(neighbors.begin(), neighbors.begin()+2, neighbors.end(), [](const GeometryVector & left, const GeometryVector & right)->bool{return left.Modulus2()<right.Modulus2();});
			HGen.Report(std::sqrt(neighbors[1].Modulus2()));
		}

		TotalAtomCount += Config.NumParticle();
		AverageDensity += Config.NumParticle()/Config.PeriodicVolume();
	}
    
	AverageDensity /= NumConfigs;

	size_t TotalCount=0;
	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++) {
        TotalCount += iter->Count;
    }

	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++) {
		GeometryVector temp(4);

		if (iter != HGen.bins.end()-1) {
            temp.x[0]=((iter[0].Start)+(iter[1].Start))/2.0;
        } else {
			continue;
        }

		if (iter == HGen.bins.begin()) {
			if (iter->Start == 0.0 && (iter[2].Start-iter[1].Start) > iter[1].Start && iter->Count == 0) {
                continue;
            }
		}

		temp.x[2] = (temp.x[0]-iter[0].Start);
		temp.x[1] = static_cast<double>(iter->Count)/TotalCount/(2*temp.x[2]);

		if (iter->Count != 0) {
            temp.x[3]=temp.x[1]/std::sqrt(static_cast<double>(iter->Count));
        } else {
			temp.x[3]=0.0;
        }

		Result.push_back(temp);
	}
    std::cout<<"done!\n";
    
}

//return the minimum pair distance in the configuration
double MinDistance(const Configuration & a, size_t * pi, size_t * pj)
{
	DimensionType dim=a.GetDimension();
	double UnitSphereVolume= ::HyperSphere_Volume(dim, 1.0);
	double result=std::pow(a.PeriodicVolume()/a.NumParticle()/UnitSphereVolume, (double)(1.0)/dim)*2;
	for(size_t i=0; i<a.NumParticle(); i++)
	{
		GeometryVector loc=a.GetRelativeCoordinates(i);
		a.IterateThroughNeighbors(loc, result, [&](const GeometryVector &shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, const size_t SourceAtom) ->void
		{
			if (i == SourceAtom)
				return;
			double dis=std::sqrt(shift.Modulus2());
			if (dis < result)
			{
				result = dis;
				if (pi != nullptr)
					(*pi)=i;
				if (pj != nullptr)
					(*pj)=SourceAtom;
			}
		});
	}
	return result;
}

double MeanNearestNeighborDistance(const Configuration & a) {
	double sum=0.0;
	for (size_t j = 0; j < a.NumParticle(); j++) {
		std::vector<GeometryVector> neighbors;
		double TypicalLength = std::pow(a.PeriodicVolume()/a.NumParticle(), 1.0/a.GetDimension())*1.5;
		double l = TypicalLength;
		while (neighbors.size() < 2) {
			neighbors.clear();
			a.IterateThroughNeighbors(a.GetRelativeCoordinates(j), l, [&neighbors](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t Sourceparticle) -> void {
				neighbors.push_back(shift);
			});
			l *= 2;
		}

		std::partial_sort(neighbors.begin(), neighbors.begin()+2, neighbors.end(), [](const GeometryVector & left, const GeometryVector & right)->bool{return left.Modulus2()<right.Modulus2();});
		sum += std::sqrt(neighbors[1].Modulus2());
	}
	return sum/a.NumParticle();
}

size_t HistogramToPDF(const HistogramGenerator & HGen, std::vector<GeometryVector> & result)
{
	result.clear();
	size_t TotalCount = 0;
	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++)
		TotalCount += iter->Count;

	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++)
	{
		GeometryVector temp(4);
		if (iter != HGen.bins.end() - 1)
			temp.x[0] = ((iter[0].Start) + (iter[1].Start)) / 2.0;//r 
		else
			continue;

		if (iter == HGen.bins.begin())
		{
			if (iter->Start == 0.0 && (iter[2].Start - iter[1].Start)>iter[1].Start && iter->Count == 0)
				continue;//there shouldn't be a bin at here
		}
		temp.x[2] = (temp.x[0] - iter[0].Start);//\delta r

		temp.x[1] = static_cast<double>(iter->Count) / TotalCount / (2 * temp.x[2]);//p(r)

		if (iter->Count != 0)
			temp.x[3] = temp.x[1] / std::sqrt(static_cast<double>(iter->Count));//\delta p(r)
		else
			temp.x[3] = 0.0;
		result.push_back(temp);
	}
	return TotalCount;
}
size_t HistogramToCDF(const HistogramGenerator & HGen, std::vector<GeometryVector> & result)
{
	result.clear();
	size_t TotalCount = 0;
	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++)
		TotalCount += iter->Count;

	//debug temp
	//std::cout << "TotalCount=" << TotalCount;
	size_t AlreadyCount = 0;

	for (auto iter = HGen.bins.begin(); iter != HGen.bins.end(); iter++)
	{
		GeometryVector temp(4);
		if (iter != HGen.bins.end() - 1)
			temp.x[0] = ((iter[0].Start) + (iter[1].Start)) / 2.0;//r 
		else
			continue;

		if (iter == HGen.bins.begin())
		{
			if (iter->Start == 0.0 && (iter[2].Start - iter[1].Start)>iter[1].Start && iter->Count == 0)
				continue;//there shouldn't be a bin at here
		}
		temp.x[2] = (temp.x[0] - iter[0].Start);//\delta r
		temp.x[1] = (AlreadyCount + 0.5*iter->Count) / TotalCount;//p(r)
		temp.x[3] = (0.5*iter->Count) / TotalCount;//\delta p(r)
		result.push_back(temp);
		AlreadyCount += iter->Count;
	}
	return TotalCount;
}


/** \brief Functional implementation of getS2(...). 
 *	Compute (intersection volume of two identical but displaced packings) / (system volume). */
double getS2_exact(const SpherePacking & pConfig, const GeometryVector & x) {
	GeometryVector x_relative = pConfig.CartesianCoord2RelativeCoord(x);	// displacement x in relative coordinates.
	double rc = pConfig.GetMaxRadius(),		// maximal particle radius
		S2=0.0L;							// variable to reture at the end
	
	for (size_t i = 0; i < pConfig.NumParticle(); i++) {
		// For other spheres to be overlapped with a displaced sphere i,
		// sphere centers should be separated by at most r_search.
		double r_search = rc + pConfig.GetCharacteristics(i);
		GeometryVector new_center = x_relative + pConfig.GetRelativeCoordinates(i);
		pConfig.IterateThroughNeighbors(new_center, r_search, 
			[&pConfig, &S2, &i](const GeometryVector & shift, const GeometryVector &LatticeShift, const signed long *PeriodicShift, size_t j)->void {
				//Add intersection volume of a sphere i and a sphere j (note that i can be equal to j).
			double ri = pConfig.GetCharacteristics(i), rj = pConfig.GetCharacteristics(j), r_ij = sqrt(shift.Modulus2());
				S2 += v2(pConfig.GetDimension(), ri, rj, r_ij);	
				}
		);
	}

	S2 /= pConfig.PeriodicVolume();
	return S2;
}

/** \brief Functional implementation of IsotropicAutocovariance function(...). */
void IsotropicAutocovariance(std::function<const SpherePacking(size_t i)> GetConfig, size_t NumConfigs, double Rmax,
	std::vector<GeometryVector> & Result, double dR, double sampling_density) {
	
	Result.clear();
	RandomGenerator rng(0);				// A random number generator for random unit vector.
	std::vector<GeometryVector> data;	// a temporary container to store data.
										// data[i] = (count, sum, squared sum);
	double binWidth = dR;
	for (size_t i = 0; i < NumConfigs; i++) {
		SpherePacking c = GetConfig(i);
		c.PrepareIterateThroughNeighbors(2 * c.GetMaxRadius());
		double phi = c.PackingFraction(), phi2 = phi * phi;

		if (i == 0) {
			if (binWidth < 0) {	//If dR < 0, then binWidth is 0.5* minimal sphere radii.
				binWidth = pow(c.PeriodicVolume(), 1.0 / c.GetDimension());
				for (size_t j = 0; j < c.NumParticle(); j++) {
					binWidth = std::min(binWidth, c.GetCharacteristics(j));
				}
				binWidth *= 0.5;
			}
			// Determine bins
			for (double R = 0; R < Rmax; R += binWidth) {
				Result.emplace_back(R, 0.0L, 0.0L, 0.0L);
				data.emplace_back(0.0L, 0.0L, 0.0L, 0.0L);
			}
		}

#pragma omp parallel for schedule (guided)
		for (int j = 0; j < data.size(); j++) {
			double chiV = 0;
			if (j == 0) {//r=0
				chiV = phi * (1.0 - phi);
				//atomic clauses are probably unnecessary here.
//#pragma omp atomic
				data[0].x[0]++;
//#pragma omp atomic
				data[0].x[1] += chiV;
//#pragma omp atomic
				data[0].x[2] += chiV * chiV;
			}
			else {//r > 0
				double r = binWidth * j;
				size_t num_samp = ceil(sampling_density);//ceil(sampling_density* HyperSphere_SurfaceArea(c.GetDimension(), r));	// the number of sampling directions.
				GeometryVector x;
				for (size_t idx = 0; idx < num_samp; idx++) {
					x = (r * RandomUnitVector(c.GetDimension(), rng));	//Displacement between two copies.
					chiV = getS2_exact(c, x) - phi2;
					// atomic clauses are probably unnecessary here.

//#pragma omp atomic
					data[j].x[0]++;
//#pragma	omp atomic
					data[j].x[1] += chiV;
//#pragma omp atomic
					data[j].x[2] += chiV * chiV;
				}
			}
		}
	}

	//Summarize data
	for (size_t i = 0; i < data.size(); i++) {
		double av = data[i].x[1] / data[i].x[0];
		double se = data[i].x[2] / data[i].x[0] - av * av;
		se = (se > 0 && data[i].x[0] > 1.0) ? sqrt(se / (data[i].x[0] - 1.0)) : 0.0L;

		Result[i].x[1] = av;
		Result[i].x[2] = 0.5*binWidth;
		Result[i].x[3] = se;
	}
}


/** \brief Compute the local packing fraction within an annulus of radii r+dr/2 and r-dr/2. 
 *	@param[in] packing	A sphere packing of interest
 *  @param[out] phi		A list of local packing fractions for annuli.
						phi[i]: local packing fraction for r+dr/2 and r-dr/2.
 *	@param[in] Rmin		Minimal radius.	If Rmin < dr/2, then Radius begins from dr/2.
 *  @param[in] dR		The thickness of annulus.
 *	@param[in] NumBins	The number of bins. Rmax = Rmin + dR* (NumBins-1)
 *	@param[in] center	The center of an observation window in relative coordinates. */
void GetLocalPackingFraction(const SpherePacking & packing, std::vector<double> &phi, 
	double Rmin, double dR, size_t NumBins, const GeometryVector & center) {
	//Initialize phi.
	phi.clear();
	phi.resize(NumBins, 0.0L);

	double R1 = (Rmin < 0.5*dR) ? 0.0L : Rmin-0.5*dR;	//Inner radius of the smallest annulus
	double R2 = R1 + dR * (NumBins);					//Outer radius of the largest annulus
	double rc = packing.GetMaxRadius();					//Maximal particle raidus
	DimensionType d = packing.GetDimension();
	//Do not consider the lower bound.
	if (R1 == 0.0L) {
		packing.IterateThroughNeighbors(center, R2 + rc,
			[&phi, dR, &packing, d](const GeometryVector & x_i, const GeometryVector & LatticeShift, const signed long * PereodicShift, const size_t i) ->void {
			double abs_x_i = sqrt(x_i.Modulus2());
			double r_prt = packing.GetCharacteristics(i);
			//Index of the largest annulus which this particle is intersecting with.
			size_t rU = std::min(phi.size() - 1, (size_t)floor((abs_x_i + r_prt) / dR));
			rU = std::max(rU, (size_t)0);
			//Index of the smallest annulus which this particle is intersecting with.
			size_t rL = (size_t)std::max(0.0, floor((abs_x_i - r_prt) / dR));
			rL = std::min(phi.size() - 1, rL);

			double v_int_prev = 0;
			double v_int_curr = 0;
			double r_win = 0;
			// add the contribution of particle i into shells to which the particle belongs
			for (size_t j = rL; j <= rU; j++) { 
				r_win = dR * (j + 1.0);
				v_int_curr = v2(d, r_win, r_prt, abs_x_i);
				phi[j] += (v_int_curr - v_int_prev);
				v_int_prev = v_int_curr;
			}
		});
	}
	//There is a lower bound of window radius (R1)
	else {
		packing.IterateThroughNeighbors(center, R2 + rc,
			[&phi, dR, &packing, d, R1, R2](const GeometryVector & x_i, const GeometryVector & LatticeShift, const signed long * PereodicShift, const size_t i) ->void {
			double abs_x_i = sqrt(x_i.Modulus2());
			double r_prt = packing.GetCharacteristics(i);
			//Index of the largest annulus which this particle is intersecting with.
			size_t rU = std::min(phi.size() - 1, (size_t)floor((abs_x_i + r_prt-R1) / dR));
			rU = std::max(rU, (size_t)0);
			//Index of the smallest annulus which this particle is intersecting with.
			size_t rL = (size_t)std::max(0.0, floor((abs_x_i - r_prt - R1) / dR));
			rL = std::min(phi.size() - 1, rL);

			double v_int_prev = 0;
			double v_int_curr = 0;
			double r_win = 0;
			// add the contribution of particle i into shells to which the particle belongs
			for (size_t j = rL; j <= rU; j++) {
				r_win = R1 + dR * (j + 1.0);
				v_int_curr = v2(d, r_win, r_prt, abs_x_i);
				phi[j] += (v_int_curr - v_int_prev);
				v_int_prev = v_int_curr;
			}
		});
	}

	// Divide phi[j] by volume of annuli
	for (size_t i = 0; i < phi.size(); i++) {
		double R_inner = R1 + dR * i, R_outer = R_inner + dR;
		phi[i] /= (HyperSphere_Volume(d, R_outer) - HyperSphere_Volume(d, R_inner));
	}
}

/** /brief Functional implementation of IsotropicAutocovariance_hybrid(...) function. */
void IsotropicAutocovariance_hybrid(std::function <const SpherePacking(size_t i)> GetConfig, size_t NumConfigs, double Rmax,
	std::vector<GeometryVector> & Result, double dR, const Parameters_Autocovariance & extra_parameters) {
	
	double Rmin = extra_parameters.Rmin;
	Rmin = (Rmin < 0.5*dR) ? 0.5*dR : Rmin;
	size_t num_centers = extra_parameters.TotalSamplingPts / NumConfigs,
		expected_samples;
	RandomGenerator rng = extra_parameters.rng;
	Result.clear();
	Result.emplace_back(0.0L, 0.0L, 0.0L, 0.0L);
	for (double R = Rmin; R <= Rmax; R += dR)
		Result.emplace_back(R, 0.0L, 0.0L, 0.0L);

	std::cout << "Computing Autocovariance function\n";
	progress_display pd(NumConfigs);

	for (size_t i = 0; i < NumConfigs; i++) {
		SpherePacking pC(GetConfig(i));	
		pC.PrepareIterateThroughNeighbors(Rmax + pC.GetMaxRadius());
		DimensionType d = pC.GetDimension();

		//Autocovariance at the origin.
		double phi = pC.PackingFraction(), phi2 = phi * phi;
		Result[0].x[1] += phi - phi2;
		Result[0].x[2] += (phi - phi2)*(phi - phi2);
		Result[0].x[3]++;

		//Determine centers of windows.
		expected_samples = round(num_centers / phi);
		std::vector<GeometryVector> centers;	centers.reserve(expected_samples);
		for (size_t j = 0; j < expected_samples; j++) {
			GeometryVector x(static_cast<int> (d));
			for (int k = 0; k < d; k++)
				x.x[k] = rng.RandomDouble();
			centers.push_back(x);
		}
		
		//S2 at r.
		std::vector<GeometryVector> S2(Result.size() - 1, GeometryVector(0.0L, 0.0L, 0.0L, 0.0L));
		//size_t idx = 0;
#pragma omp parallel num_threads(extra_parameters.NumThreads)
		{
			SpherePacking P(pC);
#pragma omp for schedule(guided)
			for (int j = 0; j < centers.size(); j++) {
				//Window center is in the particle phase
				if (P.CheckOverlap(centers[j], 0)) {
					std::vector<double> local_phi;
					GetLocalPackingFraction(P, local_phi, Rmin, dR, Result.size() - 1, centers[j]);

					//Incorportate data
					double Frac_at_r = 0;	//Local packing fraction in annuli
					for (int idx1 = 0; idx1 < local_phi.size(); idx1++) {
						Frac_at_r = local_phi[idx1];
#pragma omp atomic
						S2[idx1].x[1] += Frac_at_r;
#pragma omp atomic
						S2[idx1].x[2] += Frac_at_r * Frac_at_r;
#pragma omp atomic
						S2[idx1].x[3]++;
					}
				}
			}
		}

		
		//Normalize
		for (size_t j = 0; j < S2.size(); j++) {
			double chi = S2[j].x[1] * phi ;	//Sum of S_2(r). Normalized by the global packing fraction, instead of a sampled one. 
											//This normalization factor is empirically chosen...
			chi -= phi2 * S2[j].x[3];		//Sum of (S_2 (r) - phi^2)
			Result[j + 1].x[1] += chi;
			
			Result[j + 1].x[2] += phi2*(S2[j].x[2] - 2.0*phi*S2[j].x[1] + S2[j].x[3]* phi2); // Sum of (I - phi2)^2
			Result[j + 1].x[3] += S2[j].x[3];
		}
		pd++;
	}
	
	//Summarize results
	for (size_t i = 0; i < Result.size(); i++) {
		double av = Result[i].x[1] / Result[i].x[3];// / (double)NumConfigs;
		double se = Result[i].x[2] / Result[i].x[3] - av * av;//(double) (NumConfigs*num_centers) - av * av;
		if (Result[i].x[3] == 1 || se < 0) {
			se = 0.0L;
		}
		else {
			se = sqrt(se / (Result[i].x[3] - 1.0));
		}

		Result[i].x[1] = av;
		Result[i].x[2] = 0.5*dR;
		Result[i].x[3] = se;
	}
}
