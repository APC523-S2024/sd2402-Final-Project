/*
Header file for functions for computing (an)isotropic structure factor and spectral density of point patterns/packings

Sam Dawley
5/2024

References
...
*/

#include <algorithm>
#include <complex>
#include <cmath>

#include "StructureFactor.h"

/** \file StructureFactor.cpp
 *	\brief	Function implementations for structure factor. */

/** \struct A class to store data in each bin. */
struct SkBin
{
	double Sum1,	//< count
		SumK1,		//<	sum of the magnitudes of wavevectors (|k|)
		SumK2,		//< sum of squared magnitudes of wavevectors (|k|^2)
		SumS,		//< sum of spectral functions
		SumS2;		//< sum of squared spectral functions
	SkBin() : Sum1(0), SumK1(0), SumK2(0), SumS(0), SumS2(0)
	{}
};


double StructureFactor(const Configuration & Config, const GeometryVector & k)
{
	std::complex<double> rho;
	size_t NumParticle = Config.NumParticle();
	for(size_t i=0; i<NumParticle; i++)
	{
		rho+=std::exp(std::complex<double>(0, k.Dot(Config.GetCartesianCoordinates(i))));
	}
	return (rho.real()*rho.real()+rho.imag()*rho.imag())/NumParticle;
}

double WeightedStructureFactor(const PeriodicCellList<std::complex<double>> & Config, const GeometryVector & k) {
	std::complex<double> rho;
	size_t NumParticle = Config.NumParticle();
	for (size_t i = 0; i<NumParticle; i++)
	{
		rho += Config.GetCharacteristics(i)*std::exp(std::complex<double>(0, k.Dot(Config.GetCartesianCoordinates(i))));
	}
	return (rho.real()*rho.real() + rho.imag()*rho.imag()) / NumParticle;
}


std::vector<GeometryVector> GetKs(const PeriodicCellList<Empty> & tempList, double CircularKMax, double LinearKMax, double SampleProbability)
{
	//Exclude probablematic cases.
	if (CircularKMax <= 0.0) {
		std::cerr << "GetKs():: CircularKMax should be positive." << std::endl;
		return std::vector<GeometryVector>();
	}
	if (LinearKMax <= 0.0) {
		std::cerr << "GetKs():: Linear KMax should be positive." << std::endl;
		return std::vector<GeometryVector>();
	}

	RandomGenerator gen(98765);
	std::vector<GeometryVector> results;
	DimensionType dim = tempList.GetDimension();
	if (dim == 0) {
		std::cerr << "Dimension of PeriodicCellList must be positive!\n";
		exit(1);
	}
	else if (dim > 1)
	{
		std::vector<GeometryVector> bs;
		for (DimensionType i = 0; i<dim; i++)
			bs.push_back(tempList.GetReciprocalBasisVector(i));
		//a periodic list of reciprocal lattice
		PeriodicCellList<Empty> reciprocal(dim, &bs[0], std::sqrt(bs[0].Modulus2()));
		reciprocal.Insert(Empty(), GeometryVector(dim));

		std::vector<GeometryVector> ks;
		std::vector<GeometryVector> preKs;
		reciprocal.IterateThroughNeighbors(GeometryVector(dim), CircularKMax, [&ks, &dim](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom)->void{
			bool Trivial = true;
			for (DimensionType d = 0; d<dim; d++)
			{
				if (PeriodicShift[d]>0)
				{
					Trivial = false;
					break;
				}
				else if (PeriodicShift[d]<0)
				{
					Trivial = true;
					break;
				}
			}
			if (!Trivial)
				ks.push_back(shift);
		});
		std::sort(ks.begin(), ks.end(), [](const GeometryVector & left, const GeometryVector & right) ->bool {return left.Modulus2()<right.Modulus2(); });
		//ks contains K points in that circle

		if (LinearKMax == CircularKMax)
			return ks;

		std::vector<GeometryVector> tempBase;
		for (DimensionType i = 0; i<dim; i++)
		{
			GeometryVector t(dim);
			t.x[i] = 2;
			tempBase.push_back(t);
		}
		PeriodicCellList<Empty> KDirection(dim, &tempBase[0], std::sqrt(tempBase[0].Modulus2())*std::pow(1.0 / ks.size(), 1.0 / (dim))*10.0);//use this list to discover k points of the same direction
		for (auto iter = ks.begin(); iter != ks.end(); iter++)
		{
			double Length = std::sqrt(iter->Modulus2());
			if (Length == 0)
				continue;
			GeometryVector dir(dim);
			for (DimensionType i = 0; i<dim; i++)
				dir.x[i] = iter->x[i] / Length / 2.0;

			bool NeighborFound = false;
			KDirection.IterateThroughNeighbors(dir, 1e-10, [&NeighborFound](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom)->void{NeighborFound = true; });
			if (NeighborFound == false)
			{
				preKs.push_back(*iter);
				KDirection.Insert(Empty(), dir);
			}
		}
		//preKs contains K points in that circle with different directions

		ks.clear();
		for (auto iter = preKs.begin(); iter != preKs.end(); iter++)
		{
			for (size_t i = 1;; i++)
			{
				GeometryVector temp = static_cast<double>(i)*(*iter);
				if (temp.Modulus2()>LinearKMax*LinearKMax)
					break;
				else
					if(SampleProbability>=1.0 || gen.RandomDouble()<SampleProbability)
						results.push_back(temp);
			}
		}
		preKs.clear();
	}
	else
	{
		GeometryVector b = tempList.GetReciprocalBasisVector(0);
		for (double i = 1.0;; i += 1.0)
		{
			GeometryVector k = i*b;
			if (k.x[0] > LinearKMax)
				break;
			if (SampleProbability >= 1.0 || gen.RandomDouble()<SampleProbability)
				results.push_back(k);
		}
	}
	return results;
}


void IsotropicStructureFactor(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision, double SampleProbability)
{
	Results.clear();
	if(!(CircularKMax>0))
		return;
	if(!(LinearKMax>0))
		return;

	if (KPrecision == 0.0)
	{
		std::cerr << "Warning in IsotropicStructureFactor : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
		Configuration c = GetConfigsFunction(0);
		KPrecision = std::sqrt(c.GetReciprocalBasisVector(0).Modulus2());
	}
	size_t NumBin = std::floor(LinearKMax / KPrecision) + 1;
	std::vector<SkBin> vSkBin(NumBin, SkBin());


	GeometryVector prevBasis [ ::MaxDimension];
	std::vector<GeometryVector> ks;
	if(Verbosity>1)
		std::cout<<"Computing S(k)";
	progress_display pd(NumConfigs);
	for(size_t j=0; j<NumConfigs; j++)
	{
		//if(Verbosity>3 || (Verbosity>2&&j%100==0) )
		//	std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
		Configuration CurrentConfig = GetConfigsFunction(j);
		if(CurrentConfig.GetDimension()==0)
			break;
		if(j!=0)
		{
			bool SameBasis = true;
			for(DimensionType i=0; i<CurrentConfig.GetDimension(); i++)
				if( !(prevBasis[i]==CurrentConfig.GetBasisVector(i)) )
					SameBasis=false;

			if(SameBasis==false)
				ks=GetKs(CurrentConfig, CircularKMax, LinearKMax, SampleProbability);
		}
		else
			ks = GetKs(CurrentConfig, CircularKMax, LinearKMax, SampleProbability);

		signed long end = ks.size();

#pragma omp parallel for schedule(guided)
		for(signed long i=0; i<end; i++)
		{
			double s=StructureFactor(CurrentConfig, ks[i]);
			double k2 = ks[i].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / KPrecision);
#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
			//-----added-----
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
			//---------------
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s*s;
			
		}
		for(DimensionType i=0; i<CurrentConfig.GetDimension(); i++)
			prevBasis[i]=CurrentConfig.GetBasisVector(i);
		pd++;
	}
	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			//temp.x[0] = std::sqrt(iter->SumK2 / iter->Sum1);
			//temp.x[1] = iter->SumS / iter->Sum1;
			//temp.x[2] = KPrecision;
			//temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			Results.push_back(temp);
		}
	}
	
	if(Verbosity>2)
		std::cout<<"done!\n";
}

void IsotropicStructureFactor_weighted(std::function<const PeriodicCellList<std::complex<double>>(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision, double SampleProbability)
{
	Results.clear();
	if (!(CircularKMax>0))
		return;
	if (!(LinearKMax>0))
		return;

	if (KPrecision == 0.0)
	{
		std::cerr << "Warning in IsotropicStructureFactor : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
		PeriodicCellList<std::complex<double>> c = GetConfigsFunction(0);
		KPrecision = std::sqrt(c.GetReciprocalBasisVector(0).Modulus2());
	}
	size_t NumBin = std::floor(LinearKMax / KPrecision) + 1;
	std::vector<SkBin> vSkBin(NumBin, SkBin());


	GeometryVector prevBasis[::MaxDimension];
	std::vector<GeometryVector> ks;
	if (Verbosity>1)
		std::cout << "Computing S(k)";
	progress_display pd(NumConfigs);



	for (size_t j = 0; j<NumConfigs; j++)
	{
		//if(Verbosity>3 || (Verbosity>2&&j%100==0) )
		//	std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
		PeriodicCellList<std::complex<double>> CurrentConfig = GetConfigsFunction(j);
		if (CurrentConfig.GetDimension() == 0)
			break;
		if (j != 0)
		{
			bool SameBasis = true;
			for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
				if (!(prevBasis[i] == CurrentConfig.GetBasisVector(i)))
					SameBasis = false;

			if (SameBasis == false)
				ks = GetKs(CurrentConfig, CircularKMax, LinearKMax, SampleProbability);
		}
		else
			ks = GetKs(CurrentConfig, CircularKMax, LinearKMax, SampleProbability);

		signed long end = ks.size();

#pragma omp parallel for schedule(guided)
		for (signed long i = 0; i<end; i++)
		{
			double s = WeightedStructureFactor(CurrentConfig, ks[i]);
			double k2 = ks[i].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / KPrecision);
#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
			//-----added-----
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
			//---------------
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s * s;

		}
		for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
			prevBasis[i] = CurrentConfig.GetBasisVector(i);
		pd++;
	}


	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			//temp.x[0] = std::sqrt(iter->SumK2 / iter->Sum1);
			//temp.x[1] = iter->SumS / iter->Sum1;
			//temp.x[2] = KPrecision;
			//temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			Results.push_back(temp);
		}
	}

	if (Verbosity>2)
		std::cout << "done!\n";
}


#include <boost/math/special_functions.hpp>
double Mtilde(double k, double R, DimensionType d)
{
	if (d == 1)
		return 2 * std::sin(k*R) / k;
	else if (d == 2)
		return 2 * pi*R*boost::math::cyl_bessel_j(1, k*R) / k;
	else if (d == 3)
		return 4 * pi*(std::sin(k*R) - k*R*std::cos(k*R)) / k / k / k;
	else if (d == 4)
		return 4 * pi*pi*R*(2 * boost::math::cyl_bessel_j(1, k*R) - k*R*boost::math::cyl_bessel_j(0, k*R)) / k / k / k;
	else
	{
		std::cerr << "Error in Mtilde : unsupported dimension!\n";
		return 0.0;
	}
}

double SpectralDensity(const SpherePacking & Config, const GeometryVector & k)
{
	std::complex<double> rho;
	std::map<double, double> mtildes;
	size_t NumParticle = Config.NumParticle();
	double kl = std::sqrt(k.Modulus2());
	DimensionType d = Config.GetDimension();
	for (size_t i = 0; i<NumParticle; i++)
	{
		std::complex<double> a = std::exp(std::complex<double>(0, k.Dot(Config.GetCartesianCoordinates(i))));
		double mt;
		double R = Config.GetCharacteristics(i);
		auto iter = mtildes.find(R);
		if (iter != mtildes.end())
			mt = iter->second;
		else
		{
			mt = Mtilde(kl, R, d);
			mtildes.insert(std::make_pair(R, mt));
		}
		rho += a*mt;
	}
	return (rho.real()*rho.real() + rho.imag()*rho.imag()) / Config.PeriodicVolume();
}



void IsotropicSpectralDensity(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision)
{
	Results.clear();
	if (!(CircularKMax>0))
		return;
	if (!(LinearKMax>0))
		return;

	if (KPrecision == 0.0)
	{
		std::cerr << "Warning in IsotropicSpectralDensity : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
		SpherePacking c = GetConfigsFunction(0);
		KPrecision = std::sqrt(c.GetReciprocalBasisVector(0).Modulus2());
	}
	size_t NumBin = std::floor(LinearKMax / KPrecision) + 1;
	std::vector<SkBin> vSkBin(NumBin, SkBin());


	GeometryVector prevBasis[::MaxDimension];
	std::vector<GeometryVector> ks;
	if (Verbosity>1)
		std::cout << "Computing chi(k)";

	//modified part
	{	
		SpherePacking CurrentConfig = GetConfigsFunction(0);
		ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);
		if (Verbosity > 0) {
			std::cout << "the number of considered k-points = " << ks.size() << std::endl;
		}
	}

	progress_display pd(NumConfigs);
	for (size_t j = 0; j<NumConfigs; j++)
	{
		//if(Verbosity>3 || (Verbosity>2&&j%100==0) )
		//	std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
		SpherePacking CurrentConfig = GetConfigsFunction(j);
		if (CurrentConfig.GetDimension() == 0)
			continue;
		if (j != 0)
		{
			bool SameBasis = true;
			for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
				if (!(prevBasis[i] == CurrentConfig.GetBasisVector(i)))
					SameBasis = false;

			if (SameBasis == false)
				ks=GetKs(CurrentConfig, CircularKMax, LinearKMax);
		}
		//else{
		//	ks=GetKs(CurrentConfig, CircularKMax, LinearKMax);
		//}
		signed long end = ks.size();
#pragma omp parallel for schedule(guided)
		for (signed long i = 0; i<end; i++)
		{
			double s = SpectralDensity(CurrentConfig, ks[i]);
			double k2 = ks[i].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / KPrecision);
#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
			//-----added-----
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
			//---------------
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s*s;
		}
		for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
			prevBasis[i] = CurrentConfig.GetBasisVector(i);
		pd++;
	}
	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			//temp.x[0] = std::sqrt(iter->SumK2 / iter->Sum1);
			//temp.x[1] = iter->SumS / iter->Sum1;
			//temp.x[2] = KPrecision;
			//temp.x[3] = std::sqrt(iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / sqrt(iter->Sum1);
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			
			Results.push_back(temp);
		}
	}

	if (Verbosity>2)
		std::cout << "done!\n";
}

void IsotropicSpectralDensity_Lstorage(std::function<const SpherePacking(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision)
{
	Results.clear();
	if (!(CircularKMax>0))
		return;
	if (!(LinearKMax>0))
		return;

	if (KPrecision == 0.0)
	{
		std::cerr << "Warning in IsotropicSpectralDensity : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
		SpherePacking c = GetConfigsFunction(0);
		KPrecision = std::sqrt(c.GetReciprocalBasisVector(0).Modulus2());
	}
	size_t NumBin = std::floor(LinearKMax / KPrecision) + 1;
	std::vector<SkBin> vSkBin(NumBin, SkBin());


	GeometryVector prevBasis[::MaxDimension];
	std::vector<GeometryVector> ks;
	if (Verbosity>1)
		std::cout << "Computing chi(k)";

	//modified part
	{
		SpherePacking CurrentConfig = GetConfigsFunction(0);
		ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);
		if (Verbosity > 0) {
			std::cout << "the number of considered k-points = " << ks.size() << std::endl;
		}
	}

	progress_display pd(NumConfigs);
	for (size_t j = 0; j<NumConfigs; j++)
	{
		//if(Verbosity>3 || (Verbosity>2&&j%100==0) )
		//	std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
		SpherePacking CurrentConfig = GetConfigsFunction(j);
		if (CurrentConfig.GetDimension() == 0)
			continue;
		if (j != 0)
		{
			bool SameBasis = true;
			for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
				if (!(prevBasis[i] == CurrentConfig.GetBasisVector(i)))
					SameBasis = false;

			if (SameBasis == false)
				ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);
		}
		//else
		//	ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);

		signed long end = ks.size();
#pragma omp parallel for schedule(guided)
		for (signed long i = 0; i<end; i++)
		{
			double s = SpectralDensity_Lstorage(CurrentConfig, ks[i]);
			double k2 = ks[i].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / KPrecision);
#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
			//-----added-----
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
			//---------------
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s * s;
		}
		for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
			prevBasis[i] = CurrentConfig.GetBasisVector(i);
		pd++;
	}
	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it

			Results.push_back(temp);
		}
	}

	if (Verbosity>2)
		std::cout << "done!\n";
}


std::vector<GeometryVector> DirectionalStructureFactor(std::function<const Configuration(size_t i)> GetConfigsFunction, size_t numConfig, long limit, std::vector<GeometryVector> & Results, int dk , double SampleProbability ) {
	
	std::vector<GeometryVector> Grid;
	Results.clear();
	if (numConfig == 0) {
		std::cout << "DirectionalStructrureFactor:: no configurations \n";
		return Grid;
	}
	else {
		Configuration initConfig = GetConfigsFunction(0);
		DimensionType d = initConfig.GetDimension();
		//Generate Reciprocal Vectors
		std::vector<GeometryVector> r_b;
		for (int i = 0; i < d; i++)
			r_b.push_back(initConfig.GetReciprocalBasisVector(i));

		////check whether all supercells are the same or not
		//bool isSameCells = true;
		//for (int i = 0; i < numConfig; i++) {
		//	for (int j = 0; j < d; j++) {
		//		isSameCells = isSameCells || (r_b[j] == config[i].GetReciprocalBasisVector(j));
		//	}
		//}

		//if (!isSameCells) {
		//	std::cout << "Wrong set of configurations \n";
		//	return Grid;
		//}
		//
			if (dk == 1.0) {

			}
			else {
				std::cout << "dk>1 is not supported yet \n";
			}

			//Generate a grid of vectors
			int index_origin = 0;
			GeometryVector indices(static_cast<int> (d));
			for (int i = 0; i < d; i++) {
				indices.x[i] = -limit;
			}
			Grid.reserve((int)pow(2 * limit + 1, d));
			while (indices.x[d-1] <= (double)limit) {
				GeometryVector x(static_cast<int> (d));
				for (int i = 0; i < d; i++)
					x = x + (double)indices.x[i] * r_b[i];
				
				Grid.push_back(GeometryVector(x));
				indices.x[0]++;
				for (int i = 0; i < d - 1; i++) {
					if (indices.x[i] > limit) {
						indices.x[i] = -limit;
						indices.x[i + 1] ++;
					}
				}
				if (index_origin == 0 && indices.Modulus2() == 0) {
					index_origin = Grid.size();
				}
			}
			
			Results.resize(Grid.size());
			// sum of structure factors;	sum of square of structure factor;	count;	
			
			for (int i = 0; i < numConfig; i++) {
				Configuration temp;
				if (i == 0)
					temp = Configuration(initConfig);
				else
					temp = Configuration(GetConfigsFunction(i));
#pragma omp parallel for schedule(guided,10)
				for (int j = 0; j < Grid.size(); j++) {
					double s = 0;
					if (j != index_origin) {
						s = StructureFactor(temp, Grid[j]);
					}
						Results[j].x[0] += s;
						if (numConfig > 1) {
							Results[j].x[1] += s*s;
						}
						Results[j].x[2] ++;
					
				}
			}

			//Data in Results are transformed into 
			//Structure factor;	standard error in Structure factor;
			for (int i = 0; i < Results.size(); i++) {
				Results[i].x[0] /= Results[i].x[2];
				if (numConfig > 1) {
					Results[i].x[1] = std::sqrt((Results[i].x[1] / Results[i].x[2] - Results[i].x[0] * Results[i].x[0]) / (Results[i].x[2] - 1.0));
				}
				Results[i].x[2] = 0;
			}

			return Grid;
		
			
	}
	

}


std::vector<GeometryVector> DirectionalSpectralDensity(std::function<const SpherePacking(size_t i)> config, size_t numConfig, long limit, std::vector<GeometryVector> & Results, double dk , double SampleProbability ) {
	std::vector<GeometryVector> Grid;
	Results.clear();
	if (numConfig == 0) {
		std::cout << "DirectionalStructrureFactor:: no configurations \n";
		return Grid;
	}
	else {
		DimensionType d = config(0).GetDimension();
		//Generate Reciprocal Vectors
		std::vector<GeometryVector> r_b;
		for (int i = 0; i < d; i++)
			r_b.push_back(config(0).GetReciprocalBasisVector(i));

		//check whether all supercells are the same or not
		bool isSameCells = true;
		for (int i = 0; i < numConfig; i++) {
			for (int j = 0; j < d; j++) {
				isSameCells = isSameCells || (r_b[j] == config(0).GetReciprocalBasisVector(j));
			}
		}

		if (!isSameCells) {
			std::cout << "Wrong set of configurations \n";
			return Grid;
		}
		else {
			if (dk == 1.0) {

			}
			else {
				std::cout << "dk>1 is not supported yet \n";
			}

			//Generate a grid of vectors
			int index_origin = 0;
			GeometryVector indices(static_cast<int> (d));
			for (int i = 0; i < d; i++) {
				indices.x[i] = -limit;
			}
			Grid.reserve((int)pow(2 * limit + 1, d));
			while (indices.x[d - 1] <= (double)limit) {
				GeometryVector x(static_cast<int> (d));
				for (int i = 0; i < d; i++)
					x = x + (double)indices.x[i] * r_b[i];

				Grid.push_back(GeometryVector(x));
				indices.x[0]++;
				for (int i = 0; i < d - 1; i++) {
					if (indices.x[i] > limit) {
						indices.x[i] = -limit;
						indices.x[i + 1] ++;
					}
				}
				if (index_origin == 0 && indices.Modulus2() == 0) {
					index_origin = Grid.size();
				}
			}

			Results.resize(Grid.size());
			// sum of structure factors;	sum of square of structure factor;	count;	

			for (int i = 0; i < numConfig; i++) {
#pragma omp parallel for schedule(guided,10)
				for (int j = 0; j < Grid.size(); j++) {
					double s = 0;
					if (j != index_origin) {
						s = SpectralDensity(config(0), Grid[j]);
					}
					Results[j].x[0] += s;
					if (numConfig > 1) {
						Results[j].x[1] += s*s;
					}
					Results[j].x[2] ++;

				}
			}

			//Data in Results are transformed into 
			//spectral density;	standard error in Structure factor;
			for (int i = 0; i < Results.size(); i++) {
				Results[i].x[0] /= Results[i].x[2];
				if (numConfig > 1) {
					Results[i].x[1] = std::sqrt((Results[i].x[1] / Results[i].x[2] - Results[i].x[0] * Results[i].x[0]) / (Results[i].x[2] - 1.0));
				}
				Results[i].x[2] = 0;
			}

			return Grid;
		}

	}

}


#if defined USE_SHAPES
	
/** \brief Function implementations to compute spectral density for nonsphere packings. */

/** Compute a spectral density for a given configuration. */
double SpectralDensity_nonspherical(const Dispersion & config, const GeometryVector &k) {
	//Check Dimensions
	if (config.GetDimension() == k.Dimension) {
		std::complex <double> tildeI(0.0, 0.0);

		for (int i = 0; i < config.NumParticle(); i++) {
			tildeI += config.GetShape(i)->GetFormFactor(k) * std::exp(std::complex<double>(0.0, +k.Dot(config.GetCartesianCoordinates(i))));
		}

		// |tildeI|^2 / V_F
		return std::norm(tildeI) / config.PeriodicVolume();	
	}
	else {
		std::cerr << "SpectralDensity_nonspherical() have incorrect arguements." << std::endl;
	}
}

/** Compute values of the spectral density. */
void IsotropicSpectralDensity_nonspherical(std::function<const Dispersion(size_t)> GetConfiguration, size_t NumConfigs,
	double CircularKmax, double LinearKmax, std::vector<GeometryVector> & Result, double KPrecision) {
	
	std::vector<SkBin> vSkBin;		//	data in each bin
	std::vector<GeometryVector> ks;	//	Wavevectors
	GeometryVector prevBasis[::MaxDimension];	//	An array of basis vectors of the fundamental cell.
	double dK = KPrecision, SampleProbability = 1.0;
	DimensionType d = 0;
	progress_display pd(NumConfigs);

	for (int i = 0; i < NumConfigs; i++) {
		//Get a Dispersion object
		const Dispersion CurrConfig = GetConfiguration(i);
		
		//For the first configuration,
		//obtain wavevectors.
		//If dK = 0, assign a new value.
		if (i == 0) {
			if (Verbosity > 1)
				std::cout << "Computing S(k)";

			d = CurrConfig.GetDimension();
			ks = GetKs(CurrConfig, CircularKmax, LinearKmax, SampleProbability);	

			if (dK == 0) {
				std::cerr << "Warning in IsotropicStructureFactor : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
				KPrecision = std::sqrt(CurrConfig.GetReciprocalBasisVector(0).Modulus2());
			}
		}
		else {
			//GetKs() function can be very time-consuming.
			//So save time, we recycle wavevectors used in the previous configuration, 
			//if the current one has the same basis vectors.
			bool SameBasis = true;
			if (d == CurrConfig.GetDimension()) {
				for (size_t idx = 0; idx < d && SameBasis; idx++)
					SameBasis = prevBasis[idx] == CurrConfig.GetBasisVector(idx);
			}
			else {
				std::cerr << "Dimensions of configurations are inconsistent." << std::endl;
				return; 
			}
			
			//if the basis vectors are different, obtain a new set of basis vectors.
			if (SameBasis == false)
				ks = GetKs(CurrConfig, CircularKmax, LinearKmax, SampleProbability);
		}

		//Parallelize computations with respect to wavevectors.
		signed long end = ks.size();
#pragma omp parallel for schedule(guided)
		for (signed long j = 0; j < end; j++) {

			double s = SpectralDensity_nonspherical(CurrConfig, ks[j]);
			double k2 = ks[j].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / dK);

#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s * s;
		}
		//save the basis vectors of the current configuration.
		for (size_t idx = 0; idx < d; idx++)
			prevBasis[idx] = CurrConfig.GetBasisVector(idx);
		pd++;
	}

	//Summarize data in vSkBin
	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it
			Result.push_back(temp);
		}
	}

	if (Verbosity>2)
		std::cout << "done!\n";
}

/** Compute spectral density at each wavevector. 
 * Assume all configurations have identical basis vectors. 
 * Assume dk = 1. */
void DirectionalSpectralDensity_nonspherical(std::function<const Dispersion(size_t)> GetConfigsFunction, size_t NumConfigs,
	long limit, std::vector<GeometryVector> & ks, std::vector<GeometryVector> & chik, double dk) {

	ks.clear();
	chik.clear();

	GeometryVector prevBasis[::MaxDimension];
	DimensionType dim;
	std::vector<SkBin> Bins;

	if (NumConfigs == 0) {
		std::cerr << "DirectionalSpectralDensity_nonspherical:: no configurations \n";
		return;
	}
	else {

		for (size_t i = 0; i < NumConfigs; i++) {
			const Dispersion CurrConfig = GetConfigsFunction(i);

			//Check wavevectors;
			//For the first configuration, obtain ks.
			if (i == 0) {
				dim = CurrConfig.GetDimension();
				if (dim == 0) {
					std::cerr << "Realizations should have positive dimensions.\n";
					return;
				}

				for (size_t j = 0; j < dim; j++)
					prevBasis[j] = CurrConfig.GetReciprocalBasisVector(j);

				//Obtain nontrivial wavevectors (parity pairs are excluded).
				{
					int num = pow(2 * limit + 1, dim);
					n_naryNumber rel_coord(dim, 2 * limit + 1);

					for (int idx = 0; idx < num; idx++) {
						GeometryVector k;	k.SetDimension(dim);
						bool nontrivial = true;
						int component;

						for (int j = 0; j < dim && nontrivial; j++) {
							component = rel_coord.GetNthDigit(j) - limit;
							if (component > 0)
								nontrivial = true;
							else if (component < 0)
								nontrivial = false;

							k = k + component * prevBasis[j];
						}
						if (nontrivial)
							ks.push_back(k);

						rel_coord++;
					}
				}
				//data
				std::vector<SkBin> Bins(ks.size(), SkBin());
			}
			else {
				//Check whether this system has identical basis vectors as the previous ones.

				if (dim != CurrConfig.GetDimension()) {
					std::cerr << "Realizations should have identical space dimensions.\n";
					return;
				}

				bool sameBasis = true;
				for (int j = 0; j < dim; j++)
					sameBasis &= (prevBasis[j] == CurrConfig.GetReciprocalBasisVector(j));

				if (!sameBasis) {
					std::cerr << "Realizations should have the same basis vectors\n";
					return;
				}

			}

			if (dk == 1.0) {

				int end = ks.size();
#pragma omp parallel for schedule (guided)
				for (int n = 0; n < end; n++) {
					double s = SpectralDensity_nonspherical(CurrConfig, ks[n]);

#pragma omp atomic
					Bins[n].Sum1++;
#pragma omp atomic
					Bins[n].SumS += s;
#pragma omp atomic
					Bins[n].SumS2 += s * s;
				}
			}
			else {
				std::cerr << "It isn't implemented\n";
				return;
			}



		}

		//Summarize data
		chik.resize(Bins.size(), GeometryVector());
		for (size_t i = 0; i < Bins.size(); i++) {
			chik[i].x[0] = Bins[i].SumS / Bins[i].Sum1;
			chik[i].x[1] = Bins[i].SumS2 / Bins[i].Sum1 - chik[i].x[0] * chik[i].x[0];
			chik[i].x[1] = std::sqrt(std::abs(chik[i].x[1]) / Bins[i].Sum1);
		}
	}
}





#endif


/*
#if defined(USE_SHAPES)
double MtildeCube(const GeometryVector & k, const Cube & cube, DimensionType d) {
	double result = 1.0;

	for (DimensionType i = 0; i < d; i++) {
		double kL = k.Dot(cube.GetBasis(i));
		//for (DimensionType j = 0; j < d; j++)
		//	kL += k.Dot(cube.GetBasis(j));

		kL = 0.5 * fabs(kL);
		double temp = 0.0;

		if (kL < 1e-16)
			temp = 1.0;
		else
			temp = sin(kL) / kL;

		result *= temp;
	}
	return cube.GetVolume()*result;
}
*//*
double SpectralDensityCube(const Dispersion_cube & Config, const GeometryVector & k) {
	std::complex<double> rho;

	size_t NumParticle = Config.NumParticle();
	double kl = std::sqrt(k.Modulus2());
	DimensionType d = Config.GetDimension();
	for (size_t i = 0; i<NumParticle; i++)
	{
		std::complex<double> a = std::exp(std::complex<double>(0, k.Dot(Config.GetCartesianCoordinates(i))));
		double mt;
		Cube prt = Config.GetCharacteristics(i);

		mt = MtildeCube(k, prt, d);
		rho += a * mt;
	}
	return (rho.real()*rho.real() + rho.imag()*rho.imag()) / Config.PeriodicVolume();
}
*//*
void IsotropicSpectralDensityCube(std::function<const Dispersion_cube(size_t i)> GetConfigsFunction, size_t NumConfigs, double CircularKMax, double LinearKMax, std::vector<GeometryVector> & Results, double KPrecision) {
	Results.clear();
	if (!(CircularKMax>0))
		return;
	if (!(LinearKMax>0))
		return;
*/
	/*struct SkBin
	{
	double Sum1, SumK2, SumS, SumS2;
	SkBin() : Sum1(0), SumK2(0), SumS(0), SumS2(0)
	{}
	};*/
/*
	struct SkBin
	{
		double Sum1, SumK1, SumK2, SumS, SumS2;
		SkBin() : Sum1(0), SumK1(0), SumK2(0), SumS(0), SumS2(0)
		{}
	};
	if (KPrecision == 0.0)
	{
		std::cerr << "Warning in IsotropicSpectralDensity : This version does not support KPrecision==0.0. Auto choosing this quantity!\n";
		Dispersion_cube c = GetConfigsFunction(0);
		KPrecision = std::sqrt(c.GetReciprocalBasisVector(0).Modulus2());
	}
	size_t NumBin = std::floor(LinearKMax / KPrecision) + 1;
	std::vector<SkBin> vSkBin(NumBin, SkBin());


	GeometryVector prevBasis[::MaxDimension];
	std::vector<GeometryVector> ks;
	if (Verbosity>1)
		std::cout << "Computing chi(k)";

	//modified part
	{
		Dispersion_cube CurrentConfig = GetConfigsFunction(0);
		ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);
		if (Verbosity > 0) {
			std::cout << "the number of considered k-points = " << ks.size() << std::endl;
		}
	}

	progress_display pd(NumConfigs);
	for (size_t j = 0; j<NumConfigs; j++)
	{
		//if(Verbosity>3 || (Verbosity>2&&j%100==0) )
		//	std::cout<<j<<"/"<<NumConfigs<<"configurations processed\n";
		Dispersion_cube CurrentConfig = GetConfigsFunction(j);
		if (CurrentConfig.GetDimension() == 0)
			continue;
		if (j != 0)
		{
			bool SameBasis = true;
			for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
				if (!(prevBasis[i] == CurrentConfig.GetBasisVector(i)))
					SameBasis = false;

			if (SameBasis == false)
				ks = GetKs(CurrentConfig, CircularKMax, LinearKMax);
		}
		//else{
		//	ks=GetKs(CurrentConfig, CircularKMax, LinearKMax);
		//}
		signed long end = ks.size();
#pragma omp parallel for schedule(guided)
		for (signed long i = 0; i<end; i++)
		{
			double s = SpectralDensityCube(CurrentConfig, ks[i]);
			double k2 = ks[i].Modulus2();
			size_t Bin = std::floor(std::sqrt(k2) / KPrecision);
#pragma omp atomic
			vSkBin[Bin].Sum1 += 1.0;
			//-----added-----
#pragma omp atomic
			vSkBin[Bin].SumK1 += sqrt(k2);
			//---------------
#pragma omp atomic
			vSkBin[Bin].SumK2 += k2;
#pragma omp atomic
			vSkBin[Bin].SumS += s;
#pragma omp atomic
			vSkBin[Bin].SumS2 += s * s;
		}
		for (DimensionType i = 0; i<CurrentConfig.GetDimension(); i++)
			prevBasis[i] = CurrentConfig.GetBasisVector(i);
		pd++;
	}
	for (auto iter = vSkBin.begin(); iter != vSkBin.end(); iter++)
	{
		if (iter->Sum1 != 0.0)
		{
			GeometryVector temp(4);
			//temp.x[0] = std::sqrt(iter->SumK2 / iter->Sum1);
			//temp.x[1] = iter->SumS / iter->Sum1;
			//temp.x[2] = KPrecision;
			//temp.x[3] = std::sqrt(iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / sqrt(iter->Sum1);
			temp.x[0] = iter->SumK1 / iter->Sum1;
			temp.x[1] = iter->SumS / iter->Sum1;
			double var = (iter->SumK2 / (iter->Sum1) - temp.x[0] * temp.x[0]) / (iter->Sum1);
			temp.x[2] = (var > 0) ? std::sqrt(var) : 0;
			temp.x[3] = std::sqrt((iter->SumS2 / (iter->Sum1) - temp.x[1] * temp.x[1]) / (iter->Sum1)); // I modified it

			Results.push_back(temp);
		}
	}

	if (Verbosity>2)
		std::cout << "done!\n";
}

std::vector<GeometryVector> DirectionalSpectralDensityCube(std::function<const Dispersion_cube(size_t i)> GetConfigsFunction, size_t NumConfigs, long limit, std::vector<GeometryVector> & Results, double dk, double SampleProbability) {
	std::vector<GeometryVector> Grid;
	Results.clear();
	if (NumConfigs == 0) {
		std::cout << "DirectionalStructrureFactor:: no configurations \n";
		return Grid;
	}
	else {
		Dispersion_cube config = GetConfigsFunction(0);
		DimensionType d = config.GetDimension();
		//Generate Reciprocal Vectors
		std::vector<GeometryVector> r_b;
		for (int i = 0; i < d; i++)
			r_b.push_back(config.GetReciprocalBasisVector(i));

		//check whether all supercells are the same or not
		bool isSameCells = true;
		for (int i = 0; i < NumConfigs; i++) {
			for (int j = 0; j < d; j++) {
				isSameCells = isSameCells || (r_b[j] == config.GetReciprocalBasisVector(j));
			}
		}

		if (!isSameCells) {
			std::cerr << "Wrong set of configurations \n";
			return Grid;
		}
		else {
			if (dk == 1.0) {

			}
			else {
				std::cerr << "dk>1 is not supported yet \n";
			}

			//Generate a grid of vectors
			int index_origin = 0;
			GeometryVector indices(static_cast<int> (d));
			for (int i = 0; i < d; i++) {
				indices.x[i] = -limit;
			}
			Grid.reserve((int)pow(2 * limit + 1, d));
			while (indices.x[d - 1] <= (double)limit) {
				GeometryVector x(static_cast<int> (d));
				for (int i = 0; i < d; i++)
					x = x + (double)indices.x[i] * r_b[i];

				Grid.push_back(GeometryVector(x));
				indices.x[0]++;
				for (int i = 0; i < d - 1; i++) {
					if (indices.x[i] > limit) {
						indices.x[i] = -limit;
						indices.x[i + 1] ++;
					}
				}
				if (index_origin == 0 && indices.Modulus2() == 0) {
					index_origin = Grid.size();
				}
			}

			Results.resize(Grid.size());
			// sum of structure factors;	sum of square of structure factor;	count;	

			for (int i = 0; i < NumConfigs; i++) {
				if (i != 0)
					config = GetConfigsFunction(i);

#pragma omp parallel for schedule(guided)
				for (int j = 0; j < Grid.size(); j++) {
					double s = 0;
					if (j != index_origin) {
						s = SpectralDensityCube(config, Grid[j]);
					}
#pragma omp atomic
					Results[j].x[0] += s;
					if (NumConfigs > 1) {
#pragma omp atomic
						Results[j].x[1] += s * s;
					}
#pragma omp atomic
					Results[j].x[2] ++;

				}
			}

			//Data in Results are transformed into 
			//spectral density;	standard error in Structure factor;
			for (int i = 0; i < Results.size(); i++) {
				Results[i].x[0] /= Results[i].x[2];
				if (NumConfigs > 1) {
					Results[i].x[1] = std::sqrt((Results[i].x[1] / Results[i].x[2] - Results[i].x[0] * Results[i].x[0]) / (Results[i].x[2] - 1.0));
				}
				Results[i].x[2] = 0;
			}

			return Grid;
		}

	}
}




#endif // !USE_SHAPES
*/