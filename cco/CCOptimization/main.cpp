/*
Script for generating stealthy ground state point patterns from collective coordinate optimization

Sam Dawley
5/2024

References
...
*/

#include <fstream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "hdf5.h"
#include "../CollectiveCoordinatePotential.h"
#include "../etc.h"
#include "../GeometryVector.h"
#include "../MD_System.h"
#include "../PairCorrelation.h"
#include "../PeriodicCellList.h"
#include "../RandomGenerator.h"
#include "RepulsiveCCPotential.h"
#include "../StructureFactor.h"
#include "../StructureOptimization.h"

/*
Perform energy minimization multiple times per optimization step
Obtain ground-state configurations from multi initial configurations.
	pmetrics = refernce to vector for storing proximity metric of each run
	lmetrics = reference to vector for storing mean nearest-neighbor distance for each run
	pConfig	= pointer of a point configuration
	pPotential = pointer of a Potential object
	gen	= random number generator
	Prefix = prefix of output files
It generates three distinct ConfigurationPacks
	*_InitConfig	Initial configurations
	*				Configurations after minimizing energy
	*_Success		Configurations whose energy is lower than 1e-11 (a subset of *)
	SampleNumber	The number of initial configurations
	TimeLimit		When running time becomes larger than this, this function automatically ends.
	Algorithm	Name of alogrithm for optimizations. The followings are allowed; LBFGS, LocalGradientDescent, ConjugateGradient, SteepestDescent, MINOP
	return code = 0 if run terminates successfully
*/
int CollectiveCoordinateMultiRun(Configuration * pConfig, Potential * pPotential, RandomGenerator & gen, std::string Prefix, size_t SampleNumber, time_t TimeLimit, const std::string & AlgorithmName, double E_max, size_t max_num);

/* 
Perform a molecular dynamic simulation under NVT environment for a given potential energy
	pConfig = A pointer of a point configuration
	pPotential = A pointer of a Potential object
	gen = A random number generator
	TimeStep = Initial time step for the simulation
	Temperature = Temperature of the system
	Prefix = A prefix of output files
	SampleNumber = The number of target configurations
	StepPerSample = The number of time steps per each configuration
	AllowRestore = True if you want to load the previous setup (saved as *.MDDump)
	TimeLimit = When running time becomes larger than this, this function automatically ends
	EquilibrateSamples = The number of sampling for equilibration
	MDAutoTimeStep = True if you want to adjust time step during the equilibration
	return code = 0 if run terminates succesfully
*/
int CollectiveCoordinateMD(Configuration * pConfig, Potential * pPotential, RandomGenerator & gen, double TimeStep, double Temperature, std::string Prefix, size_t SampleNumber, size_t StepPerSample, bool AllowRestore, time_t TimeLimit, size_t EquilibrateSamples, bool MDAutoTimeStep);

class ReadConfigPack {
public:
	double Rescale;
	ConfigurationPack p;

	ReadConfigPack(std::istream & ifile, std::ostream & ofile, double Rescale)
	{
		this->Rescale = Rescale;
		ofile << "Input Prefix:";
		std::string prefix;
		ifile >> prefix;
		p.Open(prefix);
	}

	ReadConfigPack(std::string prefix, double Rescale)
	{
		this->Rescale = Rescale;
		p.Open(prefix);
	}

	Configuration operator() (size_t i)
	{
		Configuration result = p.GetConfig(i);
		result.Rescale(Rescale);
		return result;
	}
};

// #include <nlopt.h>

// compute ground states of stealthy hyperuniform potentials + soft-core repulsions.
int GetCCO(int argc, char ** argv) {
	char tempstring[1000] = {};
	char tempstring2[300] = {};
	std::istream & ifile = std::cin;
	std::ostream & ofile = std::cout;

	size_t timelimit_in_hour = 144;
	double tolerance = 1e-14;
	size_t max_steps = 10000;
	int seed = 0;

	// extract arguments passed directly at command line
	if (argc > 1) {
		timelimit_in_hour = (time_t)std::atoi(argv[1]);
		if (argc > 2) {
			tolerance = std::stof(argv[2]);
		}
		if (argc > 3) {
			max_steps = std::atoi(argv[3]);
		}
		if (argc > 4) {
			seed = std::atoi(argv[4]);
		}
	}
	RandomGenerator rngGod(seed), rng(0);

	// ofile, ifile are merely references to standard output, input, respectively
	ofile << "Time limit for simulations is " << timelimit_in_hour << " hour(s)\n";
	ofile << "Energy tolerance is           " << tolerance << std::endl;
	ofile << "Max. steps of evaluations is  " << max_steps << std::endl;
	ofile << "Random seed is                " << seed << std::endl;

	// nlopt_srand(999); // deterministically set seed
	auto start = std::time(nullptr);
	ProgramStart = start;
	TimeLimit = ProgramStart + timelimit_in_hour*3600 - 5*60; // default time limit of 144 hours - 5 mins (for saving progress)
	
	{ /* a piece of code to compute the stealthy (hyperuniform) packing at unit number density ... */
		double K1, K2, val, chi = 0, L = 10, sigma, phi = 0.1, S0=0.0;
		std::string opti_algorithm("LBFGS"); // default to fastest algorithm

		DimensionType dim = 2;
		size_t num = L*L*L, numConfig = 300, num_in_load = 0, numEquilSamples = 500, num_in_save = 0;
		int num_threads = 4;

		// HERE WE PARSE THE LONG STRING PIPED INTO EXECUTABLE 'soft_core_stealthy2.out'
		ofile << std::endl;
		ofile << "########################################" << std::endl;
		ofile << "######      Input Parameters      ######" << std::endl;
		ifile >> dim;             ofile << "Dimension = " << dim << std::endl;   // d
		ifile >> K1;              ofile << "K1        = " << K1 << std::endl;	 // K1a
		ifile >> K2;              ofile << "K2        = " << K2 << std::endl;	 // Ka
		ifile >> S0;              ofile << "S0        = " << S0 << std::endl;	 // S0
		ifile >> val;             ofile << "val       = " << val << std::endl;	 // val
		ifile >> sigma;           ofile << "sigma     = " << sigma << std::endl; // sigma
		ifile >> phi;             ofile << "phi       = " << phi << std::endl;   // phi_fake

		L = pow(num, 1.0/(double)dim);	
		double a = pow(phi/(HyperSphere_Volume(dim, 1.0)), 1.0/(double)dim);
		
		ofile << "a         = " << a << std::endl; // calculated particle radius
		ofile << std::endl;                       

		ifile >> num_threads;     ofile << "# threads              = " << num_threads << std::endl;     // num_threads
		ifile >> num;             ofile << "num particle           = " << num << std::endl;             // N
		ifile >> numConfig;       ofile << "num configs            = " << numConfig << std::endl;       // Nc
		ifile >> numEquilSamples; ofile << "num equil. sample      = " << numEquilSamples << std::endl; // Nc_equi
		ifile >> tempstring;      ofile << "Initial conditions     = " << tempstring << std::endl;      // input
		ifile >> opti_algorithm;  ofile << "Optimization algorithm = " << opti_algorithm << std::endl;  // optimization algorithm
		ofile << "######                            ######" << std::endl;
		ofile << "########################################" << std::endl;
		ofile << std::endl;
		
		char mode[100] = {};
		std::string loadname, savename;		
		std::function<const Configuration(size_t i)> GetInitConfigs = nullptr;
		
		if (strcmp (tempstring, "random") == 0) {
			GetInitConfigs = [&rngGod, &dim, &num, &L](size_t i) ->Configuration {
				Configuration pConfig = GetUnitCubicBox(dim, 0.1);
				pConfig.Rescale(L);
				for (size_t i = 0; i < num; i++)
					pConfig.Insert("a", rngGod);
				return pConfig;
			};
		} else if (strcmp (tempstring, "input") == 0) {
			ifile >> loadname; 
			ReadConfigPack c(loadname, 1.0);
			GetInitConfigs = c;
			num_in_load = c.p.NumConfig();

			ofile << "Load the .ConfigPack " << loadname << " containing " << num_in_load << " realizations" << std::endl;

		} else {
			ofile << tempstring << " is undefined \n";
			return 1;
		}
		
		ifile >> savename;
		ofile << "Save as " << savename << ".ConfigPack" << std::endl; 
		ifile >> mode; ofile << "mode (MD / ground) = " << mode << std::endl; 

		// Define potential
		RepulsiveCCPotential * potential = nullptr;
		{
			double k1 = K1 / a, k2 = K2 / a; // COMPUTE DIMENSIONLESS EXCLUSION REGION
			ofile << "Dimensionless exclusion region = the interval [" << k1 << ", " << k2 << "]" << std::endl;

			Configuration Config = GetInitConfigs(0);

			ofile << "Defining Potential..." << std::endl;
			potential = new RepulsiveCCPotential(dim, val, sigma, S0);

			ofile << "\tWith RepulsiveCCPotential, val = " << val << " and sigma = " << sigma << " (above)" << std::endl;

			potential->ParallelNumThread = num_threads;
			potential->CCPotential->ParallelNumThread = num_threads;

			// COMPTE chi BY ADDING CONSTRAINTS TO WAVE VECTORS
			// hyperuniform case
			if (S0 == 0.0) {
				std::vector<double> vals;
				vals.emplace_back(1.0);
				double K1_modulus = k1 * k1;
				std::vector<GeometryVector> ks_temp = GetKs(Config, k2, k2, 1);
				
				for (auto k = ks_temp.begin(); k != ks_temp.end(); k++) {
					if (k->Modulus2() > K1_modulus) {
						potential->CCPotential->AddConstraint(*k, vals);
						chi++;
					}
				}
				chi /= dim * (num - 1);

			// Otherwise
			} else {
				ofile << "\tequiluminous patterns with S0 = " << S0 << std::endl;
				std::vector<double> vals;
				vals.emplace_back(1.0);
				vals.emplace_back(S0);
				double K1_modulus = k1 * k1;
				std::vector<GeometryVector> ks_temp = GetKs(Config, k2, k2, 1);
				
				for (auto k = ks_temp.begin(); k != ks_temp.end(); k++) {
					if (k->Modulus2() > K1_modulus) {
						potential->CCPotential->AddConstraint(*k, vals);
						chi++;
					}					
				}
				chi /= dim * (num - 1);				
			}
			//potential->CCPotential->PrintConstraints(ofile);
			ofile << "\tSTEALTHINESS PARAMETER, chi = " << chi << std::endl;			
		}
		ofile << std::endl;

		if (strcmp (mode, "MD") == 0) {
			// {
			// 	std::cout << "Performing molecular dynamics simulation..." << std::endl;
			// 	ConfigurationPack save_(savename);				
			// 	num_in_save = save_.NumConfig();
			// }

			// /* NVT MD simulations */
			// double MDTimeStep = 0.01;
			// double MDTemperature = 1e-4;
			// size_t MDStepPerSample = 100000; 

			// if (dim == 1) {
			// 	MDTemperature = 2e-4;

			// } else if (dim == 2) {
			// 	MDTemperature = 2e-6;

			// } else if (dim == 3) {
			// 	MDTemperature = 1e-6;
			// }

			// if (strcmp (tempstring, "random") == 0){
			// 	// Start from a random initial condition and basic properties
			// 	Configuration pConfig = GetInitConfigs(0);
			// 	bool MDAllowRestore = true;
			// 	size_t MDEquilibrateSamples = numEquilSamples;
			// 	bool MDAutoTimeStep = false;
			// 	CollectiveCoordinateMD(&pConfig, potential, rngGod, MDTimeStep, MDTemperature, savename, numConfig, MDStepPerSample, MDAllowRestore, TimeLimit, MDEquilibrateSamples, MDAutoTimeStep);
			// } else if (strcmp (tempstring, "input") == 0) {
			// 	// For continue from the latest configuration in the previous run
			// 	ofile << "MD Temperature = " << MDTemperature << std::endl;
			
			// 	Configuration pConfig = GetInitConfigs(num_in_load - 1);
			// 	bool MDAllowRestore = true;
			// 	size_t MDEquilibrateSamples = numEquilSamples;
			// 	size_t numConfig_comp = (numConfig > num_in_save)? numConfig - num_in_save : 0 ;
				
			// 	bool MDAutoTimeStep = false;
			// 	CollectiveCoordinateMD(&pConfig, potential, rngGod, MDTimeStep, MDTemperature, savename, numConfig_comp, MDStepPerSample, MDAllowRestore, TimeLimit, MDEquilibrateSamples, MDAutoTimeStep);
			// //TODO
			// }
			
		} else if (strcmp (mode, "ground") == 0) {
			std::cout << "Optimizing toward ground state..." << std::endl;

			// Quench to zero temperature
			if (strcmp (tempstring, "random") == 0) {
				// random initial conditions  => use MultiRun
				Configuration pConfig = GetInitConfigs(0);
				pConfig.PrepareIterateThroughNeighbors(sigma);

				{
					ConfigurationPack save_(savename+"_Success");				
					num_in_save = save_.NumConfig();
				}

				size_t numConfig_comp = (numConfig > num_in_save) ? numConfig - num_in_save : 0;

				ofile << "We already have " << num_in_save << " Configurations in the savefile \n";
				ofile << "just compute "<< numConfig_comp <<" more Configurations \n";
				ofile << "max_eval = " << max_steps << std::endl;

				// CALL OPTIMIZATION HERE
				// CollectiveCoordinateMultiRun(&pConfig, potential, rngGod, savename, numConfig_comp, TimeLimit, opti_algorithm, tolerance, max_steps);
				delete potential;

			} else if (strcmp (tempstring, "input") == 0) {
				// read ConfigPack
				size_t idx_beg = 0;
				{
					ConfigurationPack save_(savename+"_InitConfig");				
					num_in_save = save_.NumConfig();
				}

				size_t numConfig_upper = std::max(num_in_load, numConfig);

				ofile << "We have already used " << num_in_save << " configurations loaded ConfigPack" << std::endl;
				ofile << "compute from the " << num_in_save << "th configuration to the ";
				ofile << numConfig << "th configuration (out of " << numConfig_upper << ") total" << std::endl;

				for (size_t i = num_in_save; i < numConfig; i++) {
					Configuration pConfig = GetInitConfigs(i);
					pConfig.PrepareIterateThroughNeighbors(sigma);

					CollectiveCoordinateMultiRun(&pConfig, potential, rngGod, savename, 1, TimeLimit, opti_algorithm, tolerance, max_steps);
					}

				delete potential;				
			}

		} else {
			ofile << "mode " << mode << " is undefined!\n";
			return 1;
		}

		// Some basic analyses
		// Compute the nearest distance of each config
		ConfigurationPack ConfigSet;
		if (strcmp (mode, "MD") == 0) {
			ConfigSet.Open((savename ).c_str());
		} else if (strcmp (mode, "ground") == 0) {
			ConfigSet.Open((savename + "_Success").c_str());
		}

		double rmin = L;
		{
			ofile << "Minimum interparticle distances in each configuration:" << std::endl;
			for (int i = 0; i < ConfigSet.NumConfig(); i++) {
				Configuration x = ConfigSet.GetConfig(i);
				double rm = L;
				for (size_t j = 0; j < x.NumParticle(); j++) {
					double temp = x.NearestParticleDistance(j);
					rm = (temp < rm) ? temp : rm; // talk about cheeky
				}
				ofile << i << ":\t" << rm << std::endl;
			}
		}

		// anonymous function for computing isotropic pair correlation function
		auto getC = [&rmin, &ConfigSet](size_t i) -> Configuration {
			Configuration c = ConfigSet.GetConfig(i);
			double R = rmin;
			for (size_t j = 0; j < c.NumParticle(); j++) {
				double temp = c.NearestParticleDistance(j);
				R = (temp < R) ? temp : R;
			}
			rmin = (R < rmin) ? R : rmin;
			return c;
		};

		// COMPUTE PAIR CORRELATION
		std::vector<GeometryVector> g2;
		numConfig = ConfigSet.NumConfig();
		IsotropicTwoPairCorrelation(getC, numConfig, 0.2 * L , g2);
		WriteFunction(g2, (savename + "_g2").c_str());
		phi = 1.0*HyperSphere_Volume(dim, rmin / 2.0);
		ofile << "Maximum Packing Fraction = " << phi << std::endl;

		// COMPUTE STRUCTURE FACTOR
		std::vector<GeometryVector> Sk;
		// Args: (Configuration, Number of configs, max K radius, max wave number, place to store results)
		IsotropicStructureFactor(getC, numConfig, 5, 400, Sk);
		WriteFunction(Sk, (savename + "_Sk").c_str());

		// COMPUTE NEAREST-NEIGHBOR DISTRIBUTION
		// std::vector<GeometryVector> Hp;
		// NearestNeighborDistrubution(getC, numConfig, Hp);
	
	}
	return 0;
}

int main(int argc, char ** argv){
	char tempstring[1000] = {};
	char tempstring2[300] = {};
	std::istream & ifile = std::cin;
	std::ostream & ofile = std::cout;
	RandomGenerator rngGod(0), rng(0);
	{
		GetCCO(argc, argv); // cf. line ~100
	}
	return 0;
	{
		// generate stitched point configurations from an ensemble.
		size_t num_selected = 0, num_generated = 0;
		std::string loadname, savename;

		ofile << "load an ensemble from \n"; ifile >> loadname;
		ofile << "save as \n"; ifile >> savename;
		
		ConfigurationPack loaded_ensemble(loadname);
		ConfigurationPack result(savename);
		result.Clear();

		ofile << "num. of selected configurations from " + loadname <<"\n";       ifile >> num_selected;
		ofile << "num. of generated configurations, saved as " + savename <<"\n"; ifile >> num_generated;

		int option = 0;
		ofile << "how to (0:identical, 1:distinct) \n"; ifile >> option;

		double L = 0.0, l = 0.0;
		size_t num_configs = 0;
		{
			Configuration c = loaded_ensemble.GetConfig(0);
			l = c.GetBasisVector(0).x[0];
			L = l * num_selected;
			num_configs = loaded_ensemble.NumConfig();
			ofile << "Using " << num_configs << "configurations in an ensemble, we generate " << num_generated << "configurations of N = "<< c.NumParticle() * num_selected << " and L = " << L << "\n";
 		}
		Configuration c = GetUnitCubicBox(1);
		c.Resize(L);

		for (size_t i = 0 ; i < num_generated; i++){
			Configuration finC(c);
			size_t id = 0;
			if (option == 0){
				id = static_cast<size_t> (std::floor(std::floor(num_configs * rng.RandomDouble())));
			}

			for (size_t j = 0; j < num_selected; j++){
				if (option == 1) {
					size_t id = static_cast<size_t> (std::floor(std::floor(num_configs * rng.RandomDouble())));
				}

				Configuration patch = loaded_ensemble.GetConfig(id);

				for (size_t k = 0; k < patch.NumParticle(); k++) {
					finC.Insert("a", 
						finC.CartesianCoord2RelativeCoord( 
							patch.GetCartesianCoordinates(k) + GeometryVector(static_cast<double> (j*l))
						) 
					);

				}

			}
			result.AddConfig(finC);
		}


	}

	return 0;
}


int CollectiveCoordinateMultiRun(Configuration * pConfig, Potential * pPotential, RandomGenerator & gen, std::string Prefix, size_t SampleNumber, time_t TimeLimit, const std::string & AlgorithmName, double E_max, size_t max_num)
{ // Collective Coordinate Optimization Procedure
	DimensionType dim = pConfig->GetDimension();
	size_t Num = pConfig->NumParticle();

	// Jauek-style config packs
	ConfigurationPack AfterRelaxPack(Prefix);
	ConfigurationPack InitConfigPack(Prefix+"_InitConfig");
	ConfigurationPack SuccessPack(Prefix+"_Success");
	if (AfterRelaxPack.NumConfig() > 0) {
		std::cout << "Found " << AfterRelaxPack.NumConfig() << " configurations, continue MultiRun.\n";
	}
	
	// BEGIN SIMULATION
	std::cout << "MultiRun start." << '\n';
	for(size_t idx = 0; idx < SampleNumber; idx++) {
		// Printing to SLURM/.log outputs 
		char delim = '#';
		int drepeats = 40;
		std::cout << "\nAT 'TIME' " << std::time(nullptr)-ProgramStart << ", GENERATING CONFIG " << AfterRelaxPack.NumConfig() << std::endl;
		logfile << "\nAT 'TIME' " << std::time(nullptr)-ProgramStart << ", GENERATING CONFIG " << AfterRelaxPack.NumConfig() << std::endl;
		
		Configuration result(*pConfig);
		InitConfigPack.AddConfig(result);
		
		if (AlgorithmName=="LBFGS") {
			// RelaxStructure_NLOPT(result, *pPotential, 0.0, 0, 0.0, max_num);

		} else if (AlgorithmName=="LocalGradientDescent") {
			RelaxStructure_LocalGradientDescent(result, *pPotential, 0.0, 0, 0.0, max_num);

		} else if (AlgorithmName=="ConjugateGradient") {
			RelaxStructure_ConjugateGradient(result, *pPotential, 0.0, 0, 0.0, max_num);

		} else if (AlgorithmName=="SteepestDescent") {
			RelaxStructure_SteepestDescent(result, *pPotential, 0.0, 0, 0.0, max_num);

		} else if (AlgorithmName=="MINOP") {
			RelaxStructure_MINOP_withoutPertubation(result, *pPotential, 0.0, 0, 0.0, max_num);
			
		} else if (AlgorithmName=="qnLBFGS") {
			// Configuration &, Potential &, double, int, double, size_t,
			RelaxStructure_LBFGS(result, *pPotential, 0.0, 0, 0.0, max_num);
		}

		AfterRelaxPack.AddConfig(result); // ADDING RESULTS WHICH DON'T FAIL (NOT NECESSARILY GROUND STATE)
		std::cout << " -> Add to intermediate pack(s)\n";
		logfile << " -> Add to intermediate pack(s)\n";

		pPotential->SetConfiguration(result);
		double E = pPotential->Energy();
		logfile << "E_relax=" << E;
		
		// ADDING RESULTS WHICH REACH CLASSICAL GROUND STATE
		// we add to both of the ConfigPack and HDF5 file
		if (E < E_max) {
			SuccessPack.AddConfig(result);
			std::cout << " -> Add to success pack(s)\n";
			logfile << " -> Add to success pack(s)\n";

			// We now compute the proximity metric between the initial and final configurations
			// see Batten, et al., J. Chem. Phys., 135, 054104 (2011)
			Configuration finalConfig(result);
			Configuration initialConfig(InitConfigPack.GetConfig(InitConfigPack.NumConfig()-1));

			double proximity_metric;
			double rNN = MeanNearestNeighborDistance(initialConfig);
			double total_squared_displacements(0.0);
			double squared_distance;

			if (finalConfig.NumParticle() != Num) {
				proximity_metric = -1.0;

			} else {
				for (size_t k = 0; k < Num; k++) {
					GeometryVector rdiff = initialConfig.GetCartesianCoordinates(k) - finalConfig.GetCartesianCoordinates(k);
					squared_distance = rdiff.Modulus2();
					total_squared_displacements += squared_distance;
				}

				proximity_metric = std::sqrt(total_squared_displacements);
			}

			std::cout << "Mean nearest-neighbor distance of initial state: lp=" << rNN << std::endl;
			std::cout << "Proximity metric (unscaled) between initial and final states: p=" << proximity_metric << std::endl;
			// std::cout << ", p/N=" << proximity_metric / static_cast<double>(Num);
			// std::cout << ", p/r_NN=" << proximity_metric / rNN << std::endl;
			// Done computing the proximity metric
		}

		if (std::time(nullptr) > TimeLimit) {
			return 0;
		}

		// for (size_t n=0; n<Num; n++) {
		// 	GeometryVector temp(dim);
		// 	for(DimensionType j=0; j<dim; j++)
		// 		temp.x[j]=gen.RandomDouble();
		// 	pConfig->MoveParticle(n, temp);
		// }
	}

	return 0;
}

// int CollectiveCoordinateMD(Configuration * pConfig, Potential * pPotential, RandomGenerator & gen, double TimeStep, double Temperature, std::string Prefix, size_t SampleNumber, size_t StepPerSample, bool AllowRestore, time_t TimeLimit, size_t EquilibrateSamples, bool MDAutoTimeStep)
// {
// 	ConfigurationPack BeforeRelaxPack(Prefix);

// 	size_t OneTenthStepPerSample = StepPerSample/10;
// 	if (OneTenthStepPerSample == 0) {
// 		OneTenthStepPerSample = 1;
// 	}

// 	// RelaxStructure_NLOPT(*pConfig, *pPotential, 0.0, 0, 0.0, 1000);
// 	DimensionType dim = pConfig->GetDimension();
// 	size_t Num = pConfig->NumParticle();
// 	size_t dimTensor = dim*Num;
// 	Potential * pPot=pPotential;
// 	pPot->SetConfiguration(* pConfig);
// 	double E = pPot->Energy();
// 	ParticleMolecularDynamics * psystem = NULL;

// 	bool Restart = false;
// 	signed char stage = 0;
// 	long long step = 0;

// 	if (AllowRestore) {
// 		std::fstream ifile( Prefix+std::string(".MDDump"), std::fstream::in | std::fstream::binary);

// 		if (ifile.good()) {
// 			ifile.read( (char*)(&stage), sizeof(stage) );
// 			ifile.read( (char*)(&step), sizeof(step) );
// 			psystem = new ParticleMolecularDynamics(ifile);
// 			Restart=true;
// 			std::cout << "Continue from " << Prefix+std::string(".MDDump") <<"\n";
// 		}
// 	}

// 	if (Restart == false) {
// 		BeforeRelaxPack.Clear();
// 		psystem = new ParticleMolecularDynamics(*pConfig, TimeStep, 1.0); 
// 	}

// 	size_t NumExistConfig=0;
// 	std::cout<<"CCMD start. Temperature="<<Temperature<<'\n';
// 	logfile<<"CCMD start. Temperature="<<Temperature<<'\n';
	
// 	if (stage == 0) {
// 		stage++;
// 		step = 0;
// 	}

// 	if (stage == 1) {
// 		for (long long i = step; i < EquilibrateSamples; i++) {
// 			if (std::time(nullptr) > TimeLimit || std::time(nullptr) > ::TimeLimit) {
// 				std::fstream ofile( Prefix+std::string(".MDDump"), std::fstream::out | std::fstream::binary);
// 				ofile.write( (char*)(&stage), sizeof(stage) );
// 				ofile.write( (char*)(&i), sizeof(i) );
// 				psystem->WriteBinary(ofile);
// 				delete psystem;

// 				std::fstream ofile2( Prefix+std::string("_continue.txt"), std::fstream::out);
// 				ofile2<<"Not Completed. Please run again\n";

// 				return 0;
// 			}

// 			double c0 = psystem->Position.GetCartesianCoordinates(0).x[0];

// 			for (size_t ii = 0; ii < 2; ii++) {
// 				psystem->SetRandomSpeed(Temperature, gen);
// 				if (MDAutoTimeStep) {	
// 					psystem->Evolve_AutoTimeStep(StepPerSample/2, *pPot, 0.0001/SampleNumber);

// 				} else {
// 					psystem->Evolve(StepPerSample/2, *pPot);
// 				}
// 			}

// 			pPot->SetConfiguration(psystem->Position);
// 			std::cout << "1:" << i << "/" << (EquilibrateSamples) << ", x0=" << psystem->Position.GetCartesianCoordinates(0).x[0] << ", Ep=" << pPot->Energy() << ", Ek=" << psystem->GetKineticEnergy() << ", dt=" << psystem->TimeStep << '\n';;
// 			logfile << "1:" << i << "/" << (EquilibrateSamples) << ", x0=" << psystem->Position.GetCartesianCoordinates(0).x[0] << ", Ep=" << pPot->Energy() << ", Ek=" << psystem->GetKineticEnergy() << ", dt=" << psystem->TimeStep << '\n';;
// 			std::cout.flush();
// 		}

// 		stage++;
// 		step = 0;
// 	}

// 	// stage 2: sample
// 	for (long long i = step; i < SampleNumber; i++) {
// 		if (std::time(nullptr) > TimeLimit || std::time(nullptr) > ::TimeLimit) {
// 			std::fstream ofile( Prefix + std::string(".MDDump"), std::fstream::out | std::fstream::binary);
// 			ofile.write( (char*)(&stage), sizeof(stage) );
// 			ofile.write( (char*)(&i), sizeof(i) );
// 			psystem->WriteBinary(ofile);
// 			delete psystem;

// 			std::fstream ofile2("continue.txt", std::fstream::out);
// 			ofile2 << "Not Completed. Please run again\n";

// 			return 0;
// 		}

// 		std::cout << "at time" << std::time(nullptr)-ProgramStart;
// 		logfile << "at time" << std::time(nullptr)-ProgramStart;
// 		psystem->AndersonEvolve(StepPerSample/2, *pPot, Temperature, 0.01, gen);
// 		psystem->Evolve(StepPerSample/2, *pPot);

// 		Configuration result(psystem->Position);
		
// 		pPot->SetConfiguration(result);
// 		std::cout << ", 2:" << i << "/" << (SampleNumber) << " \tE_relax=" << pPot->Energy() << " \t";
// 		logfile << ", 2:" << i << "/" << (SampleNumber) << " \tE_relax=" << pPot->Energy() << " \t";
// 		pPot->SetConfiguration(psystem->Position);

// 		std::cout << "E_p=" << pPot->Energy() << " \t";
// 		logfile << "E_p=" << pPot->Energy() << " \t";

// 		std::cout << "E_k=" << psystem->GetKineticEnergy() << ", dt=" << psystem->TimeStep << '\n';
// 		logfile << "E_k=" << psystem->GetKineticEnergy() << ", dt=" << psystem->TimeStep << '\n';

// 		BeforeRelaxPack.AddConfig(result);

// 		std::cout.flush();
// 	}
// 	delete psystem;

// 	return 0;
// }
