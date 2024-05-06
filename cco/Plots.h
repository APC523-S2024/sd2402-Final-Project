
/*
Plotting functions

Sam Dawley
5/2024

References
...
*/

#ifndef PLOTS_INCLUDED
#define PLOTS_INCLUDED

#include "ParameterSet.h"

//#if defined(LATTICESUMSOLVER_INCLUDED)
#include "LatticeSumSolver.h"
void Plots(IsotropicPairPotential * pPot, const std::vector<LatticeSumSolver *> * Solvers, const char * Prefix, size_t Enphasis = 0);
void Plots(const ParameterSet & Param, const std::vector<LatticeSumSolver *> * Solvers, const char * Prefix, size_t Enphasis = 0);
//#endif

//parameter ParticleSizeRescale only works for 2D configurations
void Plot(std::string prefix, const Configuration & c, const std::string & remark="", double ParticleSizeRescale=1.0);


//require that all configurations in this vector have the same basis vectors
void PlotDirectionalStructureFactor2D(const std::vector<Configuration> & vc, long limit, std::string prefix, bool LogScale, double MinSOverride, double MaxSOverride, double TileSize);
void PlotDirectionalStructureFactor2D(const Configuration & c, long limit, std::string prefix, bool LogScale=true, double MinSOverride=0.0, double MaxSOverride=0.0, double TileSize=1.0);
inline void PlotDirectionalStructureFactor2D(std::string prefix, const Configuration & c, long limit, bool LogScale=true, double MinSOverride=0.0, double MaxSOverride=0.0, double TileSize=1.0)
{
	PlotDirectionalStructureFactor2D(c, limit, prefix, LogScale, MinSOverride, MaxSOverride, TileSize);
}

//Plot a path in a 2D configuration
void PlotPath(std::string prefix, const Configuration & List, const std::string & remark, const std::vector<size_t> & path);

void PlotDisplacement(std::string prefix, const Configuration & List, const std::string & remark, const std::vector<double> & displacement);
template<typename add>
void PlotPacking2D(std::string prefix, const PeriodicCellList<add> & List, const std::string & remark, double r);
void PlotPacking2D(std::string prefix, const SpherePacking & List, const char * sphereColor = "b", const std::string & remark="");


void PlotFunction_MathGL(const std::vector<GeometryVector> & result, const std::string & OutFilePrefix, const std::string & xLabel, const std::string & yLabel);

void PlotFunction_Grace(const std::vector<GeometryVector> * presult, size_t NumSet, const std::string & OutFilePrefix, const std::string & xLabel, const std::string & yLabel, const std::vector<std::string> & legends, const std::string & Title, double MinX, double MaxX, double TickX, double MinY, double MaxY, double TickY);
void PlotFunction_Grace(const std::vector<GeometryVector> * presult, size_t NumSet, const std::string & OutFilePrefix, const std::string & xLabel, const std::string & yLabel, const std::vector<std::string> & legends, const std::string & Title);
void PlotFunction_Grace(const std::vector<GeometryVector> & result, const std::string & OutFilePrefix="temp", const std::string & xLabel="", const std::string & yLabel="", const std::string & Title="");

void ReadGraceData(std::vector<std::vector<GeometryVector> > & result, const std::string & InFilePrefix);
void ReadGraceData(std::vector<std::vector<GeometryVector> > & result, std::istream & ifile);
#endif