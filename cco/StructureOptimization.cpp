/*
Implementation of various optimization algorithms

Sam Dawley
5/2024

References
...
*/

#include "StructureOptimization.h"

#define GSL_RANGE_CHECK_OFF

const double MaxVolume=10000000;
const double MinVolume=0.01;
const double MaxLength2=4000000;
const double MinVolumeCoeff = 0.0; //structure is invalid if Volume < MinVolumeCoeff*|a1|*|a2|*...*|ad|
size_t StructureOptimization_SaveConfigurationInterval=std::numeric_limits<size_t>::max();
bool StructureOptimization_SaveNumberedConfiguration = false;

// /* STEEPEST DESCENT SOURCE CODE */
// // #include <config.h>
// #include <gsl/gsl_multimin.h>
// #include <gsl/gsl_blas_types.h>
// // #include <gsl/gsl_blas.h> // included in this file's header file

// typedef struct
// {
//   double step;
//   double max_step;
//   double tol;
//   gsl_vector *x1;
//   gsl_vector *g1;
// }
// steepest_descent_state_t;

// static int
// steepest_descent_alloc (void *vstate, size_t n)
// {
//   steepest_descent_state_t *state = (steepest_descent_state_t *) vstate;

//   state->x1 = gsl_vector_alloc (n);

//   if (state->x1 == NULL)
//     {
//       GSL_ERROR ("failed to allocate space for x1", GSL_ENOMEM);
//     }

//   state->g1 = gsl_vector_alloc (n);

//   if (state->g1 == NULL)
//     {
//       gsl_vector_free (state->x1);
//       GSL_ERROR ("failed to allocate space for g1", GSL_ENOMEM);
//     }

//   return GSL_SUCCESS;
// }

// static int
// steepest_descent_set (void *vstate, gsl_multimin_function_fdf * fdf,
//                       const gsl_vector * x, double *f,
//                       gsl_vector * gradient, double step_size, double tol)
// {
//   steepest_descent_state_t *state = (steepest_descent_state_t *) vstate;

//   GSL_MULTIMIN_FN_EVAL_F_DF (fdf, x, f, gradient);

//   state->step = step_size;
//   state->max_step = step_size;
//   state->tol = tol;

//   return GSL_SUCCESS;
// }


// static void
// steepest_descent_free (void *vstate)
// {
//   steepest_descent_state_t *state = (steepest_descent_state_t *) vstate;

//   gsl_vector_free (state->x1);
//   gsl_vector_free (state->g1);
// }

// static int
// steepest_descent_restart (void *vstate)
// {
//   steepest_descent_state_t *state = (steepest_descent_state_t *) vstate;

//   state->step = state->max_step;

//   return GSL_SUCCESS;
// }

// static int
// steepest_descent_iterate (void *vstate, gsl_multimin_function_fdf * fdf,
//                           gsl_vector * x, double *f,
//                           gsl_vector * gradient, gsl_vector * dx)
// {
//   steepest_descent_state_t *state = (steepest_descent_state_t *) vstate;

//   gsl_vector *x1 = state->x1;
//   gsl_vector *g1 = state->g1;

//   double f0 = *f;
//   double f1;
//   double step = state->step, tol = state->tol;

//   int failed = 0;

//   /* compute new trial point at x1= x - step * dir, where dir is the
//      normalized gradient */

//   double gnorm = gsl_blas_dnrm2 (gradient);

//   if (gnorm == 0.0)
//     {
//       gsl_vector_set_zero (dx);
//       return GSL_ENOPROG;
//     }

// trial:
//   gsl_vector_set_zero (dx);
//   gsl_blas_daxpy (-step / gnorm, gradient, dx);

//   gsl_vector_memcpy (x1, x);
//   gsl_blas_daxpy (1.0, dx, x1);

//   if (gsl_vector_equal (x, x1)) 
//     {
//       return GSL_ENOPROG;
//     }

//   /* evaluate function and gradient at new point x1 */

//   GSL_MULTIMIN_FN_EVAL_F_DF (fdf, x1, &f1, g1);

//   if (f1 > f0)
//     {
//       /* downhill step failed, reduce step-size and try again */

//       failed = 1;
//       step *= tol;
//       goto trial;
//     }

//   if (failed)
//     step *= tol;
//   else
//     step *= 2.0;

//   state->step = step;

//   gsl_vector_memcpy (x, x1);
//   gsl_vector_memcpy (gradient, g1);

//   *f = f1;

//   return GSL_SUCCESS;
// }

// static const gsl_multimin_fdfminimizer_type steepest_descent_type =
//   { "steepest_descent",         /* name */
//   sizeof (steepest_descent_state_t),
//   &steepest_descent_alloc,
//   &steepest_descent_set,
//   &steepest_descent_iterate,
//   &steepest_descent_restart,
//   &steepest_descent_free
// };

// const gsl_multimin_fdfminimizer_type
//   * gsl_multimin_fdfminimizer_steepest_descent = &steepest_descent_type;
/* DONE WITH STEEPEST DESCENT SOURCE CODE */

// codes related to elastic constants
// \epsilon_{ij}=C_{ijkl}*e_{kl}
// void PrintAllElasticConstants(std::ostream & out, Configuration stru, Potential & pot, double Pressure, bool InfTime, double * presult) {
// 	DimensionType dim = stru.GetDimension();

// 	//RelaxStructure(stru, pot, Pressure, 0.0);

// 	for (DimensionType i = 0; i<dim; i++)
// 		for (DimensionType j = 0; j<dim; j++)
// 			for (DimensionType k = 0; k<dim; k++)
// 				for (DimensionType l = 0; l < dim; l++)
// 				{
// 					double c = ElasticConstant(pot, stru, i, j, k, l, Pressure, InfTime);
// 					out << "C" << i << j << k << l << "=" << c << '\n';
// 					if (presult != nullptr)
// 						presult[i*dim*dim*dim + j*dim*dim + k*dim + l] = c;
// 				}
// }

double EnergyDerivativeToDeformation(Potential & pot, Configuration structure, DimensionType i, DimensionType j, double epsilon2, double Pressure, GeometryVector * newbasis) {
	DimensionType dim=structure.GetDimension(); 

	try {
		pot.SetConfiguration(structure);
		double EnergyDerivative = 0;
		std::vector<double> grad;
		grad.resize(dim*dim);
		pot.EnergyDerivativeToBasisVectors(&grad[0], Pressure);
		for (DimensionType t = 0; t < dim; t++)
		{
			GeometryVector temp = structure.GetBasisVector(t);
			EnergyDerivative += grad[t*dim + i] * temp.x[j];
		}
		return EnergyDerivative;
		//return grad[j*dim + i];
	} catch (EnergyDerivativeToBasisVectors_NotImplemented & a) {
		pot.SetConfiguration(structure);
		double prevEnergy = pot.Energy() + Pressure*structure.PeriodicVolume();

		//second structure deformation to calculate force
		for (DimensionType t = 0; t < dim; t++) {
			newbasis[t] = structure.GetBasisVector(t);
			if (i == j) {
				newbasis[t].x[i] += epsilon2*newbasis[t].x[j];
			} else {
				double temp = newbasis[t].x[i];
				newbasis[t].x[i] += 0.5*epsilon2*newbasis[t].x[j];
				newbasis[t].x[j] += 0.5*epsilon2*temp;
			}
		}

		structure.ChangeBasisVector(&newbasis[0]);
		pot.SetConfiguration(structure);
		double afterEnergy = pot.Energy() + Pressure*structure.PeriodicVolume();

		//calculate result
		return (afterEnergy - prevEnergy)/ epsilon2;
	}
}

// double ElasticConstant(Potential & pot, Configuration structure, DimensionType i, DimensionType j, DimensionType k, DimensionType l, double Pressure, bool InfiniteTime)
// {
// 	const double epsilon1 = 1e-6;
// 	const double epsilon2=1e-9;

// 	DimensionType dim = structure.GetDimension(); 
// 	assert(i < dim);
// 	assert(j < dim);
// 	assert(k < dim);
// 	assert(l < dim);
// 	double Volume=structure.PeriodicVolume();
// 	GeometryVector newbasis[::MaxDimension];

// 	// deform structure so that strain_{kl}=epsilon1
// 	for (DimensionType t = 0; t < dim; t++) {
// 		newbasis[t]=structure.GetBasisVector(t);
// 		double temp=newbasis[t].x[k];
// 		newbasis[t].x[k]+=epsilon1*newbasis[t].x[l];
// 	}

// 	double PrevDerivative = EnergyDerivativeToDeformation(pot, structure, i, j, epsilon2, Pressure, newbasis);

// 	structure.ChangeBasisVector(&newbasis[0]);
// 	if (InfiniteTime) {
// 		std::cout << "Relax structure to calculate infinite-time elastic constant.\n";	
// 		RelaxStructure_NLOPT(structure, pot, Pressure, 0, 0.0);
// 	}

// 	double AfterDerivative = EnergyDerivativeToDeformation(pot, structure, i, j, epsilon2, Pressure, newbasis);
// 	return (AfterDerivative - PrevDerivative) / Volume / epsilon1;
// }

// EulerAngles should have size d(d-1)/2
gsl_matrix * GetRotation(std::vector<double> EulerAngles, int d)
{
	gsl_matrix * result = gsl_matrix_alloc(d, d);
	gsl_matrix * temp = gsl_matrix_alloc(d, d);
	gsl_matrix * temp2 = gsl_matrix_calloc(d, d);
	gsl_matrix_set_identity(result);
	std::vector<double>::iterator iter=EulerAngles.begin();

	for (int i = 0; i < d; i++) {
		for(int j = i+1; j < d; j++) {
			gsl_matrix_set_identity(temp);
			double c = std::cos(*iter);
			double s = std::sin(*iter);

			gsl_matrix_set(temp, i, i, c);
			gsl_matrix_set(temp, j, j, c);
			gsl_matrix_set(temp, i, j, s);
			gsl_matrix_set(temp, j, i, (-1.0)*s);
			iter++;

			gsl_matrix_swap(result, temp2);
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp2, temp, 0.0, result);
		}
	}

	gsl_matrix_free(temp);
	gsl_matrix_free(temp2);
	return result;
}


//it is possible to optimize this class by using symmetry of the elastic tensor, not implemented yet
class AllElasticConstants : public BigVector
{
public:
	DimensionType d;
	//relax the structure before calling this function
	AllElasticConstants(const Configuration & c, Potential & p, double Pressure) : BigVector(c.GetDimension()*c.GetDimension()*c.GetDimension()*c.GetDimension())
	{	
		this->d=c.GetDimension();
		for (DimensionType i = 0; i < d; i++) {
			for (DimensionType j = 0; j < d; j++) {
				for (DimensionType k = 0; k < d; k++) {
					for (DimensionType l = 0; l < d; l++) {
						this->x[i*d*d*d+j*d*d+k*d+l] = ElasticConstant(p, c, i, j, k, l, Pressure);
					}
				}
			}
		}
	}

	AllElasticConstants(DimensionType d) : BigVector(d*d*d*d)
	{
		this->d = d;
	}

	AllElasticConstants(const BigVector & src) : BigVector(src)
	{
		this->d = std::floor(0.5+std::pow(src.dim, 0.25));
	}

	AllElasticConstants(const AllElasticConstants & src) : BigVector(src), d(src.d)
	{}

	AllElasticConstants Rotate(gsl_matrix * rot)
	{
		assert(rot->size1 == d);
		assert(rot->size2 == d);
		AllElasticConstants result(*this);
		for (DimensionType i = 0; i < d; i++) {
			for (DimensionType j = 0; j < d; j++) {
				for (DimensionType k = 0; k < d; k++) {
					for (DimensionType l = 0; l < d; l++) {
						result.x[i*d*d*d+j*d*d+k*d+l] = 0;

						for (DimensionType p = 0; p < d; p++) {
							for (DimensionType q = 0; q < d; q++) {
								for (DimensionType r = 0; r < d; r++) {
									for (DimensionType s = 0; s < d; s++) {
										result.x[i*d*d*d+j*d*d+k*d+l] += gsl_matrix_get(rot, i, p)*gsl_matrix_get(rot, j, q)*gsl_matrix_get(rot, k, r)*gsl_matrix_get(rot, l, s)*this->x[p*d*d*d+q*d*d+r*d+s];
									}
								}	
							}
						}
					}
				}	
			}
		}			
		return result;
	}
};

struct ElasticOptimizeStruct {
	int d; //dimension of the configuration
	int n; //number of parameters in the elastic
	std::function<BigVector(const std::vector<double> & param)> GetElasticFunc;
	const AllElasticConstants * pElastic;
	size_t EvaluateCount;
};

double Elastic_Objective(unsigned n, const double *x, double *grad, void *data) {
	ElasticOptimizeStruct * pData = reinterpret_cast<ElasticOptimizeStruct *>(data);
	assert(n == pData->d*(pData->d-1)/2+pData->n);
	assert(grad == nullptr);
	std::vector<double> EulerAngles, param;

	for (int i = 0; i < pData->n; i++) {
		param.push_back(x[i]);
	}

	for(int i = pData->n; i < n; i++) {
		EulerAngles.push_back(x[i]);
	}

	gsl_matrix * rot = GetRotation(EulerAngles, pData->d);
	AllElasticConstants el = pData->GetElasticFunc(param);
	AllElasticConstants el2 = el.Rotate(rot);
	gsl_matrix_free(rot);
	
	double result = (el2-*pData->pElastic).Modulus2()/pData->pElastic->Modulus2();
	pData->EvaluateCount++;

	return result;
}

// void ElasticOptimize(std::function<BigVector(const std::vector<double> & param)> getElasticFunc, int nparam, std::vector<double> & param, std::vector<double> ubound, std::vector<double> lbound, const Configuration & c, Potential & pot, double pressure) {
// 	AllElasticConstants target(c, pot, pressure);
// 	ElasticOptimizeStruct aux;
// 	aux.d = target.d;
// 	aux.n = nparam;
// 	aux.pElastic = &target;
// 	aux.GetElasticFunc = getElasticFunc;
// 	aux.EvaluateCount = 0;

// 	int OptimizerDim = aux.d*(aux.d-1)/2+nparam;
// 	nlopt_opt opt = nlopt_create(NLOPT_LN_SBPLX, OptimizerDim);
// 	param.resize(OptimizerDim, 0.0);
// 	lbound.resize(OptimizerDim, (-1.0)*pi);
// 	ubound.resize(OptimizerDim, pi);
// 	nlopt_set_lower_bounds(opt, &lbound[0]);
// 	nlopt_set_upper_bounds(opt, &ubound[0]);
// 	nlopt_set_ftol_rel(opt, 1e-10);
// 	nlopt_set_min_objective(opt, Elastic_Objective, reinterpret_cast<void *>(&aux));

// 	double result;
// 	nlopt_optimize(opt, &param[0], &result);
// 	if (result > 1e-5) {
// 		std::cout<<"Elastic Optimization complete, resulting distance="<<result<<", distance large, maybe model doesn't fit!\n";
// 	}

// 	param.resize(nparam);
// 	nlopt_destroy(opt);
// }

////////////////////////////////////////////////////////////////////////////
//codes related to RelaxStructure
class InvalidStructure : public std::exception
{
public:
	InvalidStructure()
	{}

	virtual const char * what() const throw()
	{
		return "Optimizer want to try an Invalid Structure!\n";
	}
};


namespace
{
	struct Aux
	// Broad class for storing numerical data contained in objective function to be minimized
	// Morevoer, defines Objective() function
	{
		Potential * pPot;
		size_t NumParticle;
		DimensionType Dim;
		double * pPressure, *pMinDisntace; // why the heck is this mispelled
		size_t NumEvaluation; // TRACKS NUMBER OF FUNCTION CALLS
		Configuration origConfig;
		int Switch;//Switch==1: move Basis Vectors, Switch==0: move atoms, Switch==2:move both
		double RelativeCoordinatesRescale;
		size_t NumParam;
		Aux(const Configuration & orig) : origConfig(orig) {
		}
	};

	Configuration GetConfig(Aux * pAux, const double * x) {
		if (x[0] != x[0]) {
			throw NotANumberFound();
		}

		DimensionType & dim = pAux->Dim;
		assert(dim <= ::MaxDimension);
		GeometryVector base[::MaxDimension];
		Configuration list(pAux->origConfig);

		if (pAux->Switch != 0) {
			for(DimensionType i = 0; i < dim; i++) {
				base[i].SetDimension(dim);

				for (DimensionType j = 0; j < dim; j++) {
					base[i].x[j] = x[i*dim+j];
				}

				if (base[i].Modulus2() > ::MaxLength2) {
					throw InvalidStructure();
				}
			}

			if ( ::Volume(&base[0], dim) < std::pow( *pAux->pMinDisntace, dim)) {
				throw InvalidStructure();
			}

			if ( ::Volume(&base[0], dim) > ::MaxVolume) {
				throw InvalidStructure();
			}

			if ( ::Volume(&base[0], dim) < ::MinVolume) {
				throw InvalidStructure();
			}

			double cubeVolume = 1.0;

			for (DimensionType i = 0; i < dim; i++) {
				cubeVolume *= base[i].Modulus2();
			}

			cubeVolume = std::sqrt(cubeVolume);
			double Skewedness = ::Volume(&base[0], dim) / cubeVolume;
			// a very low skewedness decreases the efficiency of cell list
			if (Skewedness < MinVolumeCoeff) {
				throw InvalidStructure();
			}
			list.ChangeBasisVector(&base[0]);
		}

		if (pAux->Switch == 2) {
			list.RemoveParticles();
			for(size_t i=0; i<pAux->NumParticle; i++)
			{
				GeometryVector ParticleRelative(dim);
				for(DimensionType j=0; j<dim; j++)
					ParticleRelative.x[j]=x[dim*dim+dim*i+j]/pAux->RelativeCoordinatesRescale;
				list.Insert(pAux->origConfig.GetCharacteristics(i).name, ParticleRelative);
			}

		}
		
		if (pAux->Switch == 0) {
			list.RemoveParticles();

			for (size_t i = 0; i < pAux->NumParticle; i++) {
				GeometryVector ParticleRelative(dim);

				for (DimensionType j = 0; j < dim; j++) {
					ParticleRelative.x[j] = x[dim*i+j]/pAux->RelativeCoordinatesRescale;
				}

				list.Insert(pAux->origConfig.GetCharacteristics(i).name, ParticleRelative);
			}
		}

		if (std::time(nullptr) > ::TimeLimit ) {
			std::fstream ofile("StructureOptimization_Progress_TimeLimit.configuration", std::fstream::out | std::fstream::binary);
			list.WriteBinary(ofile);
			ofile.close();
			exit(0);
		}

		if ((pAux->NumEvaluation + 1) % (StructureOptimization_SaveConfigurationInterval) == 0 && StructureOptimization_SaveNumberedConfiguration) {
			ConfigurationPack pk("StructureOptimization_Progress");
			pk.AddConfig(list);
		}

		if ( (pAux->NumEvaluation + 1) % (2*StructureOptimization_SaveConfigurationInterval) == StructureOptimization_SaveConfigurationInterval) {
			std::fstream ofile("StructureOptimization_Progress_a.configuration", std::fstream::out | std::fstream::binary);
			list.WriteBinary(ofile);
			
		} else if( (pAux->NumEvaluation + 1) % (2*StructureOptimization_SaveConfigurationInterval) == 0) {
			std::fstream ofile("StructureOptimization_Progress_b.configuration", std::fstream::out | std::fstream::binary);
			list.WriteBinary(ofile);
		}

		return list;
	}

	double Objective(unsigned n, const double *x, double *grad, void *data)
	{
		Aux * pAux = reinterpret_cast<Aux *>(data);
		DimensionType & dim = pAux->Dim;
		assert(n == pAux->NumParam);

		try { // This block does all the work unless invalid structure is passed
			Configuration list = GetConfig(pAux, x);
			double Volume = list.PeriodicVolume();
			if (Volume < std::pow(::LengthPrecision, static_cast<double>(dim))) {
				return ::MaxEnergy;
			}

			if( *pAux->pMinDisntace > LengthPrecision) {

				for (size_t i = 0; i < list.NumParticle(); i++) {
					bool TooNearNeighbor=false;
					list.IterateThroughNeighbors(list.GetRelativeCoordinates(i), *pAux->pMinDisntace, [&TooNearNeighbor, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) -> void
					{
						if (SourceAtom != i || LatticeShift.Modulus2() != 0) {
							TooNearNeighbor = true;
						}

					}, &TooNearNeighbor);

					if (TooNearNeighbor) {
						return ::MaxEnergy;
					}
				}
			}

			double Pressure = (*pAux->pPressure);

			pAux->pPot->SetConfiguration(list);
			double result = pAux->pPot->Energy()+ Pressure*Volume;

			if (grad != nullptr) {
				for(size_t i = 0; i < n; i++) {
					grad[i]=0.0;
				}

				if(pAux->Switch == 2) {
					std::vector<GeometryVector> forces;
					pAux->pPot->AllForce(forces);

					for (size_t i = 0; i < list.NumParticle(); i++) {
						GeometryVector & Force=forces[i];

						// derivatives of atom coordinates
						for (DimensionType j = 0; j < dim; j++) {
							grad[dim*dim+i*dim+j] = (-1.0)*Force.Dot(list.GetBasisVector(j))/pAux->RelativeCoordinatesRescale;
						}
					}
					
				} else if (pAux->Switch == 0) {
					std::vector<GeometryVector> forces;
					pAux->pPot->AllForce(forces);

					for (size_t i = 0; i < list.NumParticle(); i++) {
						GeometryVector & Force = forces[i];

						for (DimensionType j = 0; j < dim; j++) {
							grad[i*dim+j] = (-1.0)*Force.Dot(list.GetBasisVector(j))/pAux->RelativeCoordinatesRescale;
						}
					}
				}

				if (pAux->Switch != 0) {
					pAux->pPot->EnergyDerivativeToBasisVectors(grad, Pressure);
				}

			}

			pAux->NumEvaluation++;

			return result;
		}

		catch (InvalidStructure a) {
			if (grad != nullptr) {
				for (unsigned i = 0; i < n; i++) {
					grad[i] = 0.0;
				}
			}

			return ::MaxEnergy;
		}
	}

	void Gradient(unsigned n, const double *x, double *grad, void *data) {
		Aux * pAux = reinterpret_cast<Aux *>(data);
		DimensionType & dim = pAux->Dim;
		assert(n == pAux->NumParam);

		try {
			Configuration list = GetConfig(pAux, x);
			double Volume = list.PeriodicVolume();
			if (Volume < std::pow(::LengthPrecision, static_cast<double>(dim))) {
				return ;
			}

			if(*pAux->pMinDisntace > LengthPrecision) {
				for (size_t i = 0; i < list.NumParticle(); i++) {
					//Configuration::particle * pa=list.GetParticle(i);
					bool TooNearNeighbor=false;
					list.IterateThroughNeighbors(list.GetRelativeCoordinates(i), *pAux->pMinDisntace, [&TooNearNeighbor, &i](const GeometryVector & shift, const GeometryVector & LatticeShift, const signed long * PeriodicShift, const size_t SourceAtom) -> void
					{
						if (SourceAtom != i || LatticeShift.Modulus2() != 0) {
							TooNearNeighbor=true;
						}

					}, &TooNearNeighbor);

					if (TooNearNeighbor) {
						return ;
					}
				}
			}

			double Pressure = (*pAux->pPressure);

			pAux->pPot->SetConfiguration(list);

			if (grad != nullptr) {
				for (size_t i = 0; i < n; i++) {
					grad[i] = 0.0;
				}

				if (pAux->Switch == 2) {
					std::vector<GeometryVector> forces;
					pAux->pPot->AllForce(forces);

					for( size_t i = 0; i < list.NumParticle(); i++){
						GeometryVector & Force = forces[i];

						// derivatives of atom coordinates
						for (DimensionType j = 0; j < dim; j++) {
							grad[dim*dim+i*dim+j] = (-1.0)*Force.Dot(list.GetBasisVector(j))/pAux->RelativeCoordinatesRescale;
						}
					}

				} else if (pAux->Switch == 0) {
					std::vector<GeometryVector> forces;
					pAux->pPot->AllForce(forces);

					for (size_t i = 0; i < list.NumParticle(); i++) {
						GeometryVector & Force = forces[i];

						for (DimensionType j = 0; j < dim; j++) {
							grad[i*dim+j] = (-1.0)*Force.Dot(list.GetBasisVector(j))/pAux->RelativeCoordinatesRescale;
						}
					}
				}

				if(pAux->Switch != 0) {
					pAux->pPot->EnergyDerivativeToBasisVectors(grad, Pressure);
				}
			}

			pAux->NumEvaluation++;

			return ;
		} catch(InvalidStructure a) {
			if (grad != nullptr)
				for (unsigned i = 0; i < n; i++) {
					grad[i] = 0.0;
				}
				
			return ;
		}
	}

	// convert the configuration into parameters
	double * GetParam(Configuration & List, int Switch, double rescale, size_t & NumParam)
	{
		DimensionType dim = List.GetDimension();
		if (Switch == 2) {
			NumParam = dim*dim+dim*List.NumParticle();

		} else if (Switch == 1) {
			NumParam = dim*dim;

		} else if (Switch == 0) {
			NumParam = dim*List.NumParticle();

		} else {
			std::cerr << "\nError in StructureOptimization : GetParam() : value of Switch not supported!\n";
			return nullptr;
		}

		double * pParams = new double[NumParam];
		//double * LBounds = new double[NumParam];
		//double * UBounds = new double[NumParam];
		//double * StepSizes = new double[NumParam];
		//double * Tolerences = new double[NumParam];
		//set basis vector parameters
		if (Switch != 0) {
			for (DimensionType i = 0; i < dim; i++) {
				GeometryVector vBase = List.GetBasisVector(i);
				double Length = std::sqrt(vBase.Modulus2());

				for (DimensionType j = 0; j < dim; j++) {
					pParams[i*dim+j] = vBase.x[j];
					//LBounds[i*dim+j]=-HUGE_VAL;
					//UBounds[i*dim+j]=HUGE_VAL;
					//StepSizes[i*dim+j]=0.001*Length;
					//Tolerences[i*dim+j]=1e-10*Length;
				}
			}

			if (Switch == 2) {
				for (size_t i = 0; i < List.NumParticle(); i++) {
					for (DimensionType j = 0; j < dim; j++) {
						//Configuration::particle * pA=List.GetParticle(i);
						pParams[dim*dim+i*dim+j] = List.GetRelativeCoordinates(i).x[j]*rescale;
						//LBounds[dim*dim+i*dim+j] = -HUGE_VAL;
						//UBounds[dim*dim+i*dim+j] = HUGE_VAL;
						//StepSizes[dim*dim+i*dim+j] = 0.001;
						//Tolerences[dim*dim+i*dim+j]=1e-10;
					}
				}
			}

		} else {
			for (size_t i = 0; i < List.NumParticle(); i++) {
				for (DimensionType j = 0; j < dim; j++) {
					pParams[i*dim+j] = List.GetRelativeCoordinates(i).x[j]*rescale;
					//LBounds[i*dim+j] = -HUGE_VAL;
					//UBounds[i*dim+j] = HUGE_VAL;
					//StepSizes[i*dim+j] = 0.001;
					//Tolerences[i*dim+j]=1e-10;
				}
			}
		}
		return pParams;
	}
};

void RelaxStructure_LocalGradientDescent(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep)
{
	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout << "(TOTAL) Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;

	DimensionType dim=List.GetDimension();
	size_t NumParam;

	// Perform optimization
	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = &Pressure;
	aux.pMinDisntace = &MinDistance;
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = 1.0;
	void * pAux = reinterpret_cast<void *>(&aux);

	// Allowed values of 'Switch' parameter include
	// Switch == 0 : move atoms
	// Switch == 1 : move Basis Vectors
	// Switch == 2 : move both
	double * pParams = GetParam(List, Switch, 1.0, NumParam);
	aux.NumParam = NumParam;

	if (pParams == nullptr) {
		return;
	}

	double * gradient = new double [NumParam];
	double * newx = new double [NumParam];
	double StepSize = 1e-3;
	while (StepSize > 1e-15 && MaxStep != 0) {
		double f1 = Objective(NumParam, pParams, gradient, pAux);

		// normalize the gradient
		double sum = 0.0;
		for (int i = 0; i < NumParam; i++)
			sum += gradient[i]*gradient[i];

		double norm = std::sqrt(sum);
		for (int i = 0; i < NumParam; i++)
			gradient[i] /= norm;

		for (int i = 0; i < NumParam; i++)
			newx[i] = pParams[i]-gradient[i]*StepSize;

		double f2 = Objective(NumParam, newx, nullptr, pAux);
		double df = f1-f2;
		double df2 = StepSize*norm;

		if (std::abs(df-df2)/df2 > 0.01) {
			StepSize *= 0.5;
		} else {
			std::memcpy(pParams, newx, sizeof(double)*NumParam);
		}

		MaxStep -= 2;
	}
	
	try {
		Configuration List2 = GetConfig(&aux, pParams);
		List=List2;
	} catch(InvalidStructure a) {
		std::cerr<<"Warning in RelaxStructure_LocalGradientDescent : caught InvalidStructure error, stop structure optimization!\n";
	};

	delete [] gradient;
	delete [] newx;	
	delete [] pParams;
}

// void RelaxStructure_NLOPT_NoDerivative(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, double rescale, size_t MaxStep, double fmin)//Switch==1: move Basis Vectors, Switch==0: move atoms, Switch==2:move both
// {
// 	pot.SetConfiguration(List);
// 	if(pot.Energy()<fmin)
// 		return;
// 	DimensionType dim=List.GetDimension();
// 	size_t NumParam;
// 	nlopt_result status;

// 	//do optimization
// 	Aux aux(List);
// 	aux.Dim = dim;
// 	aux.NumParticle = List.NumParticle();
// 	aux.pPot = &pot;
// 	aux.pPressure = & Pressure;
// 	aux.pMinDisntace = & MinDistance;
// 	aux.NumEvaluation = 0;
// 	aux.Switch = Switch;
// 	aux.RelativeCoordinatesRescale = rescale;
// 	nlopt_opt opt;

// 	double * pParams = GetParam(List, Switch, rescale, NumParam);
// 	aux.NumParam = NumParam;

// 	if (pParams == nullptr) {
// 		return;
// 	}

// 	opt = nlopt_create(NLOPT_LN_SBPLX, NumParam);
// 	//nlopt_set_lower_bounds(opt, LBounds);
// 	//nlopt_set_upper_bounds(opt, UBounds);
// 	//nlopt_set_initial_step(opt, StepSizes);
// 	nlopt_set_min_objective(opt, Objective, reinterpret_cast<void *>(&aux));
// 	//nlopt_set_xtol_abs(opt, Tolerences);
// 	nlopt_set_ftol_rel(opt, 1e-15);
// 	nlopt_set_xtol_rel(opt, 1e-15);
// 	//nlopt_set_maxtime(opt, 300.0);
// 	nlopt_set_stopval(opt, fmin);
// 	nlopt_set_maxeval(opt, MaxStep);

// 	double minf; 

// 	//std::cout<<"NLopt fvalue before:"<<Objective(NumParam, pParams, nullptr, reinterpret_cast<void *>(&aux))<<'\n';

// 	status=nlopt_optimize(opt, pParams, &minf);

// 	//debug temp
// 	//std::cout<<"NLopt return:"<<status<<'\n';
// 	//std::cout<<"NLopt fvalue after:"<<minf<<'\n';

// 	try {
// 		Configuration List2 = GetConfig(&aux, pParams);
// 		List = List2;

// 	} catch(InvalidStructure a) {
// 		std::cerr<<"Warning in RelaxStructure_NLOPT_inner : caught InvalidStructure error, stop structure optimization!\n";
// 	};

// 	nlopt_destroy(opt);
// 	delete [] pParams;
// }

// void RelaxStructure_NLOPT_inner(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, double rescale, size_t MaxStep, double fmin)//Switch==1: move Basis Vectors, Switch==0: move atoms, Switch==2:move both
// {
// 	pot.SetConfiguration(List);
// 	if (pot.Energy() < fmin) {
// 		return;
// 	}

// 	DimensionType dim = List.GetDimension();
// 	size_t NumParam;
// 	nlopt_result status;

// 	//do optimization
// 	Aux aux(List);
// 	aux.Dim = dim;
// 	aux.NumParticle = List.NumParticle();
// 	aux.pPot = &pot;
// 	aux.pPressure = & Pressure;
// 	aux.pMinDisntace = & MinDistance;
// 	aux.NumEvaluation = 0;
// 	aux.Switch = Switch;
// 	aux.RelativeCoordinatesRescale = rescale;
// 	nlopt_opt opt;

// 	double * pParams = GetParam(List, Switch, rescale, NumParam);
// 	aux.NumParam = NumParam;
// 	if (pParams==nullptr) {
// 		return;
// 	}

// 	// All of these are standard in the NLopt library
// 	opt = nlopt_create(NLOPT_LD_LBFGS, NumParam);
// 	nlopt_set_vector_storage(opt, 100);

// 	// 'Objective' is defined within 'Aux' class above
// 	nlopt_set_min_objective(opt, Objective, reinterpret_cast<void *>(&aux));  // (nlopt_opt, objective function, function data)
// 	nlopt_set_stopval(opt, fmin);
// 	nlopt_set_maxeval(opt, MaxStep);

// 	double minf; 

// 	std::cout << "\nNLopt fvalue pre-optimization: " << Objective(NumParam, pParams, nullptr, reinterpret_cast<void *>(&aux));
// 	// status = [xopt (optimal parameters), fmin (corresponding value of objective function), retcode (>0 for success, <0 otherwise)]
// 	status = nlopt_optimize(opt, pParams, &minf);
// 	//debug temp
// 	//std::cout<<"NLopt return:"<<status<<'\n';
// 	std::cout << "\nNLopt fvalue post-optimization: " << minf;
// 	std::cout << "\nTotal number of iterations = " << aux.NumEvaluation << std::endl;

// 	// Update ground state configuration
// 	try {
// 		Configuration List2 = GetConfig(&aux, pParams);
// 		List = List2;
// 	} catch(InvalidStructure a) {
// 		std::cerr << "Warning in RelaxStructure_NLOPT_inner : caught InvalidStructure error, stop structure optimization!\n";
// 	};

// 	nlopt_destroy(opt); // dispose of nlopt object after optimization is complete
// 	delete [] pParams;
// }

// void RelaxStructure_NLOPT(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep)//Switch==1: move Basis Vectors, Switch==0: move atoms, Switch==2:move both
// { // Outer NLOPT loop; works by optimizing at progressively stricter scales 
// 	pot.SetConfiguration(List);
// 	double E = pot.Energy();
// 	double V = List.PeriodicVolume();
// 	std::cout << "(TOTAL) Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;

// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 10000, MaxStep, (-1.0)*MaxEnergy);
// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 1, MaxStep, (-1.0)*MaxEnergy);
// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 0.01, MaxStep, (-1.0)*MaxEnergy);

// 	std::cout << "(TOTAL) Relaxation complete." << std::endl;

// 	pot.SetConfiguration(List);
// 	E = pot.Energy();
// 	V = List.PeriodicVolume();
// 	std::cout<< "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';
// }

// void RelaxStructure_NLOPT_Emin(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep, double Emin)//Switch==1: move Basis Vectors, Switch==0: move atoms, Switch==2:move both
// {
// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 10000, MaxStep, Emin);
// 	pot.SetConfiguration(List);

// 	if (pot.Energy() < Emin) {
// 		return;
// 	}

// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 1, MaxStep, Emin);
// 	pot.SetConfiguration(List);

// 	if (pot.Energy() < Emin) {
// 		return;
// 	}

// 	RelaxStructure_NLOPT_inner(List, pot, Pressure, Switch, MinDistance, 0.01, MaxStep, Emin);
// }

/////////////////////////////////////
// interface for gsl
#include <gsl/gsl_multimin.h>
double my_f (const gsl_vector  * x, void * params) {
	Aux * pParam = reinterpret_cast<Aux *>(params);
	return Objective(pParam->NumParam, x->data, nullptr, params);
}

void my_df (const gsl_vector * x, void * params,gsl_vector * df) {
	Aux * pParam = reinterpret_cast<Aux *>(params);
	Gradient(pParam->NumParam, x->data, df->data, params);
}

void my_fdf (const gsl_vector * x, void * params, double *f, gsl_vector * df) {
	Aux * pParam = reinterpret_cast<Aux *>(params);
	(*f)=Objective(pParam->NumParam, x->data, df->data, params);
}

void RelaxStructure_SteepestDescent(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep)
{
	/*
	Modified steepest descent algorithm to ensure fixed step size at each iteration
	Iterative step takes the from
		x_{k+1} = x_k - \alpha g(x_k)
	where
		x_i    = ith position in phase space
		alpha  = current step size (fixed) - see double 'sd_step_size' below
		g(x_k) = NORMALIZED gradient of objective function (ensures step size depends on alpha exclusively)
	Implementation is far from optimal, but hey, it works!
	Source code from GSL here:
		https://github.com/ampl/gsl/blob/master/multimin/steepest_descent.c
	*/
	size_t iter = 0;
	volatile int status;

	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout << "Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;

	// Objective function parameters
	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;
	gsl_vector *x;
	gsl_vector *current_pos;
	gsl_vector *current_step;
	gsl_multimin_function_fdf my_func;

	// gsl_vector *particle_arc_lengths; // Vector for stoaring arc lengths of all particles
	// gsl_vector_set_zero(particle_arc_lengths);
	
	double current_ss;
	double current_fmin;
	double sd_step_size = 1e-2;
	double sd_tol = 1e-1;
	int print_modulo_this = 100;

	DimensionType dim = List.GetDimension();
	size_t NumParam;

	// shebang
	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = & Pressure;
	aux.pMinDisntace = & MinDistance;
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = 1.0;

	double * pParams = GetParam(List, Switch, 1.0, NumParam);
	aux.NumParam = NumParam;
	if (pParams == nullptr)
		return;

	// Add configurational parameters to objective function
	my_func.n = NumParam;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = reinterpret_cast<void *>(&aux);

	// memory allocation shenanigans
	x = gsl_vector_alloc (NumParam);
	std::memcpy(x->data, pParams, sizeof(*pParams)*NumParam);

	T = gsl_multimin_fdfminimizer_steepest_descent; // STEEPEST DESCENT
	s = gsl_multimin_fdfminimizer_alloc (T, NumParam);
	gsl_multimin_fdfminimizer_set (s, &my_func, x, sd_step_size, sd_tol);

	double PrevF = ::MaxEnergy;
	size_t SameFCount = 0;
	do {
		// getting intermediate step statistics
		if (iter < 400 || iter % print_modulo_this == 0) {
			current_fmin = gsl_multimin_fdfminimizer_minimum(s);
			current_pos = gsl_multimin_fdfminimizer_x(s);
			current_step = gsl_multimin_fdfminimizer_dx(s);
			current_ss = gsl_blas_dnrm2(current_step);

			// computing arc lengths; particle_arc_lengths is updated as if particle_arc_lengths += current_step
			// The naive programmer would do this by summing the steps at each iteration using 'current_step' above
			// I am a naive programmer
			// gsl_blas_daxpy(1.0, current_step, particle_arc_lengths);

			std::cout << "\nITERATION NO=" << iter << ",OBJ_FUNC=" << current_fmin;
			std::cout << ",STEP_SIZE=" << std::setprecision(4) << std::scientific << current_ss << ",CONFIG=\n";
			gsl_vector_fprintf(stdout, current_pos, "%1.10e");
		}

		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient (s->gradient, 1e-13); // |g| < abs_tol=1e-13 (stopping criterion)

		gsl_multimin_fdfminimizer_restart(s); // restart so we fix the initial step size
	}

	while (status == GSL_CONTINUE && iter < MaxStep);

	// Once loop ends (attempt to) update configurations
	// Update fails if optimizer created something nonphysical
	try {
		Configuration List2 = GetConfig(&aux, s->x->data);
		List = List2;
	} catch(InvalidStructure a) {
		std::cerr << "Warning in RelaxStructure_SteepestDescent : caught InvalidStructure error, stop structure optimization!\n";
	};

	// print vector of arc lengths before we finish!
	// double ensemble_arc_length = gsl_blas_dnrm2(particle_arc_lengths);
	// std::cout << "Unscaled ensemble arc length mathcal{L}=" << ensemble_arc_length;
	// std::cout << ", and vector of arc lengths:" << std::endl;
	// gsl_vector_fprintf(stdout, particle_arc_lengths, "%1.10e");

	std::cout << "\n(TOTAL) Relaxation complete. Required " << iter << " iterations." << std::endl;
	pot.SetConfiguration(List);
	E = pot.Energy();
	V = List.PeriodicVolume();
	std::cout<< "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	delete [] pParams;
	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);

}

void RelaxStructure_ConjugateGradient_inner(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, double rescale, size_t MaxStep, double minG)
{
	// MOST OF THIS MIRRORS THE DOCUMENTATION FROM GSL (ALMOST) EXACTLY: 
	// https://www.gnu.org/software/gsl/doc/html/multimin.html
	// Note that default value of minG=1e-17 (from StructureOptimization.hpp)
	pot.SetConfiguration(List);
	if (pot.Energy() < minG) {
		return;
	}
	DimensionType dim=List.GetDimension();
	size_t NumParam;

	// Switch == 0 : move atoms, Switch == 1 : move Basis Vectors, Switch == 2 : move both
	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = &Pressure;
	aux.pMinDisntace = &MinDistance; // this is mispelled in struct Aux and therefore carries over to the ENTIRE project
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = rescale;

	double * pParams = GetParam(List, Switch, rescale, NumParam);
	aux.NumParam = NumParam;
	if (pParams == nullptr) {
		return;
	}

	// Gradient descent shenanigans
	size_t iter = 0;
	volatile int status;

	// Define a general function of n variables to be minimized 
	gsl_vector *x;
	gsl_multimin_function_fdf opti_func;
	gsl_multimin_fdfminimizer *s;
	const gsl_multimin_fdfminimizer_type *T;

	opti_func.n = NumParam;
	opti_func.f = my_f;
	opti_func.df = my_df;
	opti_func.fdf = my_fdf;
	opti_func.params = reinterpret_cast<void *>(&aux);

	x = gsl_vector_alloc (NumParam);
	std::memcpy(x->data, pParams, sizeof(*pParams)*NumParam);

	T = gsl_multimin_fdfminimizer_conjugate_pr;

	// Creates a pointer to newly allocated instance of minimizer of type given by 'T'
	s = gsl_multimin_fdfminimizer_alloc (T, NumParam);

	// initialize the minimizer 's' corresponding to function 'opti_func' from initial point 'x'
	// the size of the first trial step size is step_size=0.01
	// accuracy of line minimization specified by tol=1e-1
		// small tolerance of tol=1e-1 is sufficient for most applications since line minimization need only be approximate
		// setting tol=0 forces "exact" search which is very computationally expensive
	gsl_multimin_fdfminimizer_set (s, &opti_func, x, 0.01, 1e-1);
	double objval_pre = gsl_multimin_fdfminimizer_minimum(s);

	double PrevF = ::MaxEnergy;
	size_t SameFCount=0;
	do {
		iter++;
		// Perform a single iteration of the minimizer 's'
		// This minimizer contains the current best estimate of the minimum at all times, accessed via a slew of methods (see docs)
		// If iteration encounters an error code 'GSL_ENOPROG' is returned
		status = gsl_multimin_fdfminimizer_iterate (s); // catches if we're in a local minimum
		if (status)
			break;

		// Test the norm of the gradient 'g=s->gradient' against absolute tolerance epsabs=minG
		// Returns 'GSL_SUCCESS' if |g| < epsabs and 'GSL_CONTINUE' otherwise
		status = gsl_multimin_test_gradient (s->gradient, minG);
	}

	while (status == GSL_CONTINUE && iter < MaxStep);

	// Here we ensure the current configuration is physically plausible
	try {
		Configuration List2 = GetConfig(&aux, s->x->data);
		List=List2;
	} catch(InvalidStructure a) {
		std::cerr<<"Warning in RelaxStructure_ConjugateGradient : caught InvalidStructure error, stop structure optimization!\n";
	};

	double objval_post = gsl_multimin_fdfminimizer_minimum(s);
	std::cout << "\nGSL objective val pre-optimization : " << objval_pre;
	std::cout << "\nGSL objective val post-optimization: " << objval_post;
	std::cout << "\nTotal number of iterations = " << aux.NumEvaluation << std::endl;

	// Free all memory associated with current optimizer 's'
	delete [] pParams;
	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);

	return;
}

void RelaxStructure_ConjugateGradient(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep, double minG)
{
	// MOST OF THIS MIRRORS THE DOCUMENTATION FROM GSL (ALMOST) EXACTLY: 
	// https://www.gnu.org/software/gsl/doc/html/multimin.html
	// Note that default value of minG=1e-17 (from StructureOptimization.hpp)
	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout << "Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;

	// adds extra rescale parameter
	RelaxStructure_ConjugateGradient_inner(List, pot, Pressure, Switch, MinDistance, 10000, MaxStep, minG);
	RelaxStructure_ConjugateGradient_inner(List, pot, Pressure, Switch, MinDistance, 1, MaxStep, minG);
	RelaxStructure_ConjugateGradient_inner(List, pot, Pressure, Switch, MinDistance, 0.01, MaxStep, minG);

	std::cout << "(TOTAL) Relaxation complete." << std::endl;
	pot.SetConfiguration(List);
	E = pot.Energy();
	V = List.PeriodicVolume();
	std::cout<< "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';
}

// void RelaxStructure_ConjugateGradient(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep, double minG)
// {
// 	pot.SetConfiguration(List);
// 	double E = pot.Energy();
// 	double V = List.PeriodicVolume();
// 	std::cout << "Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;
// 	if (E < minG)
// 		return;
// 	DimensionType dim=List.GetDimension();
// 	size_t NumParam;

// 	// Switch == 0 : move atoms, Switch == 1 : move Basis Vectors, Switch == 2 : move both
// 	Aux aux(List);
// 	aux.Dim=dim;
// 	aux.NumParticle=List.NumParticle();
// 	aux.pPot=&pot;
// 	aux.pPressure= & Pressure;
// 	aux.pMinDisntace= & MinDistance;
// 	aux.NumEvaluation=0;
// 	aux.Switch=Switch;
// 	aux.RelativeCoordinatesRescale=1.0;

// 	double * pParams=GetParam(List, Switch, 1.0, NumParam);
// 	aux.NumParam=NumParam;
// 	if(pParams==nullptr)
// 		return;

// 	// Gradient descent shenanigans
// 	size_t iter = 0;
// 	volatile int status;

// 	// Define a general function of n variables to be minimized 
// 	gsl_vector *x;
// 	gsl_multimin_function_fdf opti_func;
// 	gsl_multimin_fdfminimizer *s;
// 	const gsl_multimin_fdfminimizer_type *T;

// 	opti_func.n = NumParam;
// 	opti_func.f = my_f;
// 	opti_func.df = my_df;
// 	opti_func.fdf = my_fdf;
// 	opti_func.params = reinterpret_cast<void *>(&aux);

// 	x = gsl_vector_alloc (NumParam);
// 	std::memcpy(x->data, pParams, sizeof(*pParams)*NumParam);

// 	T = gsl_multimin_fdfminimizer_conjugate_pr;

// 	// Creates a pointer to newly allocated instance of minimizer of type given by 'T'
// 	s = gsl_multimin_fdfminimizer_alloc (T, NumParam);

// 	// initialize the minimizer 's' corresponding to function 'opti_func' from initial point 'x'
// 	// the size of the first trial step size is step_size=0.01
// 	// accuracy of line minimization specified by tol=1e-1
// 		// small tolerance of tol=1e-1 is sufficient for most applications since line minimization need only be approximate
// 		// setting tol=0 forces "exact" search which is very computationally expensive
// 	gsl_multimin_fdfminimizer_set (s, &opti_func, x, 0.01, 1e-1);
// 	double objval_pre = gsl_multimin_fdfminimizer_minimum(s);

// 	double PrevF = ::MaxEnergy;
// 	size_t SameFCount=0;
// 	do {
// 		iter++;
// 		// Perform a single iteration of the minimizer 's'
// 		// This minimizer contains the current best estimate of the minimum at all times, accessed via a slew of methods (see docs)
// 		// If iteration encounters an error code 'GSL_ENOPROG' is returned
// 		status = gsl_multimin_fdfminimizer_iterate (s);
// 		if (status)
// 			break;

// 		// Test the norm of the gradient 'g=s->gradient' against absolute tolerance epsabs=minG
// 		// Returns 'GSL_SUCCESS' if |g| < epsabs and 'GSL_CONTINUE' otherwise
// 		status = gsl_multimin_test_gradient (s->gradient, minG);

// 		double current_objval = gsl_multimin_fdfminimizer_minimum(s);
// 		std::cout << "\nCurrent objective function val: " << current_objval;

// 		// if (s->f == PrevF) {
// 		// 	SameFCount++;
// 		// 	if (SameFCount > 20) // choose stopping point to be when value of objective function remains constant long enough
// 		// 		break;
// 		// } else {
// 		// 	PrevF = s->f;
// 		// 	SameFCount = 0;
// 		// }
// 	}
// 	while (status == GSL_CONTINUE && iter < MaxStep);

// 	try {
// 		Configuration List2 = GetConfig(&aux, s->x->data);
// 		List=List2;
// 	} catch(InvalidStructure a) {
// 		std::cerr<<"Warning in RelaxStructure_ConjugateGradient : caught InvalidStructure error, stop structure optimization!\n";
// 	};

// 	double objval_post = gsl_multimin_fdfminimizer_minimum(s);
// 	std::cout << "\nGSL objective val pre-optimization : " << objval_pre;
// 	std::cout << "\nGSL objective val post-optimization: " << objval_post;
// 	std::cout << "\nTotal number of iterations = " << aux.NumEvaluation << std::endl;

// 	// Free all memory associated with current optimizer 's'
// 	delete [] pParams;
// 	gsl_multimin_fdfminimizer_free (s);
// 	gsl_vector_free (x);

// 	return;
// }

///////////////////////////////////////////////////////
//  RelaxStructure_MINOP
class OptimFunc{
public:
	// Potential * pPot;
	// size_t NumParticle;
	// DimensionType Dim;
	// double * pPressure, *pMinDisntace;
	// Configuration origConfig;
 
	// Configuration GetConfig(const double * x) const;
	Aux * pAux;

	// Functions required by the MINOP/LBFGS algorithm/update:
	int getDataLength() const;
	double evalF(const gsl_vector* input) const;
	void evalG(const gsl_vector* input, gsl_vector* output) const;
	void normalize(gsl_vector* input) const;

	// Destructor, only non-pure virtual method
	OptimFunc() {}
	~OptimFunc();
};

// Functions required by the MINOP algorithm:
int OptimFunc::getDataLength() const
{ // get codomain of objective function
	size_t NumParam;
	if (this->pAux->Switch == 2) {
		NumParam=this->pAux->Dim*this->pAux->Dim+this->pAux->Dim*this->pAux->NumParticle;

	} else if (this->pAux->Switch == 1) {
		NumParam=this->pAux->Dim*this->pAux->Dim;

	} else if (this->pAux->Switch == 0) {
		NumParam=this->pAux->Dim*this->pAux->NumParticle;
	}

	return NumParam;
}

double OptimFunc::evalF(const gsl_vector* input) const
{ // evaluate objective function
	return Objective(this->getDataLength(), input->data, nullptr, reinterpret_cast<void *>(this->pAux));
}

void OptimFunc::evalG(const gsl_vector* input, gsl_vector* output) const
{ // evaluate gradient of objective function
	Gradient(this->getDataLength(), input->data, output->data, reinterpret_cast<void *>(this->pAux));
}

void OptimFunc::normalize(gsl_vector* input) const {
	// should NOT normalize because we don't know the cell size and whether basis vectors are included in input data
	if (this->pAux->Switch == 2) {
		double * data = input->data;

		for (size_t i = 0; i < this->pAux->NumParticle; i++) {
			for (DimensionType j = 0; j < this->pAux->Dim; j++) {
				double & c = data[this->pAux->Dim*this->pAux->Dim+i*this->pAux->Dim+j];
				c -= std::floor(c);
			}
		}

	} else if (this->pAux->Switch == 0) {
		double * data = input->data;

		for (size_t i = 0; i < this->pAux->NumParticle; i++) {
			for (DimensionType j = 0; j < this->pAux->Dim; j++) {
				double & c = data[i*this->pAux->Dim+j];
				c -= std::floor(c);
			}
		}
	}
}

OptimFunc::~OptimFunc() { 
	/* Nothing to do here */ 
}

/*
Configuration OptimFunc::GetConfig(const double * x) const
{
if(x[0]!=x[0])
throw NotANumberFound();
assert(Dim< ::MaxDimension);
GeometryVector base[::MaxDimension];
for(DimensionType i=0; i<Dim; i++)
{
base[i].Dimension=Dim;
for(DimensionType j=0; j<Dim; j++)
base[i].x[j]=x[i*Dim+j];
}
//Configuration list(Dim, cellrank, base);
double Vol=Volume(&base[0], Dim);
if(Vol<std::pow( *this->pMinDisntace, this->Dim))
throw InvalidStructure();
double CellSize=std::pow(Vol/this->NumParticle, 1.0/this->Dim);
Configuration list(Dim, base, CellSize);
for(size_t i=0; i<NumParticle; i++)
{
GeometryVector ParticleRelative(Dim);
for(DimensionType j=0; j<Dim; j++)
ParticleRelative.x[j]=x[Dim*Dim+Dim*i+j];
list.Insert(origConfig.GetParticle(i)->Characteristics.name, ParticleRelative);
}
return list;
}
*/

void RelaxStructure_MINOP_withoutPertubation(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep, double Emin)
{
	// Called from within RelaxStructure_MINOP() below
	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout<< "(TOTAL) Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	DimensionType dim=List.GetDimension();
	size_t NumParam;

	// don't try and reorder this block
	double * ptemp=::GetParam(List, Switch, 1.0, NumParam);
	gsl_vector * vParams = gsl_vector_calloc(NumParam);
	gsl_vector * vResults = gsl_vector_calloc(NumParam);
	double * pParams = vParams->data;
	std::memcpy(pParams, ptemp, NumParam*sizeof(double));
	delete [] ptemp;

	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = & Pressure;
	aux.pMinDisntace = & MinDistance;
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = 1.0;
	aux.NumParam = NumParam;

	OptimFunc opt;
	opt.pAux = &aux;

	void runMINOP79(const OptimFunc& function, const gsl_vector* startPos, gsl_vector* resultPos, int maxIter, double goal, double minG, int verbose);

	try {
		runMINOP79(opt, vParams, vResults, MaxStep, Emin, 1e-13, Verbosity-6); // omit verbosity

	} catch (NotANumberFound except) {
		std::cerr << "found NaN in MinopStructureOptimization, stopping!\n";
		std::cerr << "Output structure to stdcerr\n";
		std::cerr.precision(17);
		::Output(std::cerr, List);
		throw;

	} catch (std::bad_alloc a) {
		std::cerr<<"Warning in MinopStructureOptimization : caught std::bad_alloc error, stop structure optimization!\n";
	}

	try {
		List = GetConfig(&aux, vResults->data);
	} catch(InvalidStructure & a) {
		std::cerr<<"Warning in MinopStructureOptimization : caught InvalidStructure error, stop structure optimization!\n";
	}

	std::cout << "(TOTAL) Relaxation complete." << std::endl;
	pot.SetConfiguration(List);
	E = pot.Energy();
	V = List.PeriodicVolume();
	std::cout<< "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	gsl_vector_free(vParams);
	gsl_vector_free(vResults);
}

void RelaxStructure_MINOP(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep)
{
	Configuration List2(List);
	if (List2.NumParticle() > 0 && List2.GetDimension() != 0) {
		GeometryVector temp = List2.GetRelativeCoordinates(0);
		temp.x[0] += 1e-6;
		List2.MoveParticle(0, temp);
	}

	RelaxStructure_MINOP_withoutPertubation(List, pot, Pressure, Switch, MinDistance, MaxStep);
	// RelaxStructure_MINOP_withoutPertubation(List2, pot, Pressure, Switch, MinDistance, MaxStep);

	pot.SetConfiguration(List);
	double H = pot.Energy() + List.PeriodicVolume()*Pressure;

	pot.SetConfiguration(List2);
	double H2 = pot.Energy() + List2.PeriodicVolume()*Pressure;

	if (H2 < H) {		
		List = List2;
	}

	return;
}

void runMINOP79(const OptimFunc& function, const gsl_vector* startPos, gsl_vector* resultPos, int maxIter, double goal, double minG, int verbose) {
	// REFERENCE:
	// https://link.springer.com/article/10.1007/BF00932218
	// The BLAS (Basic Linear Algebra Subroutine) library supports all of the matrix operations below
	// and are illustrated in the GSL documentation:
	// https://www.gnu.org/software/gsl/doc/html/blas.html

	// maxIter : the number of iterations at which to stop
	// goal    : if the function goes under it, stop
	// verbose : if >= 2, display the optimization progress
	
	// scalar variables
	const int n = function.getDataLength();

	int t_freq; // controls frequency of output
	double stepSize = 1e-3;  // Called \Delta in the paper <=> "initial trust region"
	double maxStepSize = 1;
	double minStepSize = 1e-15;
	const int pfp = 10; // iteration no. we print outputs modulo this

	// objective function values
	double f_x;    // func(v_x)
	double f_xa;   // func(v_xa)
	double norm_g; // Reused incessantly
	double current_ss;

	// vector variables (all of them should start with v_)	
	gsl_vector* v_x   = gsl_vector_calloc(n); // position
	gsl_vector* v_xa  = gsl_vector_calloc(n); // new position
	gsl_vector* v_dx  = gsl_vector_calloc(n); // position difference
	gsl_vector* v_g   = gsl_vector_calloc(n); // gradient
	gsl_vector* v_ga  = gsl_vector_calloc(n); // new gradient
	gsl_vector* v_dg  = gsl_vector_calloc(n); // gradient difference
	gsl_vector* v_n   = gsl_vector_calloc(n); // -H*g
	gsl_vector* v_p   = gsl_vector_calloc(n);
	gsl_vector* v_w   = gsl_vector_calloc(n); // temporary vector
	gsl_vector* v_Gdx = gsl_vector_calloc(n);
	gsl_vector* v_Hdg = gsl_vector_calloc(n);

	// matrix variables (all of which should start with m_)
	// both of these are symmetric so only the lower triangular part of them is actually used
	gsl_matrix* m_G = gsl_matrix_calloc(n, n); // approximate Hessian
	gsl_matrix* m_H = gsl_matrix_calloc(n, n); // inverse approximate Hessian

	// ALGORITHM
	// References at each line correspond to steps in Ref. above
	gsl_vector_memcpy(v_x, startPos);

	// START
	// begin with preprocessing steps
	// initial calculation of objective function f(x) and associated gradient g(x)=grad(f)
	f_x = function.evalF(v_x);
	function.evalG(v_x, v_g);
	norm_g = gsl_blas_dnrm2(v_g);

	// vector whose ith element is the ith component of the gradient g evaluated at x
	gsl_matrix_set_zero(m_G);
	for (int i = 0; i < n; i++) {
		gsl_matrix_set(m_G, i, i, norm_g/stepSize);
	}
	// Stores inverse of vector G above, H = G^{-1}
	gsl_matrix_set_zero(m_H);
	for (int i = 0; i < n; i++) {
		gsl_matrix_set(m_H, i, i, stepSize/norm_g);
	}

	// preprocessing done and iterations begin
	int iter;
	for (iter = 1; iter<=maxIter && f_x>goal && norm_g>minG && stepSize>=minStepSize; iter++) {
			// If verbose == 3, then print the information at every iteration.
			// If verbose == 2, then print it up to 100 times.
			//if(verbose >= 2 && (100*iter+50)/maxIter > (100*iter-50)/maxIter ||
			//	verbose >= 3){
			//		printf("[%6d]  step=%.2e  f(x)=%.8e  |g|=%.4e\n", iter, stepSize, f_x, norm_g);
			//}

			// Repeat until we actually get a decrease of the function value
			// The limit is there to avoid infinite loops
			while (stepSize >= 0.1 * minStepSize) {
				// STEP 1
				// n = -H*g
				// Compute the maxtrix-vector product and sum \alpha Ax + \beta y
				// with \alpha=-1.0, A=m_H, x=v_g, \beta=0.0, y=v_n
				// 'CblasLower' specifies that A is symmetric so we only store the lower triangular part
				gsl_blas_dsymv(CblasLower, -1.0, m_H, v_g, 0.0, v_n);
				double norm_n = gsl_blas_dnrm2(v_n); // finish by computing Euclidean norm of vector n

				// If 'norm_n' < 'stepSize' let \Delta x = -Hg and break and go to Step 4
				// Otherwise calculate a ton of stuff
				if (norm_n < stepSize) {
					gsl_vector_memcpy(v_dx, v_n);
				} else { // if(norm_n < stepSize)
					double gGg = 0.0;
					double gHg = 0.0;

					// Update matrices G and H
					// both are symmetric and therefore contribution from the
					// lower triangular part is same as contribution from the upper triangular part
					for (int i = 0; i < n; i++) {
						// contribution from upper/lower triangular parts
						for (int j = 0; j < i; j++) {
							gGg += 2.0 * gsl_vector_get(v_g, i) * gsl_matrix_get(m_G, i, j) * gsl_vector_get(v_g, j);
							gHg += 2.0 * gsl_vector_get(v_g, i) * gsl_matrix_get(m_H, i, j) * gsl_vector_get(v_g, j);
						}
						// contribution from main diagonals
						gGg += gsl_vector_get(v_g, i) * gsl_matrix_get(m_G, i, i) * gsl_vector_get(v_g, i);
						gHg += gsl_vector_get(v_g, i) * gsl_matrix_get(m_H, i, i) * gsl_vector_get(v_g, i);
					}

					double c = pow(norm_g, 4) / (gGg * gHg);
					double t = 0.2 + 0.8*c;

					// when iter==1, t*norm_n should be equal to stepSize
					// add || iter==1 to go to this part even if we have numerical imprecision
					if (t * norm_n <= stepSize || iter==1) {
						for (int i = 0; i < n; i++) {
							// Compute dx = (stepSize / ||n||) * n
							gsl_vector_set(v_dx, i, (stepSize / norm_n) * gsl_vector_get(v_n, i));
						}

						double temp = gsl_vector_get(v_dx, 0);
						if (temp != temp) throw NotANumberFound();
					} else { // if (t * norm_n <= stepSize) 
						// STEP 2
						// Rescale norm n=||Hg|| by factor of t, i.e., n = t * n
						gsl_blas_dscal(t, v_n);

						// Compute the Cauchy step  p = -(||g||^2 / g*G*g) * g
						{
							double factor = -pow(norm_g, 2) / gGg;
							for (int i = 0; i < n; i++) {
								gsl_vector_set(v_p, i, factor * gsl_vector_get(v_g, i));
							}
						}

						double norm_p = gsl_blas_dnrm2(v_p);
						if (norm_p >= stepSize) {
							// If Cauchy step is greater than specified tolerance update dx and proceed to Step 4
							// Compute dx = -(delta / ||g||) * g;
							double factor = -stepSize / norm_g;
							for (int i = 0; i < n; i++) {
								gsl_vector_set(v_dx, i, factor * gsl_vector_get(v_g, i));
							}

							double temp = gsl_vector_get(v_dx, 0);
							if (temp != temp) throw NotANumberFound();
						} else { // if (gsl_blas_dnrm2(v_p) >= stepSize)
							// STEP 3
							// This is where we end up if none of the other conditions are met
							// Compute w = n - p
							for (int i = 0; i < n; i++) {
								gsl_vector_set(v_w, i, gsl_vector_get(v_n, i) - gsl_vector_get(v_p, i));
							}

							// theta = (stepSize^2 - ||p||^2) / 
							// 		(p*w + sqrt((p*w)^2 + ||w||^2 * (stepSize^2 - ||p||^2)));
							double pw = 0.0;
							gsl_blas_ddot(v_p, v_w, &pw);
							double norm_w = gsl_blas_dnrm2(v_w);

							double theta = (stepSize*stepSize - norm_p*norm_p) /
								(pw + sqrt(pw*pw + norm_w*norm_w * 
								(stepSize*stepSize - norm_p*norm_p)));
							if (theta != theta) theta=0.0;

							// dx = p + theta * w;
							for (int i = 0; i < n; i++) {
								gsl_vector_set(v_dx, i, gsl_vector_get(v_p, i) + theta * gsl_vector_get(v_w, i));
							}
							double temp = gsl_vector_get(v_dx, 0);
							if (temp != temp) 
								throw NotANumberFound();
						} // if (gsl_blas_dnrm2(v_p) >= stepSize)

					} // if (t * norm_n <= stepSize) 

					double temp = gsl_vector_get(v_dx, 0);
					if (temp != temp) throw NotANumberFound();

				} // if(norm_n < stepSize)

				// STEP 4
				// Finally we update the vector x
				// That is, make a move in configuration space based on all the derivative shenanigans above

				// Compute xa = x + dx
				for (int i = 0; i < n; i++) {
					gsl_vector_set(v_xa, i, gsl_vector_get(v_x, i) + gsl_vector_get(v_dx, i));
				}

				// If fa < f PROCEED TO STEP 5 and get out of this loop
				// otherwise cut the step size in half and go back to Step 1
				f_xa = function.evalF(v_xa);
				if (f_xa <= f_x) {
					if (iter % pfp == 0) {
						// elements of v_dx stored as 'x1 x2 ... xN y1 y2 ... yN' in series i.e. as a contiguous 1D array
						current_ss = gsl_blas_dnrm2(v_dx);

						std::cout << "\nITERATION NO=" << iter << ",OBJ_FUNC=" << f_xa;
						std::cout << ",STEP_SIZE=" << std::setprecision(4) << std::scientific << current_ss << ",CONFIG=";
						// gsl_vector_fprintf(stdout, v_xa, "%1.10e");
					}
					break;
				} else {
					stepSize /= 2;
				}
			} // while (stepSize >= 0.1 * minStepSize)

			// STEP 5
			function.evalG(v_xa, v_ga);
			for (int i = 0; i < n; i++) {
				gsl_vector_set(v_dg, i, gsl_vector_get(v_ga, i) - gsl_vector_get(v_g, i));
			}

			// These are needed both to decide the new value of stepSize
			// and for the update of G and H.
			gsl_blas_dsymv(CblasLower, 1.0, m_G, v_dx, 0.0, v_Gdx);
			gsl_blas_dsymv(CblasLower, 1.0, m_H, v_dg, 0.0, v_Hdg);

			// gsl_blas_ddot(a, b, r) computes scalar product of a*b and stores result in r
			double gdx;
			double gadx;
			double dxdg;
			double dxGdx;
			double dgHdg;
			gsl_blas_ddot(v_g, v_dx, &gdx);
			gsl_blas_ddot(v_ga, v_dx, &gadx);
			gsl_blas_ddot(v_dx, v_dg, &dxdg);
			gsl_blas_ddot(v_dx, v_Gdx, &dxGdx);
			gsl_blas_ddot(v_dg, v_Hdg, &dgHdg);

			// Note: fa - f is normally negative, same for right-hand side
			if (f_xa - f_x > 0.1 * (gdx + 0.5 * dxGdx)) {
				stepSize = 0.5 * gsl_blas_dnrm2(v_dx);
			} else {
				// Using v_w as a temporary vector as its value is no longer needed past Step 3
				// Compute w = dg - Gdx
				for (int i = 0; i < n; i++) {
					gsl_vector_set(v_w, i, gsl_vector_get(v_dg, i) - gsl_vector_get(v_Gdx, i));
				}

				if (gdx >= 2.0*gadx || gsl_blas_dnrm2(v_w) <= 0.5 * gsl_blas_dnrm2(v_dg)) {
						stepSize = 2.0 * gsl_blas_dnrm2(v_dx);
				} else {
					stepSize = gsl_blas_dnrm2(v_dx);
				}
			} // if (f_xa - f_x > 0.1 * (gdx + 0.5 * dxGdx))

			/*  I believe this was added as a test
			stepSize *= 2;
			if(stepSize > maxStepSize){
				stepSize = maxStepSize;
			}
			*/

			// After finding a new position which reduces the value of the
			// objective, we need to update the Hessian and inverse Hessian approximations.

			// STEP 6
			f_x = f_xa;
			gsl_vector_memcpy(v_x, v_xa);
			gsl_vector_memcpy(v_g, v_ga);
			norm_g = gsl_blas_dnrm2(v_g);

			// b0 = dg*dx
			double b0;
			gsl_blas_ddot(v_dg, v_dx, &b0);

			// STEP X: BFGS matrices update
			// Instead of following the Dennis and Mei paper,
			// I followed Kaufman's code in using the BFGS update.
			if (b0 >= 1e-30) {
				// Finally, we can update the G and H matrices...
				// G* = G + dg*dg'/(dx'*dg) - (G*dx)*(G*dx)' / (dx'*G*dx)
				// H* = G*^-1
				// H* = H + ((dx'*dg + dg'*H*dg) / (dx'*dg)^2) * dx*dx'
				//        - ((H*dg)*dx' + dx*(H*dg)') / (dx'*dg)
				if (dxdg != 0.0)
					gsl_blas_dsyr(CblasLower, 1.0 / dxdg, v_dg, m_G);
				if (dxGdx != 0.0)
					gsl_blas_dsyr(CblasLower, -1.0 / dxGdx, v_Gdx, m_G);

				if (dxdg != 0.0)
					gsl_blas_dsyr(CblasLower, (dxdg + dgHdg) / (dxdg*dxdg), v_dx, m_H);
				if (dxdg != 0.0)
					gsl_blas_dsyr2(CblasLower, -1.0 / dxdg, v_Hdg, v_dx, m_H);
			} else {
				// b0 is too small, don't update G and H
				if (verbose >= 3) {
					std::cout << "Small b0 : " << b0 << "\n";
				}
			} // if (b0 >= 1e-30)
	} // for (iter = 1; iter<=maxIter && f_x>goal && norm_g>minG && stepSize>=minStepSize; iter++)

	// All my homies hate extra outputs
	if (verbose >= 1) {
		std::cout << "Iterated " << iter << " times.\n";
		std::cout << "Reason for ending: ";
		if (iter > maxIter)         { std::cout << "reached the maximum number of iterations.\n"; }
		if (f_x <= goal)            { std::cout << "reached the desired value for the function.\n"; }
		if (norm_g <= minG)         { std::cout << "gradient became too small.\n"; }
		if (stepSize < minStepSize) { std::cout << "step size became too small.\n"; }
	}

	// We are done, so let's write the answer in its proper variable
	gsl_vector_memcpy(resultPos, v_x);
	function.normalize(resultPos);

	// Freeing memory
	// for vectors
	gsl_vector_free(v_x);
	gsl_vector_free(v_xa);
	gsl_vector_free(v_dx);
	gsl_vector_free(v_g);
	gsl_vector_free(v_ga);
	gsl_vector_free(v_dg);
	gsl_vector_free(v_n);
	gsl_vector_free(v_p);
	gsl_vector_free(v_w);
	gsl_vector_free(v_Gdx);
	gsl_vector_free(v_Hdg);

	// for matrices
	gsl_matrix_free(m_G);
	gsl_matrix_free(m_H);

}

void RelaxStructure_LBFGS(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, size_t MaxStep, double Emin)
{ // Quasi-Newton optimization using LBFGS update step
	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout<< "(TOTAL) Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	DimensionType dim = List.GetDimension();
	size_t NumParam;

	// don't try and reorder this block
	double * ptemp = ::GetParam(List, Switch, 1.0, NumParam);
	gsl_vector * vParams = gsl_vector_calloc(NumParam);
	gsl_vector * vResults = gsl_vector_calloc(NumParam);
	double * pParams = vParams->data;
	std::memcpy(pParams, ptemp, NumParam*sizeof(double));
	delete [] ptemp;

	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = & Pressure;
	aux.pMinDisntace = & MinDistance;
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = 1.0;
	aux.NumParam = NumParam;

	OptimFunc opt;
	opt.pAux = &aux;

	void qnBFGSupdate(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, double rescale, size_t MaxStep, double Emin);

	try {
		// qnBFGSupdate(List, pot, Pressure, Switch, MinDistance, 1000, MaxStep);
		qnBFGSupdate(List, pot, Pressure, Switch, MinDistance, 1, MaxStep, Emin);
		// qnBFGSupdate(List, pot, Pressure, Switch, MinDistance, 0.001, MaxStep);
		
	} catch (NotANumberFound except) {
		std::cerr << "found NaN in qnLBFGSupdate, stopping!\n";
		std::cerr << "Output structure to stdcerr\n";
		std::cerr.precision(17);
		::Output(std::cerr, List);
		throw;

	} catch (std::bad_alloc a) {
		std::cerr<<"Warning in qnLBFGSupdate : caught std::bad_alloc error, stop structure optimization!\n";
	}

	try {
		List = GetConfig(&aux, vResults->data);

	} catch(InvalidStructure & a) {
		std::cerr << "caught InvalidStructure error in qnLBFGSupdate : stop structure optimization!\n";
	}

	std::cout << "(TOTAL) Relaxation complete." << std::endl;
	pot.SetConfiguration(List);
	E = pot.Energy();
	V = List.PeriodicVolume();
	std::cout<< "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	gsl_vector_free(vParams);
	gsl_vector_free(vResults);
}

void qnBFGSupdate(Configuration & List, Potential & pot, double Pressure, int Switch, double MinDistance, double rescale, size_t MaxStep, double Emin)
{ // Quasi-Newton optimization using Limited Memory BFGS update step	
	size_t iter = 0;
	volatile int status;

	pot.SetConfiguration(List);
	double E = pot.Energy();
	double V = List.PeriodicVolume();
	std::cout << "Initial E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V;

	// Objective function parameters
	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;
	gsl_vector *x;
	gsl_vector *current_pos;
	gsl_vector *current_step;
	gsl_multimin_function_fdf my_func;
	
	double current_ss;
	double current_fmin;
	double sd_step_size = 1e-2;
	double sd_tol = 1e-1;
	int print_modulo_this = 100;

	DimensionType dim = List.GetDimension();
	size_t NumParam;

	// shebang
	Aux aux(List);
	aux.Dim = dim;
	aux.NumParticle = List.NumParticle();
	aux.pPot = &pot;
	aux.pPressure = & Pressure;
	aux.pMinDisntace = & MinDistance;
	aux.NumEvaluation = 0;
	aux.Switch = Switch;
	aux.RelativeCoordinatesRescale = 1.0;

	double * pParams = GetParam(List, Switch, 1.0, NumParam);
	aux.NumParam = NumParam;
	if (pParams == nullptr) {
		return;
	}

	// Add configurational parameters to objective function
	my_func.n = NumParam;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = reinterpret_cast<void *>(&aux);

	// memory allocation shenanigans
	x = gsl_vector_alloc(NumParam);
	std::memcpy(x->data, pParams, sizeof(*pParams)*NumParam);

	T = gsl_multimin_fdfminimizer_vector_bfgs2; // BFGS UPDATE
	s = gsl_multimin_fdfminimizer_alloc (T, NumParam);
	gsl_multimin_fdfminimizer_set (s, &my_func, x, sd_step_size, sd_tol);

	double PrevF = ::MaxEnergy;
	size_t SameFCount = 0;
	do {
		// getting intermediate step statistics
		if (iter < 400 || iter % print_modulo_this == 0) {
			current_fmin = gsl_multimin_fdfminimizer_minimum(s);
			current_pos = gsl_multimin_fdfminimizer_x(s);
			current_step = gsl_multimin_fdfminimizer_dx(s);
			current_ss = gsl_blas_dnrm2(current_step);

			std::cout << "\nITERATION NO=" << iter << ",OBJ_FUNC=" << current_fmin;
			std::cout << ",STEP_SIZE=" << std::setprecision(4) << std::scientific << current_ss << ",CONFIG=\n";
			// gsl_vector_fprintf(stdout, current_pos, "%1.10e");
		}

		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status) {
			break;
		}

		// status = gsl_multimin_test_gradient (s->gradient, 1e-13); // |g| < abs_tol=1e-13 (stopping criterion)

	}

	while (status == GSL_CONTINUE && iter < MaxStep);

	// Once loop ends (attempt to) update configurations
	// Update fails if optimizer created something nonphysical
	try {
		Configuration List2 = GetConfig(&aux, s->x->data);
		List = List2;
	} catch(InvalidStructure a) {
		std::cerr << "Warning in qnBFGSupdate : caught InvalidStructure error, stop structure optimization!\n";
	};

	std::cout << "\n(TOTAL) Relaxation complete. Required " << iter << " iterations." << std::endl;
	pot.SetConfiguration(List);
	E = pot.Energy();
	V = List.PeriodicVolume();
	std::cout << "Final E=" << E << ", Vol=" << V << ", Enthalpy=" << E+Pressure*V << '\n';

	delete [] pParams;
	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);
}

// /* 
// 	EVERYTHING BELOW IS MANUAL IMPLEMENTATION OF LIMITED MEMORY BFGS UPDATING FOR OPTIMIZATION
// */
// inline int lewisoverton_linesearch(gsl_vector * x, double & f, gsl_vector * g, double & stp, const gsl_vector * s, const gsl_vector * gp, const double stpmin, const double stpmax, const callback_data_t & cd, const lbfgs_parameter_t & param)
// {
// 		bool brackt = false;
// 		bool touched = false;

//         int count = 0;

//         double finit(f);
// 		double * pdginit;
// 		double dgtest;
// 		double dstest;
//         double mu = 0.0;
// 		double nu = stpmax;

//         /* Check the input parameters for errors. */
//         // if (!(stp > 0.0))
//         // {
//         //     return LBFGSERR_INVALIDPARAMETERS;
//         // }

//         /* Compute the initial gradient in the search direction. */
// 		gsl_blas_ddot(s, gp, pdginit);

//         /* Make sure that s points to a descent direction. */
//         // if (0.0 < dginit)
//         // {
//         //     return LBFGSERR_INCREASEGRADIENT;
//         // }

//         /* The initial value of the cost function. */
//         dgtest = param.f_dec_coeff * (*pdginit);
//         dstest = param.s_curv_coeff * (*pdginit);

//         while (true) {
//             gsl_blas_daxpy(stp, s, x);

//             /* Evaluate the function and gradient values. */
//             f = cd.proc_evaluate(cd.instance, x, g);
//             count++;

//             /* Test for errors. */
//             // if (std::isinf(f) || std::isnan(f))
//             // {
//             //     return LBFGSERR_INVALID_FUNCVAL;
//             // }
//             /* Check the Armijo condition. */
//             if (f > finit + stp * dgtest) {
//                 nu = stp;
//                 brackt = true;
//             } else {
//                 /* Check the weak Wolfe condition. */
// 				double temp_res = gsl_blas_ddot(s, g, pdginit);
//                 if (temp_res < dstest) {
//                     mu = stp;
//                 } else {
//                     return count;
//                 }
//             }
//             // if (param.max_linesearch <= count)
//             // {
//             //     /* Maximum number of iteration. */
//             //     return LBFGSERR_MAXIMUMLINESEARCH;
//             // }
//             // if (brackt && (nu - mu) < param.machine_prec * nu)
//             // {
//             //     /* Relative interval width is at least machine_prec. */
//             //     return LBFGSERR_WIDTHTOOSMALL;
//             // }

//             if (brackt) {
//                 stp = 0.5 * (mu + nu);
//             } else {
//                 stp *= 2.0;
//             }

//             // if (stp < stpmin)
//             // {
//             //     /* The step is the minimum value. */
//             //     return LBFGSERR_MINIMUMSTEP;
//             // }

//             // if (stp > stpmax)
//             // {
//             //     if (touched)
//             //     {
//             //         /* The step is the maximum value. */
//             //         return LBFGSERR_MAXIMUMSTEP;
//             //     }
//             //     else
//             //     {
//             //         /* The maximum value should be tried once. */
//             //         touched = true;
//             //         stp = stpmax;
//             //     }
//             // }
//         }
//     }
