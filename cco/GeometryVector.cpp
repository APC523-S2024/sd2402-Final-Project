#include "GeometryVector.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <algorithm>

/** Give a random unit vector whose direction is "statistically isotropic."
 * @param d		Space dimension.
 * @param gen	A random number generator.
 * @return A GeometryVector object whose magnitude is 1. */
GeometryVector RandomUnitVector(DimensionType d, RandomGenerator & gen) {
	GeometryVector result(static_cast<int>(d));
	double theta = 0;
	double z = 0;
	switch (d)
	{
	case 1:
		result.x[0] = (gen.RandomDouble() < 0.5)? -1:1;
		break;
	case 2:
		theta = 2.0*pi*gen.RandomDouble();
		result.x[0] = cos(theta);
		result.x[1] = sin(theta);
		break;
	case 3:
		//Equal-area projection was implemented.
		theta = 2.0*pi*gen.RandomDouble();
		z = 2.0*(gen.RandomDouble() - 0.5);
		result.x[0] = sqrt(1.0 - z*z)*cos(theta);
		result.x[1] = sqrt(1.0 - z*z)*sin(theta);
		result.x[2] = z;
		break;
	default:
		for (int i = 0; i < d; i++)
			result.x[i] = gen.RandomDouble_normal(1.0);// 2.0*(gen.RandomDouble() - 0.5);
		
		double mod = 1.0 / sqrt(result.Dot(result));
		for (int i = 0; i < d; i++)
			result = mod * result;

		break;
	}
	return result;
}

GeometryVector RandomUnitVector(DimensionType d) {
	RandomGenerator gen;
	return RandomUnitVector(d, gen);

}

//rotate these vectors so that
//the 1st basis vector is (c11, 0, 0, ...)
//the 2nd basis vector is (c21, c22, 0, ...)
// ......
//AND the basis vector length l1>=l2>=l3...
void StandardizeOrientation(GeometryVector * bas, DimensionType d)
{
	//checks
	double prevVolume = Volume(bas, d);

	auto CompFunc = [] (const GeometryVector & l, const GeometryVector & r) -> bool
	{
		return l.Modulus2() > r.Modulus2();
	};
	std::sort(bas, bas+d, CompFunc);
	gsl_matrix * pM = gsl_matrix_calloc(d, d);
	for(DimensionType i=0; i<d; i++)
		for(DimensionType j=0; j<d; j++)
			gsl_matrix_set(pM, j, i, bas[i].x[j]);
	gsl_vector * pV = gsl_vector_alloc(d);
	gsl_linalg_QR_decomp(pM, pV);
	for(DimensionType i=0; i<d; i++)
	{
		for(DimensionType j=0; j<=i; j++)
			bas[i].x[j]=gsl_matrix_get(pM, j, i);
		for(DimensionType j=i+1; j<d; j++)
			bas[i].x[j]=0.0;
	}

	double afterVolume = Volume(bas, d);
	if(std::abs( (afterVolume-prevVolume)/prevVolume ) >1e-15)
		std::cerr<<"Error in GeometryVector.cpp : StandardizeOrientation : rotation changed volume!\n";
}


double get_det(gsl_matrix * mat_ptr) 
{
	// Matrix determinant
	int sign=0; 
	double det=0.0; 
	int row_sq = mat_ptr->size1;
	gsl_permutation * p = gsl_permutation_calloc(row_sq);
	gsl_matrix * tmp_ptr = gsl_matrix_calloc(row_sq, row_sq);
	int * signum = &sign;
	gsl_matrix_memcpy(tmp_ptr, mat_ptr);
	gsl_linalg_LU_decomp(tmp_ptr, p, signum);
	det = gsl_linalg_LU_det(tmp_ptr, *signum);
	gsl_permutation_free(p);
	gsl_matrix_free(tmp_ptr);
	return det;
}

double Volume (const GeometryVector * vecs, DimensionType dimension)
{
	// d-dimensional volume of parallelipiped defined by passed vectors
	if (!(dimension > 0))
		return 0.0;
	gsl_matrix * mat=gsl_matrix_alloc(dimension, dimension);
	for( ::DimensionType i=0; i<dimension; i++)
		for( ::DimensionType j=0; j<dimension; j++)
			gsl_matrix_set(mat, i, j, vecs[i].x[j]);

	double result=std::abs(get_det(mat));

	gsl_matrix_free(mat);

	return result;
}

double SimplexVolume (const GeometryVector * vecs, DimensionType d)
{
	double result = Volume(vecs, d);
	for(DimensionType i=2; i<=d; i++)
		result/=i;
	return result;
}


void GeometryVector::WriteBinary(std::ostream & ofile, DimensionType dimension) const
{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
	assert(this->Dimension==dimension);
#endif
	ofile.write( (char *)(this->x), sizeof(double)*dimension);
}
void GeometryVector::ReadBinary(std::istream & ifile, DimensionType dimension)
{
#ifdef GEOMETRYVECTOR_RECORDDIMENSION
	this->Dimension=dimension;
#endif
	ifile.read( (char *)(this->x), sizeof(double)*dimension);
}



void WriteFunction(const std::vector<GeometryVector> & result, std::ostream & ofile, const std::vector<std::string> & fields)
{
	/* first print field names */
	if (fields.size() > 0) {
		for (auto iter = fields.begin(); iter != fields.end(); iter++) {
			ofile << "# " << *iter << "\t";
		}
		ofile << "\n";
	}
	for (auto iter = result.begin(); iter != result.end(); iter++)
	{
		char name[300]={};
		for (DimensionType d = 0; d < ::MaxDimension; d++){
			sprintf(name, "%1.10e", iter->x[d]);
			ofile << name << "\t";
		}
		ofile << '\n';
		//ofile<<iter->x[0]<<" \t"<<iter->x[1]<<" \t"<<iter->x[2]<<" \t"<<iter->x[3]<<" \n";
	}
}
void WriteFunction(const std::vector<GeometryVector> & result, const std::string & OutFilePrefix, const std::vector<std::string> & fields)
{
	// Override WriteFunction() in case that we pass string rather than data itself
	std::string name = OutFilePrefix + std::string(".txt");
	std::fstream ofile(name.c_str(), std::fstream::out);
	WriteFunction(result, ofile, fields);
}
void ReadFunction(std::vector<GeometryVector> & result, std::istream & ifile, size_t NumColumns=4)
{
	if (ifile.good() == false)
	{
		std::cerr << "Error in ReadFunction : input stream is not good!\n";
		return;
	}
	result.clear();
	/* Ignore the comments */
	std::string buffer;	std::istringstream iss;
	while (std::getline(ifile, buffer)) {
		if (buffer[0] != '#') {/* ignore comments or field names. */
			GeometryVector temp(static_cast<DimensionType>(NumColumns));
			iss.str(buffer);
			iss.clear();	//THIS IS VERY CRUCIAL!!!

			for (size_t i = 0; i < NumColumns; i++) {
				iss >> temp.x[i];
			}

			if (ifile.eof() == false)
				result.push_back(temp);
		}
	}
}
void ReadFunction(std::vector<GeometryVector> & result, const std::string & InFilePrefix, size_t NumColumns=4)
{
	// Override ReadFunction() in case that we pass string rather than data itself
	std::string name = InFilePrefix + std::string(".txt");
	std::fstream ifile(name.c_str(), std::fstream::in);
	ReadFunction(result, ifile, NumColumns);
}