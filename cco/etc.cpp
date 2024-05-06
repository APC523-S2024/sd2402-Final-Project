#include "etc.h"
#include <cmath>

std::fstream logfile("DetailedLog.txt", std::fstream::app|std::fstream::out);

//unsigned long MCThreadsPerConfiguration=0;
unsigned long MCNumParallelConfigurations=0;
unsigned long PTParallelConfigurationsPerSystem=8;
unsigned long OptimizationNumThreadsPerTrial=0;
unsigned long OptimizationNumThreads=7;

bool AlsoWriteEPS = true;//false;

double TwoBodyDistance_MaxLength=4;

double PlotPhononFrequencies_MaxFrequencySquared_Override=0.0;
size_t Verbosity = 3; // controls how many outputs are written
time_t ProgramStart;
time_t TimeLimit;

#include <gsl/gsl_sf_gamma.h>

double HyperSphere_C(DimensionType n)
{
	switch (n) {
	case 1:
		return 2;
	case 2:
		return pi;
	case 3:
		return 4.1887902047863909846;
	case 4:
		return 4.9348022005446793094;
	default:
		return std::pow(::pi, n / 2.0) / gsl_sf_gamma(n / 2.0 + 1);
	}
}
double HyperSphere_SurfaceArea(DimensionType n, double R)
{
	return n*HyperSphere_C(n)*std::pow(R, static_cast<double>(n)-1.0);
}
double HyperSphere_Volume(DimensionType n, double R)
{
	return HyperSphere_C(n)*std::pow(R, static_cast<double>(n));
}

double alpha(DimensionType d, double x) { // x= r/2*R
	if (x > 1.0) {
		return 0;
	}
	else if (x < 0) {
		std::cout << "etc::alpha : sth wrong in x \n";
		return 0;
	}
	else{
		switch (d) {
		case 1:
			return 1.0 - x;
		case 2:
			return 2.0 / pi *(acos(x) - x*sqrt(1.0 - x*x));
		case 3:
			return 1.0 - 1.5*x + 0.5*x*x*x;
		case 4:
			return 2.0 / pi*(acos(x) - (5.0 *x - 2.0 *x*x*x)/3.0 *sqrt(1-x*x));
		case 5:
			return 1.0 - 1.875*x + 1.25 *x*x*x - 0.875 *x*x*x*x*x;
		default:
			std::cout << "etc::alpha : dimension out of the range - please add more generalized definition \n";
			return 0;
		}
	}
}


double R_from_ReducedFraction(double fraction, double NumDensity, DimensionType d) {
	return pow(fraction / (HyperSphere_Volume(d,1.0)*NumDensity), 1.0 / (double)d);
}
double ReducedFraction_from_R(double r, double NumDensity, DimensionType d) {
	return NumDensity*HyperSphere_Volume(d, r);
}

double Superdisk_vol(DimensionType d, double p, double L) {
	double vc = 0;
	double p_ = 1.0 / p;
	switch (d) {
	case 2:
		vc = p_*sqrt(pi) / pow(2, p_ - 1.0)*boost::math::tgamma(p_ / 2.0) / boost::math::tgamma((p_ + 1.0) / 2.0);
		break;
	case 3: //http://gisaxs.com/index.php/Form_Factor:Superball
		vc = 2.0 / 9.0 *p_ *pow(boost::math::tgamma(0.5*p_)*boost::math::tgamma(p_) / boost::math::tgamma(1.5*p_), 2);
		break;
	default:
		std::cout << "Superdisk_area : wrong dimension";
	}
	return pow(L, d)*vc;
}
double Superdisk_surface_area(DimensionType d, double p, double L) {
	return (double)d*Superdisk_vol(d,p,L) / L;
}


#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics.h>
//do a linear fit y=c1*x1+c2*x2+...+cn*xn
//input y[i] is y of the ith data point
//      x[j][i] is xj of the ith data point
//return a vector of <c1, c2, ..., cn>
std::vector<double> MultiVariableLinearFit(const std::vector<double> & y, const std::vector< std::vector<double> > & x, double * pchisq, double * pR2)
{
	size_t ndata = y.size();
	size_t nvar = x.size();
	gsl_multifit_linear_workspace * pwor = gsl_multifit_linear_alloc(ndata, nvar);
	gsl_matrix * px = gsl_matrix_alloc(ndata, nvar);
	gsl_matrix * pcov = gsl_matrix_alloc(nvar, nvar);
	gsl_vector * py = gsl_vector_alloc(ndata);
	gsl_vector * pc = gsl_vector_alloc(nvar);

	for (size_t i = 0; i < ndata; i++)
		*gsl_vector_ptr(py, i) = y[i];
	for (size_t i = 0; i < ndata; i++)
		for (size_t j = 0; j < nvar; j++)
			*gsl_matrix_ptr(px, i, j) = x[j][i];

	double chisq;
	gsl_multifit_linear(px, py, pc, pcov, &chisq, pwor);

	std::vector<double> result(nvar, 0.0);
	for (size_t i = 0; i < nvar; i++)
		result[i] = gsl_vector_get(pc, i);

	if (pchisq != nullptr)
		(*pchisq) = chisq;
	if (pR2 != nullptr)
		(*pR2) = 1.0 - chisq / gsl_stats_tss(py->data, py->stride, py->size);

	gsl_vector_free(pc);
	gsl_vector_free(py);
	gsl_matrix_free(pcov);
	gsl_matrix_free(px);
	gsl_multifit_linear_free(pwor);

	return result;
}

#include <chrono>
long GetPreciseClock(void)
{
	std::chrono::system_clock sc;
	return sc.now().time_since_epoch().count();
}

n_naryNumber::n_naryNumber(unsigned int numDigits, const std::vector<unsigned int> & upperbounds) {
	if (numDigits == upperbounds.size()) {
		this->upperbounds = std::vector<unsigned int>(upperbounds);
		this->digits = std::vector<unsigned int>(numDigits, 0);
	}
	else {
		std::cerr << "Dimensions are mismatched\n";
	}
}

n_naryNumber::n_naryNumber(unsigned int numDigits, unsigned int n) : n_naryNumber::n_naryNumber(numDigits, std::vector<unsigned int>(numDigits, n)) {
}

int n_naryNumber::GetNthDigit(size_t N) const {
	if (N < digits.size())
		return this->digits[N];
	else {
		std::cerr << "exceed the digits\n";
		return 0;
	}
}

n_naryNumber& n_naryNumber::operator++ (int) {
	this->digits[0]++;
	for (size_t i = 0; i < this->digits.size(); i++) {
		if (this->digits[i] >= this->upperbounds[i]) {
			this->digits[i] = 0;
			if (i + 1 == this->digits.size()) {
				std::cout << "Overflow occurs!\n";
			}
			else
				this->digits[i + 1]++;
		}
		else
			break;
	}


	return *this;
}

// Function implementations of the cont class
std::string cont::GetType() {
	if (_int_ptr != nullptr)
		return "int";
	if (_double_ptr != nullptr)
		return "double";
	if (_size_ptr != nullptr)
		return "size";
	if (_string_ptr != nullptr)
		return "string";
	else {
		std::cerr << "Type is undefined\n";
		return "";
	}
}

/** \brief Extraction operator for the cont struct. */
inline std::istream & operator >> (std::istream & is, cont & a) {
	if (a._int_ptr != nullptr) {
		is >> *(a._int_ptr);
	}
	else if (a._double_ptr != nullptr) {
		is >> *(a._double_ptr);
	}
	else if (a._size_ptr != nullptr) {
		is >> *(a._size_ptr);
	}
	else if (a._string_ptr != nullptr) {
		is >> *(a._string_ptr);
	}

	if (is.fail()) {
		is.clear();
		std::string dump;
		is >> dump;
		a.Fail = true;
		std::cout << "Fail: " << dump << " is an inappropriate parameter. \n";
	}
	return is;
}

/** \brief Insertion operator for the cont struct. */
inline std::ostream & operator << (std::ostream & os, cont & a) {
	if (a._int_ptr != nullptr) {
		os << *(a._int_ptr);
	}
	else if (a._double_ptr != nullptr) {
		os << *(a._double_ptr);
	}
	else if (a._size_ptr != nullptr) {
		os << *(a._size_ptr);
	}
	else if (a._string_ptr != nullptr) {
		os << *(a._string_ptr);
	}
	return os;
}


/* Member functions of the Parameters class. */

/** \brief A routine to input parameters.
 *	The routine stops when one inputs "Start", "start", or "START".*/
void Parameters::InputRoutine(std::istream & ifile) {
	std::string tempstring;
	size_t count_wrong_input = 0;
	bool correct_input = false;
	for (;;) {
		ifile >> tempstring;
		if (!case_insensitive)
			std::transform(tempstring.begin(), tempstring.end(), tempstring.begin(), ::tolower);

		//End the routine.
		if (tempstring.compare("start") == 0 || tempstring.compare("Start") == 0 || tempstring.compare("START") == 0) {
			return;
		}

		for (auto i = this->ParamList.begin(); i != this->ParamList.end(); i++) {
			if (i->first.compare(tempstring) == 0) {
				ifile >> i->second;
				correct_input |= !(i->second.IsFail());
				break;
			}

			if (i == this->ParamList.end() - 1)
				std::cerr << "There is no parameter name that corresponds to " << tempstring << "\n";
		}

		if (!correct_input) {
			count_wrong_input++;

			if (count_wrong_input >= limit_wrong_input) {
				std::cerr << " There were too many incorrect inputs!\n";
				return;
			}
		}

		correct_input = false;
	}
}

/** \brief Display all parameter names. */
void Parameters::ShowAllParameterNames(std::ostream & ofile) {
	ofile << "------------------------------------------\n";
	ofile << "\t\tList of parameter names\n";
	ofile << "------------------------------------------\n";
	for (auto i = ParamList.begin(); i != ParamList.end(); i++) {
		ofile << i->first << "(" << i->second.GetType() << ")\n";
	}
}

/** \brief Display all parameter names and their values. */
void Parameters::ShowAllParameters(std::ostream & ofile) {
	ofile << "------------------------------------------\n";
	ofile << "\t\tList of parameters\n";
	ofile << "------------------------------------------\n";
	for (auto i = ParamList.begin(); i != ParamList.end(); i++)
		ofile << i->first << " :\t\t" << i->second << "\n";
}
