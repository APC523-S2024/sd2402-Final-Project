#ifndef ETC_INCLUDED
#define ETC_INCLUDED

const double LengthPrecision=1e-4;
const double EnergyPrecision=1e-4;
const double MaxDistance = 1e10;
const double MaxEnergy=1e10;
const double pi=3.1415926535897932384626433832795028841971693993751058209;
const double E = 2.71828182845904523536;
typedef unsigned short DimensionType;
const DimensionType MaxDimension = 4;
const bool UseCorrectedPotential = false;

extern bool AlsoWriteEPS;//when write bmp files, also write eps

//extern unsigned long MCThreadsPerConfiguration;
extern unsigned long MCNumParallelConfigurations;
extern unsigned long PTParallelConfigurationsPerSystem;
extern unsigned long OptimizationNumThreadsPerTrial;
extern unsigned long OptimizationNumThreads;

extern double TwoBodyDistance_MaxLength;

extern double PlotPhononFrequencies_MaxFrequencySquared_Override;

#ifdef USE_PNG
const char * const FigureFormat = ".png";
#else
const char * const FigureFormat = ".bmp";
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <ctime>

extern std::fstream logfile;

extern size_t Verbosity;

struct Empty
{
	Empty()
	{}
};

extern time_t ProgramStart;

//implemented in : structure optimization, 
extern time_t TimeLimit;


double HyperSphere_C(DimensionType n);
double HyperSphere_SurfaceArea(DimensionType n, double R);
double HyperSphere_Volume(DimensionType n, double R);

//scaled intersection volume of two spheres.
double alpha(DimensionType d, double x);


//compute the radius of spheres from the reduced covered fraction of spheres
double R_from_ReducedFraction(double fraction, double NumDensity, DimensionType d);
//compute the reduced fraction from the given info
double ReducedFraction_from_R(double r, double NumDensity, DimensionType d);

#include <boost/math/special_functions/gamma.hpp>
double Superdisk_vol(DimensionType d, double p, double L);
double Superdisk_surface_area(DimensionType d, double p, double L);

class progress_display
{
public:
	explicit progress_display( unsigned long expected_count,
		std::ostream & os = (Verbosity>1) ? std::cout : logfile ,
		const std::string & s1 = "\n", //leading strings
		const std::string & s2 = "",
		const std::string & s3 = "" )
		// os is hint; implementation may ignore, particularly in embedded systems
		: m_os(os), m_s1(s1), m_s2(s2), m_s3(s3) { restart(expected_count); }

	void restart( unsigned long expected_count )
		//  Effects: display appropriate scale
		//  Postconditions: count()==0, expected_count()==expected_count
	{
		_count = _next_tic_count = _tic = 0;
		_expected_count = expected_count;

		m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
			<< m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
			<< std::endl  // endl implies flush, which ensures display
			<< m_s3;
		if ( !_expected_count ) _expected_count = 1;  // prevent divide by zero
	} // restart

	unsigned long  operator+=( unsigned long increment )
		//  Effects: Display appropriate progress tic if needed.
		//  Postconditions: count()== original count() + increment
		//  Returns: count().
	{
		if ( (_count += increment) >= _next_tic_count ) { display_tic(); }
		return _count;
	}

	unsigned long  operator++()           { return operator+=( 1 ); }
	unsigned long  operator++(int)           { return (operator+=( 1 )-1); }
	unsigned long  count() const          { return _count; }
	unsigned long  expected_count() const { return _expected_count; }

private:
	std::ostream &     m_os;  // may not be present in all imps
	const std::string  m_s1;  // string is more general, safer than 
	const std::string  m_s2;  //  const char *, and efficiency or size are
	const std::string  m_s3;  //  not issues

	unsigned long _count, _expected_count, _next_tic_count;
	unsigned int  _tic;
	void display_tic()
	{
		// use of floating point ensures that both large and small counts
		// work correctly.  static_cast<>() is also used several places
		// to suppress spurious compiler warnings. 
		unsigned int tics_needed =
			static_cast<unsigned int>(
			(static_cast<double>(_count)/_expected_count)*50.0 );
		do { m_os << '*' << std::flush; } while ( ++_tic < tics_needed );
		_next_tic_count = 
			static_cast<unsigned long>((_tic/50.0)*_expected_count);
		if ( _count == _expected_count ) {
			if ( _tic < 51 ) m_os << '*';
			m_os << std::endl;
		}
	} // display_tic
};

#include <vector>
//do a linear fit y=c1*x1+c2*x2+...+cn*xn
//input y[i] is y of the ith data point
//      x[j][i] is xj of the ith data point
//return a vector of <c1, c2, ..., cn>
std::vector<double> MultiVariableLinearFit(const std::vector<double> & y, const std::vector< std::vector<double> > & x, double * pchisq = nullptr, double * pR2 = nullptr);

//In windows, the return value of this function changes 10,000,000 every second
long GetPreciseClock(void);

/** \brief Compute the intersection volume of two spheres.
 *	@param d	Space dimension
 *	@param r1	Radius of sphere1.
 *	@param r2	Radius of sphere2.
 *	@param r	Separation of centers of two spheres.
 *	@return		Intersection volume. */
inline double v2(DimensionType d, double r1, double r2, double r) {
	if (r1 > 0 && r2 > 0 && r >= 0) {
		double R2 = std::max(r1, r2);
		double R1 = std::min(r1, r2);
		double dr2 = R2*R2 - R1*R1;

		if (r <= (R2 - R1) || r<1e-15) {
			return HyperSphere_Volume(d, R1);
		}
		else if (r < sqrt(dr2)) {
			return HyperSphere_Volume(d, R1)*(1.0 - 0.5* alpha(d, (dr2 / r - r) / (2.0*R1))) + 0.5*HyperSphere_Volume(d, R2)*alpha(d, (dr2 / r + r) / (2 * R2));
		}
		else if (r < R1 + R2) {
			return 0.5*(HyperSphere_Volume(d, R1)*alpha(d, (r - dr2 / r) / (2.0*R1)) + HyperSphere_Volume(d, R2)*alpha(d, (r + dr2 / r) / (2.0*R2)));
		}
		else {
			return 0;
		}

	}
	else if(r1 ==0.0 || r2 ==0.0)	{
		return 0.0;
	}
	else {
		std::cout << "etc1::v2; error in input" << std::endl;
		return 0;
	}

}

/* simulate binary and other number systems
unit in each digit can vary
*/
struct n_naryNumber {
protected:
	std::vector<unsigned int> digits;
	std::vector<unsigned int> upperbounds;
public:
	/*	generate 0 of numDigits
	n-th digit: 0 ~ upperbounds[n]-1;
	*/
	n_naryNumber(unsigned int numDigits, const std::vector<unsigned int> & upperbounds);
	n_naryNumber(unsigned int numDigits, unsigned int n);

	int GetNthDigit(size_t N) const;
	/* increment in the number */
	n_naryNumber& operator++ (int);
};


/** Data structures to help inserting parameters. **/
#define STR(name) std::string(#name)

/**\brief A sloppy container to deal with pointers of four types (int, size_t, double, string).*/
struct cont {
	cont(int & _val) : cont() { _int_ptr = &_val; }				//< A constructor for a int pointer.
	cont(double & _val) : cont() { _double_ptr = &_val; }		//< A constructor for a double pointer.
	cont(size_t & _val) : cont() { _size_ptr = &_val; }			//< A constructor for a size_t pointer.
	cont(std::string & _val) : cont() { _string_ptr = &_val; }	//< A constructor for a string pointer.
																/** A copy constructor. */
	cont(const cont & source) : _int_ptr(source._int_ptr), _double_ptr(source._double_ptr), _size_ptr(source._size_ptr), _string_ptr(source._string_ptr) {
		Fail = false;
	}
	/** @return this->Fail. True = operator >> failed. */
	bool IsFail() { return Fail; }
	/** @return the type of the non-null pointer. */
	std::string GetType();

	friend inline std::istream & operator >> (std::istream & is, cont & a);
	friend inline std::ostream & operator << (std::ostream & os, cont & a);
protected:
	//< pointers of four different types. Only one should be used at once. 
	int * _int_ptr;
	double * _double_ptr;
	size_t * _size_ptr;
	std::string * _string_ptr;

	bool Fail;	//< True = operator >> failed.
private:
	//A default constructor.
	cont() {
		_int_ptr = nullptr;	_double_ptr = nullptr;	_size_ptr = nullptr;	_string_ptr = nullptr; Fail = false;
	}
};
/** \brief Extraction operator for the cont struct. */
std::istream & operator >> (std::istream & is, cont & a);
/** \brief Insertion operator for the cont struct. */
std::ostream & operator << (std::ostream & os, cont & a);


/** \brief Data structure to help input a set of parameters. */
struct Parameters {
	//A default constructor
	Parameters(bool Case_Sensitive = false) { case_insensitive = Case_Sensitive; }

	/** A member function to add one item in the parameter list
	 * @param param_name	Name of the parameter.
	 * @param val			A variable into which an input parameter is transferred. 
							The type should be one of int, double, size_t, or std::string.	 */
	template<typename T> void Insert(const std::string & param_name, T & val) {
		std::string new_param_name(param_name);
		if (!this->case_insensitive) {
			// Not Case insensitive -> change the parameter name in lower cases.
			std::transform(new_param_name.begin(), new_param_name.end(), new_param_name.begin(), ::tolower);
		}
		ParamList.emplace_back(new_param_name, cont(val));
	}

	/** \brief A routine to input parameters.
	 *	The routine stops when one inputs "Start", "start", or "START".
	 *	@param ifile	An inputstream. */
	void InputRoutine(std::istream & ifile);

	/** \brief Display all parameter names. 
	 *	@param ofile	An outputstream. */
	void ShowAllParameterNames(std::ostream & ofile);

	/** \brief Display all parameter names and their values. 
	 *	@param ofile	An outputstream. */
	void ShowAllParameters(std::ostream & ofile);

protected:
	std::vector<std::pair<std::string, cont >>	ParamList;	//< ParamList[i].first = Parameter name, and ParamList[i].second = a pointer 
	size_t limit_wrong_input = 4;	//< When there are incorrect parameters more than "limit_wrong_input", InputRoutine(...) automatically ends.
	bool case_insensitive;			//< true = all parameter names are in lower case.
};

#endif
