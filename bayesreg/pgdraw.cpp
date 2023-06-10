/*
 * This file implements the Polya-gamma sampler PG(1,z).
 * This is a C++ implementation of Algorithm 6 in PhD thesis of Jesse 
 * Bennett Windle, 2013
 * URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
 *
 * Our implementation takes advantage of the OpenMP parallelisation
 * API and features parallel sampling of latent variables Omega
 * in Bayesian logistic regression.
 *
 * The code uses C11 random number generators and special math 
 * functions from <cmath> (i.e., erfc).
 *
 * References:
 *
 *   Jesse Bennett Windle
 *   Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
 *   with High-Frequency Data and Sampling P´olya-Gamma Random Variates for
 *   Posterior Distributions Derived from Logistic Likelihoods  
 *   PhD Thesis, 2013   
 *
 *   Damien, P. & Walker, S. G. Sampling Truncated Normal, Beta, and Gamma Densities 
 *   Journal of Computational and Graphical Statistics, 2001, 10, 206-215
 *
 * (c) Copyright Enes Maklic and Daniel F Schmidt, 2017
 */

#include "mex.h"

#include <cmath>
#include <random>

#include <omp.h>


// Mathematical constants computed using Wolfram Alpha
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

// FCN prototypes
double pgdraw(double, double);
double samplepg(double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);
double normcdf(double);
double tinvgauss(double, double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);        
double truncgamma(std::mt19937&, std::uniform_real_distribution<> &);
double exprnd(double, std::mt19937 &, std::uniform_real_distribution<> &);
double randinvg(double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);
double aterm(int, double, double);



// MATLAB executation starts here
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    // Get input params
    double* scale = (double*) mxGetData(prhs[0]);  
    const mwSize *dims = mxGetDimensions(prhs[0]);
  
    // Create output vector
    int n = (int) dims[0];
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);  
    
    // Output vector to store the PolyaGamma samples
    double *mat = mxGetPr(plhs[0]);

    // parallelize this
    #pragma omp parallel
    {    
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<> randu(0, 1);
        std::normal_distribution<> randn(0, 1);
              
        // Sample a vector of RNGs from PG(1, z)
        #pragma omp for
        for(int j=0; j<n; j++) {
            mat[j] = samplepg(scale[j], rng, randu, randn);
        }    
    }
}

// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z, std::mt19937 &rng, std::uniform_real_distribution<> &randu, std::normal_distribution<> &randn)
{
    //  PG(b, z) = 0.25 * J*(b, z/2)
    z = fabs(z) * 0.5;

    // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
    double t = MATH_2_PI;
        
    // Compute p, q and the ratio q / (q + p)
    // (derived from scratch; derivation is not in the original paper)
    double K = z*z/2.0 + MATH_PI2/8.0;
    double logA = log(4) - MATH_LOG_PI - z;
    double logK = log(K);
    double Kt = K * t;
    double w = sqrt(MATH_PI_2);

    double logf1 = logA + log(normcdf(w*(t*z - 1))) + logK + Kt;
    double logf2 = logA + 2*z  + log(normcdf(-w*(t*z+1))) + logK + Kt;
    double p_over_q = exp(logf1) + exp(logf2);
    double ratio = 1 / (1 + p_over_q); 
       
    // Main sampling loop; page 130 of the Windle PhD thesis
    double u, X;
    while(1) {

        // Step 1: Sample X ? g(x|z)
        u = randu(rng);        
        if(u < ratio) {
            // truncated exponential
            X = t + exprnd(1.0, rng, randu)/K;
        }
        else {
            // truncated Inverse Gaussian
            X = tinvgauss(z, t, rng, randu, randn);        
        }
        
        // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
        int i = 1;
        double Sn = aterm(0, X, t);
        double U = randu(rng) * Sn;
        int asgn = -1;
        bool even = false;
        
        while(1) {
            Sn = Sn + asgn * aterm(i, X, t);

            // Accept if n is odd
            if(!even && (U <= Sn)) {
                X = X * 0.25;
                return X;
            }

            // Return to step 1 if n is even
            if(even && (U > Sn)) {
                break;
            }

            even = !even;
            asgn = -asgn;
            i++;
        }    
    }
    
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t)
{
    double f = 0;
    if(x <= t) {
        f = MATH_LOG_PI + log(n + 0.5) + 1.5*(MATH_LOG_2_PI-log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
    }
    else {
        f = MATH_LOG_PI + log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
    }    
    return exp(f);
}


// Generate inverse gaussian random variates
double randinvg(double mu, std::mt19937 &rng, std::uniform_real_distribution<> &randu, std::normal_distribution<> &randn)
{
    double lambda = 1.0;

    // sampling
    double u = randn(rng);
    double V = u*u;
    double out = mu + 0.5*mu * ( mu*V - sqrt(4*mu*V + mu*mu * V*V) );

    if(randu(rng) > mu /(mu+out)) {    
        out = mu*mu / out; 
    }    
    return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma(std::mt19937 &rng, std::uniform_real_distribution<> &randu)
{
    double c = MATH_PI_2;
    double X, gX;

    bool done = false;
    while(!done) {
        X = exprnd(1.0, rng, randu) * 2.0 + c;
        //gX = -(0.5)*log(X) + (0.5)*MATH_LOG_PI_2;
        gX = MATH_SQRT_PI_2 / sqrt(X);
        
        //if(log(randu(rng)) <= gX) {
        if(randu(rng) <= gX) {
            done = true;
        }
    }
    
    return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t, std::mt19937 &rng, std::uniform_real_distribution<> &randu, std::normal_distribution<> &randn)
{
    double X, u;
    double mu = 1.0/z;

    // Pick sampler
    if(mu > t) {
        // Sampler based on truncated gamma 
        // Algorithm 3 in the Windle (2013) PhD thesis, page 128
        while(1) {
            u = randu(rng);
            X = 1.0 / truncgamma(rng, randu);
        
            if(log(u) < (-z*z*0.5*X)) {
                break;
            }
        }
    }  
    else {
        // Rejection sampler
        X = t + 1.0;
        while(X >= t) {
            X = randinvg(mu, rng, randu, randn);
        }
    }    
    return X;
}

// Compute the CDF of a standard normal distribution
double normcdf(double x)
{
    return 0.5 * erfc(-x * MATH_SQRT1_2);    
}

// Generate exponential distribution random variates
double exprnd(double mu, std::mt19937 &rng, std::uniform_real_distribution<> &randu)
{
    return -mu * log(1.0 - randu(rng));
}
