/*
 * This file implements a rejection sampler for variance hyperparameters
 * in local-global shrinkage hierarchies using half-Cauchy priors
 *
 * References:
 *
 *   D F Schmidt and E Makalic
 *   "Comparison of Hyperparameter Samplers for Horseshoe and Horseshoe-like Bayesian Hierarchies"
 *   unpublished
 *
 * (c) Copyright Enes Maklic and Daniel F Schmidt, 2018
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
/*double pgdraw(double, double);
double samplepg(double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);
double normcdf(double);
double tinvgauss(double, double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);        
double truncgamma(std::mt19937&, std::uniform_real_distribution<> &);
double exprnd(double, std::mt19937 &, std::uniform_real_distribution<> &);
double randinvg(double, std::mt19937&, std::uniform_real_distribution<> &, std::normal_distribution<> &);
double aterm(int, double, double);*/

double rej_sample(double, double, std::mt19937&, std::uniform_real_distribution<> &);

// MATLAB executation starts here
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    // Get input params
    double* m = (double*) mxGetData(prhs[0]);
    double* w = (double*) mxGetData(prhs[1]);
    
    const mwSize *dims = mxGetDimensions(prhs[0]);
  
    // Create output vector
    int n = (int) dims[0];
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);  
    
    // Output vector to store the lambda2 samples
    double *x = mxGetPr(plhs[0]);

    // Sample each lambda2
    #pragma omp parallel
    {    
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<> randu(0, 1);
        //std::normal_distribution<> randn(0, 1);
              
        // Sample a vector of RNGs from PG(1, z)
        //#pragma omp for
        for (int j=0; j<n; j++) 
        {
            x[j] = rej_sample(m[j], w[j], rng, randu);
        }
    }
}

double lambert_W(double x)
{
    double w = 1;
    if (x >= 3)
    {
        w = log(x) - log(log(x));
    }
    double v = 1e10;

    double e, f;
    while (fabs(w-v)/fabs(w) > 1e-8)
    {
        v = w;
        e = exp(w);
        f = w*e - x;
        w = w - f/(e*(w+1.0) - (w+2.0)*f/(2.0*w+2.0));
    }
    
    return w;
}

double rej_sample(double m, double w, std::mt19937 &rng, std::uniform_real_distribution<> &randu)
{
    if (w>300)
    {
        w = 300;
    }

    // Mode/2nd-derivative around mode
    double mode = 0.5*(lambert_W(4*exp(2.0*w)*m*w) - 2.0*w);
    double QQ = exp(-2.0*mode);
    double Lm = mode*mode/2.0/w + m*QQ + mode;
    double H = 4.0*QQ*m + 1.0/w;
   
    // Left-hand segment
    double x0 = mode - 0.8/sqrt(H);
 
    QQ = exp(-2.0*x0);
    double g0 = x0/w - 2.0*m*QQ + 1;
    double L0 = x0*x0/2.0/w + m*QQ + x0 - Lm;
    
    // Right-hand segment
    double x1 = mode + 1.1/sqrt(H);
 
    QQ = exp(-2.0*x1);
    double g1 = x1/w - 2.0*m*QQ + 1;
    double L1 = x1*x1/2.0/w + m*QQ + x1 - Lm;
 
    // Meeting points for the three segments
    double left = -(L0-g0*x0-0)/g0;
    double right = -(L1-g1*x1-0)/g1;

    // Normalizing constants for the three densities
    double left_K = -exp(-L0-g0*(left-x0))/g0;
    double right_K = exp(-L1-g1*(right-x1))/g1;
    double mid_K = (right-left);
    double K = left_K+right_K+mid_K;
 
    // Sample
    bool done=false;
    int M=1;
    double x, u, v, f, g;
    while (!done)
    {
        u = randu(rng);
        if (u < left_K/K)
        {
            v = randu(rng);
            x = -log(1-v)/g0 + left;
            f = L0 + g0*(x-x0);
        }
        else if (u < (left_K+mid_K)/K)
        {
            x = randu(rng)*(right-left) + left;
            f = 0;
        } 
        else
        {
            v = randu(rng);
            x = -log(1-v)/g1 + right;
            f = L1 + g1*(x-x1);
        }

        
        // Accept?
        QQ = exp(-2.0*x);
        g = x*x/2.0/w + m*QQ + x - Lm;
        if (log(randu(rng)) < f-g)
        {
            done = true;
        }
        else
        {
            M = M+1;
        }
    }

    return x;
}
