#include <cufftw.h>

#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include <assert.h>
#include <iostream>
#include <random>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

//#include <Random123/philox.h>

#include <climits>

//#include "dtcmp.h"

#include <fstream>
#include <string>

#include <tclap/CmdLine.h>

#include <curand.h>
#include <curand_kernel.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
extern "C" inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}



struct {
  double x;
  double y;
} Complex;

#define RedNoiseL 1
#define PI 3.1415926
#define ln_10 2.3025851

#define BISECT_ITER 100
#define BISECT_EPSILON 0.00000001

#define BURNIN 10000

class LogNormalDist {
public:

  double LowerLimit;
  double UpperLimit;
  double lambda_s_LN;
  double sigma_LN;
  double maxmode;

  double f_LowerLimit;
  double f_UpperLimit;

  double burninsum = 0.0;

  LogNormalDist(double _LowerLimit, double _UpperLimit,
        double _lambda_s_LN, double _sigma_LN){

    LowerLimit = _LowerLimit;
    UpperLimit = _UpperLimit;
    lambda_s_LN = _lambda_s_LN;
    sigma_LN = _sigma_LN;

    maxmode = exp(log10(lambda_s_LN) * log(10) - 
                  sigma_LN * sigma_LN * log(10) * log(10) );

    f_LowerLimit = f(LowerLimit);
    f_UpperLimit = f(UpperLimit);


  }

  __device__ __host__ double f(double x){
    return 1.0 / sqrt(2*PI) / sigma_LN / ln_10 / x * 
        exp( -(log10(x) - log10(lambda_s_LN)) * (log10(x) - log10(lambda_s_LN)) / 
          2 / (sigma_LN*sigma_LN) );
  }


};

class BrokenPowerLawDist {
public:

  double LowerLimit;
  double UpperLimit;
  double delta1_BPL;
  double delta2_BPL;
  double lambda_s_BPL;

  double f_LowerLimit;
  double f_UpperLimit;

  BrokenPowerLawDist(double _LowerLimit, double _UpperLimit,
        double _delta1_BPL, double _delta2_BPL, double _lambda_s_BPL){

    LowerLimit = _LowerLimit;
    UpperLimit = _UpperLimit;

    delta1_BPL = _delta1_BPL;
    delta2_BPL = _delta2_BPL;
    lambda_s_BPL = _lambda_s_BPL;

    f_LowerLimit = f(LowerLimit);
    f_UpperLimit = f(UpperLimit);

  }

  double f(double x){
    double ratio = x / lambda_s_BPL;
    return 1.0/ln_10/x / (pow(ratio, delta1_BPL) + pow(ratio, delta2_BPL));
  }


};


/* ---FUNCTIONS AND KERNELS ------------------------------- */

__global__ void get_std_array(cufftDoubleReal *ar, cufftDoubleReal *sa, cufftDoubleReal m, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    sa[index] = (ar[index] - m) * (ar[index] - m);

    index += blockDim.x * gridDim.x;

  }

}

__global__ void complex_to_real(cufftDoubleComplex *ar_c, cufftDoubleReal *ar_r, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    ar_r[index] = cuCreal(ar_c[index]);


    index += blockDim.x * gridDim.x;

  }

}


__global__ void get_fftAdj(cufftDoubleComplex *fftAdj_in, cufftDoubleComplex *fft_norm_in, cufftDoubleComplex *ffti_in, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    fftAdj_in[index].x = sqrt(fft_norm_in[index].x * fft_norm_in[index].x + fft_norm_in[index].y * fft_norm_in[index].y)
                        * cos(atan2(ffti_in[index].y,ffti_in[index].x));
    fftAdj_in[index].y = sqrt(fft_norm_in[index].x * fft_norm_in[index].x + fft_norm_in[index].y * fft_norm_in[index].y)
                        * sin(atan2(ffti_in[index].y,ffti_in[index].x));

    index += blockDim.x * gridDim.x;

  }

}

__global__ void norm_llc(cufftDoubleComplex * llc, cufftDoubleReal m_d, cufftDoubleReal s_d, cufftDoubleReal m_tk, cufftDoubleReal s_tk,  int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    llc[index].x = s_d * (llc[index].x - m_tk) / s_tk + m_d;
    llc[index].y = 0.0;

    index += blockDim.x * gridDim.x;

  }
}

__global__ void set_zero_im(cufftDoubleComplex * vv, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    vv[index].y = 0.0;

    index += blockDim.x * gridDim.x;

  }
}


__global__ void def_samples(cufftDoubleComplex * aa, cufftDoubleComplex * ss, cufftDoubleReal * ss_s, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    ss[index].x = aa[index].x;
    ss[index].y = aa[index].y;

    ss_s[index] = aa[index].x;

    index += blockDim.x * gridDim.x;

  }
}


__global__ void prepare_sort(cufftDoubleComplex * lca, cufftDoubleReal * lca_s, cufftDoubleReal * ss_s, cufftDoubleReal * ss_s_it, cufftDoubleReal * gi_s, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    lca_s[index] = lca[index].x;
    gi_s[index] = index + 0.0;
    ss_s_it[index] = ss_s[index] + 0.0;  // take sorting before iteration

    index += blockDim.x * gridDim.x;

  }
}

__global__ void set_ampAdj_ssort(cufftDoubleComplex * aa, cufftDoubleReal * ss, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    aa[index].x = ss[index];
    aa[index].y = 0.0;

    index += blockDim.x * gridDim.x;

  }
}

__device__ inline double ff_ln(double x,double lambda_s_LN,double sigma_LN){
    return 1.0 / sqrt(2*PI) / sigma_LN / ln_10 / x * 
        exp( -(log10(x) - log10(lambda_s_LN)) * (log10(x) - log10(lambda_s_LN)) / 
          2 / (sigma_LN*sigma_LN) );
  }

__device__ double dev_solve_ln(double y, double bisect_low, double bisect_max,double lsln,double sln){

  double bisect_f_low = ff_ln(bisect_low,lsln,sln) - y;
  double bisect_f_max = ff_ln(bisect_max,lsln,sln) - y;
    double xmiddle;
    double f_middle;

    for(int i=0;i<BISECT_ITER;i++){

      xmiddle = (bisect_low + bisect_max) / 2;
      f_middle = ff_ln(xmiddle,lsln,sln) - y;
      if (fabs(f_middle) < BISECT_EPSILON || 
            fabs(bisect_f_max - bisect_low) < BISECT_EPSILON){
        return xmiddle;
      }
      if(f_middle * bisect_f_low >= 0){
        bisect_low = xmiddle;
        bisect_f_low = ff_ln(bisect_low,lsln,sln) - y;
      }else{
        bisect_max = xmiddle;
        bisect_f_max = ff_ln(bisect_max,lsln,sln) - y;
      }
    }

    assert(false);
    return xmiddle;
  } 


__device__ void dev_sample_all_ln(curandState_t* states_1,double UpperLimit,double LowerLimit,double maxmode,double lsln,double sln, int lenb, int idx_in, cufftDoubleComplex * vv){

    double x0, f_x0, xmin, xmax, y0;
    double burninsum=0.0;

    x0 = curand_uniform(states_1) * (UpperLimit - LowerLimit) + LowerLimit;
    f_x0 = ff_ln(x0,lsln,sln);

    for(int iburnin = 0; iburnin < BURNIN; iburnin++){

      y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;

      if(ff_ln(LowerLimit,lsln,sln) > y0){
        xmin = LowerLimit;
      }else{
        xmin = dev_solve_ln(y0, LowerLimit, maxmode,lsln,sln);
      }

      if(ff_ln(UpperLimit,lsln,sln) > y0){
        xmax = UpperLimit;
      }else{
        xmax = dev_solve_ln(y0, maxmode, UpperLimit,lsln,sln);
      }

      x0 = curand_uniform(states_1) * (xmax - xmin) + xmin;
      f_x0 = ff_ln(x0,lsln,sln);

      burninsum ++;
    }

    int ind_start;
    ind_start = idx_in * lenb;

    for(int ii = 0; ii < lenb;  ii++){

      y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;

      if(ff_ln(LowerLimit,lsln,sln) > y0){
        xmin = LowerLimit;
      }else{
        xmin = dev_solve_ln(y0, LowerLimit, maxmode,lsln,sln);
      }

      if(ff_ln(UpperLimit,lsln,sln) > y0){
        xmax = UpperLimit;
      }else{
        xmax = dev_solve_ln(y0, maxmode, UpperLimit,lsln,sln);
      }

      x0 = curand_uniform(states_1) * (xmax - xmin) + xmin;
      f_x0 = ff_ln(x0,lsln,sln);

      vv[ind_start+ii].x = x0;

    }
    
  }


__device__ inline double ff_bpl(double x,double lbpl,double dbpl1, double dbpl2){
    double ratio = x / lbpl;
    return 1.0/ln_10/x / (pow(ratio, dbpl1) + pow(ratio, dbpl2));
  }


__device__ double dev_solve_bpl(double y, double bisect_low, double bisect_max,double lbpl,double dbpl1, double dbpl2){

    double bisect_f_low = ff_bpl(bisect_low,lbpl,dbpl1,dbpl2) - y;
    double bisect_f_max = ff_bpl(bisect_max,lbpl,dbpl1,dbpl2) - y;
    double xmiddle;
    double f_middle;

    for(int i=0;i<BISECT_ITER;i++){

      xmiddle = (bisect_low + bisect_max) / 2;
      f_middle = ff_bpl(xmiddle,lbpl,dbpl1,dbpl2) - y;
      if (fabs(f_middle) < BISECT_EPSILON || 
            fabs(bisect_f_max - bisect_low) < BISECT_EPSILON){
        return xmiddle;
      }
      if(f_middle * bisect_f_low >= 0){
        bisect_low = xmiddle;
        bisect_f_low = ff_bpl(bisect_low,lbpl,dbpl1,dbpl2) - y;
      }else{
        bisect_max = xmiddle;
        bisect_f_max = ff_bpl(bisect_max,lbpl,dbpl1,dbpl2) - y;
      }
    }

    assert(false);
    return xmiddle;
  } 


__device__ void dev_sample_all_bpl(curandState_t* states_1,double UpperLimit,double LowerLimit,double lbpl,double dbpl1, double dbpl2, int lenb, int idx_in, cufftDoubleComplex * vv, double ll_ac, double ul_ac){

    double x0, f_x0, xmin, xmax, y0;
    double burninsum=0.0;

    x0 = curand_uniform(states_1) * (UpperLimit - LowerLimit) + LowerLimit;
    f_x0 = ff_bpl(x0,lbpl,dbpl1,dbpl2);

    for(int iburnin = 0; iburnin < BURNIN; iburnin++){

      y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;

      while( (ff_bpl(LowerLimit,lbpl,dbpl1,dbpl2) - y0) * (ff_bpl(UpperLimit,lbpl,dbpl1,dbpl2) - y0) >= 0){
        y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;
      }

      xmin = LowerLimit;
      xmax = dev_solve_bpl(y0,LowerLimit,UpperLimit,lbpl,dbpl1,dbpl2);


      x0 = curand_uniform(states_1) * (xmax - xmin) + xmin;
      f_x0 = ff_bpl(x0,lbpl,dbpl1,dbpl2);

      burninsum ++;
    }

    int ind_start;
    ind_start = idx_in * lenb;

    int ii = 0;
    do{

      y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;

      while( (ff_bpl(LowerLimit,lbpl,dbpl1,dbpl2) - y0) * (ff_bpl(UpperLimit,lbpl,dbpl1,dbpl2) - y0) >= 0){
        y0 = curand_uniform(states_1) * (f_x0 - 0) + 0;
      }

      xmin = LowerLimit;
      xmax = dev_solve_bpl(y0,LowerLimit,UpperLimit,lbpl,dbpl1,dbpl2);

      x0 = curand_uniform(states_1) * (xmax - xmin) + xmin;
      f_x0 = ff_bpl(x0,lbpl,dbpl1,dbpl2);

      if (x0 >= ll_ac){
        if (x0 <= ul_ac){

          vv[ind_start+ii].x = x0;
          ii = ii + 1;

        }
      }

    }while(ii < lenb);
    
  }



  __global__ void random_draw_ln(cufftDoubleComplex * aa, int numb,const LogNormalDist * ln, curandState_t* states_1, int lenb){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  double ul = ln->UpperLimit;
  double ll = ln->LowerLimit;
  double maxmode = ln->maxmode;
  double lsln = ln->lambda_s_LN;
  double sln = ln->sigma_LN;

  while (index < numb){
    
    dev_sample_all_ln(&states_1[index],ul,ll,maxmode,lsln,sln, lenb, index, aa);

    index += blockDim.x * gridDim.x;

  }

}



  __global__ void random_draw_bpl(cufftDoubleComplex * aa, int numb,const BrokenPowerLawDist * bp, curandState_t* states_1, int lenb, double ll_acc, double ul_acc){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  double ul = bp->UpperLimit;
  double ll = bp->LowerLimit;
  double dbpl1 = bp->delta1_BPL;
  double dbpl2 = bp->delta2_BPL;
  double lbpl = bp->lambda_s_BPL;

  while (index < numb){
    
    dev_sample_all_bpl(&states_1[index], ul, ll, lbpl, dbpl1, dbpl2, lenb, index, aa, ll_acc, ul_acc);

    index += blockDim.x * gridDim.x;

  }

}



__global__ void get_aa_ss(cufftDoubleComplex * aa, cufftDoubleComplex * ss, cufftDoubleReal * ss_s, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    aa[index].y = 0.0;

    ss[index].x = aa[index].x;
    ss[index].y = aa[index].y;

    ss_s[index] = aa[index].x;

    index += blockDim.x * gridDim.x;

  }
}



__global__ void random_seed_init(unsigned int seed, curandState_t* states_1, int n){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < n){

    unsigned int seed_here_1;
    seed_here_1 = index + 1;

    curand_init(seed, seed_here_1, 0, &states_1[index]);

    index += blockDim.x * gridDim.x;

  }

}


__device__ void dev_TK_znpl(cufftDoubleComplex * znpl, curandState_t* states_1, double tb, size_t atb, double alp, double ahp, double vbp, double Ap, double cp, int n, int lenb, int idx_in){
  
  int index_start;
  index_start = idx_in * lenb;
  
  int index_here;

  for(int ii = 0; ii < lenb;  ii++){

    index_here = index_start + ii;

    if(index_here == 0){ // first element
      znpl[index_here].x = 0.0;
      znpl[index_here].y = 0.0;


    }else if(index_here <= n / 2){ // positive
      int p_idx = index_here - 1;

      double frequencies = (double) (p_idx + 1.0) / ((double)n * (double)tb * (double)atb);
      double numer = pow(frequencies, -alp);
      double denom = 1.0 + pow(frequencies / vbp, ahp-alp);
      double powerlaw = Ap * (numer / denom) + cp;

      znpl[index_here].x = sqrt(powerlaw * 0.5) * curand_normal(states_1);
      znpl[index_here].y = sqrt(powerlaw * 0.5) * curand_normal(states_1);

      // negative (complex conjugate)
      znpl[n - index_here].x = 1. * znpl[index_here].x;
      znpl[n - index_here].y = -1. * znpl[index_here].y;

      // Nyquist
      if(p_idx == n / 2 - 1){
        znpl[index_here].y = 0.0;
      }
    }

  }

}

__global__ void TK_znpl_par(cufftDoubleComplex * znpl, curandState_t* states_1, double tb, size_t atb, double alp, double ahp, double vbp, double Ap, double cp, int n, int lenb, int numb){
  
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < numb){
    
    dev_TK_znpl(znpl, &states_1[index], tb, atb, alp, ahp, vbp, Ap, cp, n, lenb, index);

    index += blockDim.x * gridDim.x;

  }

}


/* -------------------------------------------------------------------------------------------- */


int main(int argc, char **argv)
{

  cudaDeviceReset();

  cudaEvent_t start_tot, stop_tot;
  float dt_ms_tot;
  cudaEventCreate(&start_tot);
  cudaEventCreate(&stop_tot);

  cudaEventRecord(start_tot, 0);


  assert(sizeof(cufftDoubleComplex) == 16);


  std::cout << "# --------------------------------------------------" << std::endl;
  std::cout << "# READ IN INPUT" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;  

  cudaEvent_t start, stop;
  float dt_ms;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);


  TCLAP::CmdLine cmd("Command description message", ' ', "0.9");

  int aliasTbin= 1;

  TCLAP::ValueArg<int> _LClength("n","LClength","LClength in log2",true,20,"int");
  TCLAP::ValueArg<int> _RandomSeed("r","RandomSeed","RandomSeed",true,0,"int");
  TCLAP::ValueArg<double> _tbin("t","tbin","tbin",true,100,"double");

  TCLAP::ValueArg<double> _A("A","A","A in PSD",true,0.03,"float");
  TCLAP::ValueArg<double> _v_bend("v","v_bend","v_bend in PSD",true,2.3e-4,"float");
  TCLAP::ValueArg<double> _a_low("l","a_low","a_low in PSD",true,1.1,"float");
  TCLAP::ValueArg<double> _a_high("x","a_high","a_high in PSD",true,2.2,"float");
  TCLAP::ValueArg<double> _c("c","c","c in PSD",true,0.0,"float");

  TCLAP::ValueArg<int> _num_it("i","num_it","num_it",true,5,"int");

  TCLAP::ValueArg<double> _LowerLimit("y","LowerLimit","LowerLimit in PDF",true,pow(10.0, -5),"float");
  TCLAP::ValueArg<double> _UpperLimit("z","UpperLimit","UpperLimit in PDF",true,pow(10.0, 1),"float");
  TCLAP::ValueArg<double> _LowerLimit_acc("d","LowerLimit_acc","LowerLimit_acc in PDF",true,pow(10.0, -5),"float");
  TCLAP::ValueArg<double> _UpperLimit_acc("e","UpperLimit_acc","UpperLimit_acc in PDF",true,pow(10.0, 1),"float");

  TCLAP::ValueArg<double> _lambda_s_LN("f","lambda_s_LN","lambda_s_LN in LogNormal PDF",true,pow(10.0, -3.25),"float");
  TCLAP::ValueArg<double> _sigma_LN("g","sigma_LN","sigma_LN in LogNormal  PDF",true,0.64,"float");

  TCLAP::ValueArg<double> _delta1_BPL("m","delta1_BPL","delta1_BPL in Broken Powerlaw PDF",true,-1.0,"float");
  TCLAP::ValueArg<double> _delta2_BPL("j","delta2_BPL","delta2_BPL in Broken Powerlaw PDF",true,2.53,"float");
  TCLAP::ValueArg<double> _lambda_s_BPL("k","lambda_s_BPL","lambda_s_BPL in Broken Powerlaw PDF",true,0.01445,"float");

  TCLAP::ValueArg<int> _PDFFunction("p","pdf","PDF: 0 = LogNormal, 1 = Broken PowerLaw",true,-1,"int");

  TCLAP::ValueArg<int> _len_block_rd("b","len_block_rd","Lenghts of parallel blocks in random draw",true,-1,"int");

  TCLAP::ValueArg<std::string> _OutputFolder("o","output","Output Folder",true,"./","string");

  cmd.add(_LClength);
  cmd.add(_RandomSeed);
  cmd.add(_tbin);
  cmd.add(_A);
  cmd.add(_v_bend);
  cmd.add(_a_low);
  cmd.add(_a_high);
  cmd.add(_c);
  cmd.add(_num_it);
  cmd.add(_LowerLimit);
  cmd.add(_UpperLimit);
  cmd.add(_LowerLimit_acc);
  cmd.add(_UpperLimit_acc);
  cmd.add(_lambda_s_LN);
  cmd.add(_sigma_LN);
  cmd.add(_delta1_BPL);
  cmd.add(_delta2_BPL);
  cmd.add(_lambda_s_BPL);
  cmd.add(_PDFFunction);
  cmd.add(_len_block_rd);
  cmd.add(_OutputFolder);

  cmd.parse( argc, argv );

  size_t LClength = size_t(1) << _LClength.getValue();
  int RandomSeed = _RandomSeed.getValue();
  double tbin = _tbin.getValue();

  float A = _A.getValue();
  float v_bend = _v_bend.getValue();
  float a_low = _a_low.getValue();
  float a_high = _a_high.getValue();
  float c = _c.getValue();
  int num_it = _num_it.getValue();

  double LowerLimit= _LowerLimit.getValue();
  double UpperLimit = _UpperLimit.getValue();
  double LowerLimit_acc= _LowerLimit_acc.getValue();
  double UpperLimit_acc = _UpperLimit_acc.getValue();

  double lambda_s_LN = _lambda_s_LN.getValue();
  double sigma_LN = _sigma_LN.getValue();

  double delta1_BPL = _delta1_BPL.getValue();
  double delta2_BPL = _delta2_BPL.getValue();
  double lambda_s_BPL = _lambda_s_BPL.getValue();

  int PDFFunction = _PDFFunction.getValue();
  if(PDFFunction != 0 && PDFFunction != 1){
    std::cout << "ERROR: Need to Specify PDFFunction in {0, 1}"  << std::endl;
    exit(2);
  }

  int len_block_rd = _len_block_rd.getValue();
  if(len_block_rd < 0){
    std::cout << "ERROR: Need to Specify len_block_rd > 0"  << std::endl;
    exit(2);
  }

  std::string outputFolder = _OutputFolder.getValue();

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);



  std::cout << "# --------------------------------------------------" << std::endl;
  std::cout << "# INPUT VALUES" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;
  std::cout << "# LClength:       " << LClength << std::endl;
  std::cout << "# RandomSeed:     " << RandomSeed << std::endl;
  std::cout << "# tbin:           " << tbin << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# A:              " << A << std::endl;
  std::cout << "# v_bend:         " << v_bend << std::endl;
  std::cout << "# a_low:          " << a_low << std::endl;
  std::cout << "# a_high:         " << a_high << std::endl;
  std::cout << "# c:              " << c << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# num_it:         " << num_it << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# LowerLimit:     " << LowerLimit << std::endl;
  std::cout << "# UpperLimit:     " << UpperLimit << std::endl;
  std::cout << "# LowerLimit_acc: " << LowerLimit_acc << std::endl;
  std::cout << "# UpperLimit_acc: " << UpperLimit_acc << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# lambda_s_LN:    " << lambda_s_LN << std::endl;
  std::cout << "# sigma_LN:       " << sigma_LN << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# delta1_BPL:     " << delta1_BPL << std::endl;
  std::cout << "# delta2_BPL:     " << delta2_BPL << std::endl;
  std::cout << "# lambda_s_BPL:   " << lambda_s_BPL << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# PDF         :   " << (PDFFunction == 0 ? "LogNormal" : "Broken Power Law") << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# len_block_rd:   " << len_block_rd << std::endl;
  std::cout << "# -------------------------------" << std::endl;
  std::cout << "# outputFolder:   " << outputFolder << std::endl;
  std::cout << "# -------------------------------" << std::endl;




  // Timmer & Koenig (NOTE LIA: this is not the TK part, but a preparation for it...)
  double len_in = RedNoiseL * LClength;
  int pow_floor = floor(log2(len_in));
  int pow_ceil  = ceil(log2(len_in));
  size_t len_out;
  if ((len_in - pow(2.0, pow_floor)) < (pow(2.0, pow_ceil) - len_in)){
    len_out = pow(2.0, pow_floor);
  }else{
    len_out = pow(2.0, pow_ceil);
  }
  /*-------------------------------------------------------------------------------------------- */
  std::cout << "# --------------------------------------------------" << std::endl;
  std::cout << "# ALLOCATE MEMORY" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;  

  cudaEventRecord(start, 0);

  cufftDoubleComplex * znoisypowerlaw;
  cufftDoubleComplex * longlightcurve;

  cufftDoubleComplex * fft_norm;

  cufftDoubleComplex * samples;
  cufftDoubleComplex * ampAdj;
  cufftDoubleComplex * ffti;
  cufftDoubleComplex * fftAdj;

  cufftDoubleReal * ampAdj_real;
  cufftDoubleReal * longlightcurve_real;

  cufftDoubleComplex * LCadj;

  LogNormalDist *d_lognormal;
  BrokenPowerLawDist *d_bpowerlaw;

  cufftDoubleReal * LCadj_sorted;
  cufftDoubleReal * samples_sorted;
  cufftDoubleReal * samples_sorted_it;
  cufftDoubleReal * global_i_sorted;

  curandState_t* seed_pdf;

  cufftDoubleReal * final_output;

  cudaMallocManaged((void**)&znoisypowerlaw, sizeof(cufftDoubleComplex)*len_out);
  cudaMallocManaged((void**)&longlightcurve, sizeof(cufftDoubleComplex)*len_out);

  cudaMallocManaged((void**)&fft_norm, sizeof(cufftDoubleComplex)*len_out);

  cudaMallocManaged((void**)&samples, sizeof(cufftDoubleComplex)*len_out);
  cudaMallocManaged((void**)&ampAdj, sizeof(cufftDoubleComplex)*len_out);
  cudaMallocManaged((void**)&ffti, sizeof(cufftDoubleComplex)*len_out);
  cudaMallocManaged((void**)&fftAdj, sizeof(cufftDoubleComplex)*len_out);

  cudaMallocManaged((void**)&ampAdj_real, sizeof(cufftDoubleReal)*len_out);
  cudaMallocManaged((void**)&longlightcurve_real, sizeof(cufftDoubleReal)*len_out);

  cudaMallocManaged((void**)&LCadj, sizeof(cufftDoubleComplex)*len_out);

  cudaMalloc((void **)&d_lognormal, sizeof(LogNormalDist));
  cudaMalloc((void **)&d_bpowerlaw, sizeof(LogNormalDist));

  cudaMallocManaged((void**)&LCadj_sorted, sizeof(cufftDoubleReal)*len_out);
  cudaMallocManaged((void**)&samples_sorted, sizeof(cufftDoubleReal)*len_out);
  cudaMallocManaged((void**)&samples_sorted_it, sizeof(cufftDoubleReal)*len_out);
  cudaMallocManaged((void**)&global_i_sorted, sizeof(cufftDoubleReal)*len_out);

  cudaMallocManaged((void**)&final_output, sizeof(cufftDoubleReal)*len_out);

  cudaMalloc((void**) &seed_pdf, len_out * sizeof(curandState_t));

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  /*  --------------------------------------------------------------------------------------------*/

  /* ---  DEFINE PDF -------------------------------*/

  LogNormalDist lognormal(LowerLimit, UpperLimit,
         lambda_s_LN, sigma_LN);

  cudaMemcpy(d_lognormal, &lognormal, sizeof(LogNormalDist), cudaMemcpyHostToDevice);

  BrokenPowerLawDist bpowerlaw(LowerLimit, UpperLimit,
          delta1_BPL, delta2_BPL, lambda_s_BPL);

  cudaMemcpy(d_bpowerlaw, &bpowerlaw, sizeof(BrokenPowerLawDist), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();



  /* ---  TIMMER KOENIG -------------------------------*/

  std::cout << ">> Get GPU properties..." << std::endl;  

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  unsigned int threads = prop.maxThreadsPerBlock;
  threads = 256;
  unsigned int max_blocks = prop.maxGridSize[0];
  unsigned int blocks = (len_out + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

 
 
  std::cout << "threads: " << threads << std::endl; 
  std::cout << "max_blocks: " << max_blocks << std::endl;
  std::cout << "blocks: " << blocks << std::endl;

  std::cout << "# --------------------------------------------------"<< std::endl;
  std::cout << "# STEP i - TIMMER AND KOENIG" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;

  std::cout << "generate states" << std::endl;

  cudaEventRecord(start, 0);

  int num_my_block;
  num_my_block = len_out / len_block_rd;

  std::cout << "num_my_block: " << num_my_block << std::endl;

  random_seed_init<<<blocks,threads>>>(RandomSeed, seed_pdf, num_my_block);
  cudaErrCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  std::cout << ">> Assign values" << std::endl;  

  cudaEventRecord(start, 0);

  TK_znpl_par<<<blocks,threads>>>(znoisypowerlaw, seed_pdf, tbin, aliasTbin, a_low, a_high, v_bend, A, c, len_out, len_block_rd, num_my_block);

  cudaErrCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  cudaDeviceSynchronize();


  std::cout << ">> Create plan FFT..." << std::endl;  

  cudaEventRecord(start, 0);

  // plan FFT
  cufftHandle plan_fft;
  if (cufftPlan1d(&plan_fft, len_out, CUFFT_Z2Z, 1) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return 0;
  }

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  std::cout << ">> Inverse FFT" << std::endl;  

  cudaEventRecord(start, 0);

  // execute
  if (cufftExecZ2Z(plan_fft, znoisypowerlaw, longlightcurve, CUFFT_INVERSE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed");
    return 0;
  }

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  // take only real part
  set_zero_im<<<blocks,threads>>>(longlightcurve, len_out);
  cudaErrCheck(cudaPeekAtLastError());

  cudaDeviceSynchronize();

  /* --------------------------------------------------------------------------------------------*/

  std::cout << "# --------------------------------------------------"<< std::endl;
  std::cout << "# STEP iia - RANDOM DRAW" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;

  std::cout << ">> Random draw" << std::endl;  

  cudaEventRecord(start, 0);

  if (PDFFunction == 0){
    random_draw_ln<<<blocks,threads>>>(ampAdj, num_my_block, d_lognormal, seed_pdf, len_block_rd);
  }

  if (PDFFunction == 1){
    random_draw_bpl<<<blocks,threads>>>(ampAdj, num_my_block, d_bpowerlaw, seed_pdf, len_block_rd, LowerLimit_acc, UpperLimit_acc);
  }

  cudaErrCheck(cudaPeekAtLastError());
  get_aa_ss<<<blocks,threads>>>(ampAdj, samples, samples_sorted, len_out);
  cudaErrCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);
  cudaFree(seed_pdf);

  std::cout << ">> FFT" << std::endl;  

  cudaEventRecord(start, 0);

  // get fourier transform

  if (cufftExecZ2Z(plan_fft, longlightcurve, fft_norm, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
    return 0;
  }


  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);



  /* ---  SORT SAMPLES  ------------------------------- */

  std::cout << ">> Sort samples" << std::endl;  

  cudaEventRecord(start, 0);

  thrust::sort(thrust::device, samples_sorted,samples_sorted+len_out);

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);


  std::cout << "# --------------------------------------------------"<< std::endl;
  std::cout << "# START ITERATIONS" << std::endl;  
  std::cout << "# > STEP iib - FFT" << std::endl;  
  std::cout << "# > STEP iii - SPECTRAL ADJUSTEMENT" << std::endl; 
  std::cout << "# > STEP iv - AMPLITUDE ADJUSTEMENT" << std::endl;   
  std::cout << "# --------------------------------------------------" << std::endl;

  cudaEvent_t start_it, stop_it;
  float dt_ms_it;
  cudaEventCreate(&start_it);
  cudaEventCreate(&stop_it);

  /*  --------------------------------------------------------------------------------------------*/   

  for(int j=0; j<num_it; j++){


      

      std::cout << "******* ITERATION " << j << " *******" << std::endl;

      cudaEventRecord(start_it, 0);
      /*  --------------------------------------------------------------------------------------------*/   


      /* -------------------------------------------------------------------------------------------- */
      /* ##################### step iib - discrete fourier transform ############################################# */
      /* -------------------------------------------------------------------------------------------- */


      /* ---  FORWARD FFT  ------------------------------- */

      std::cout << ">> iib - FFT" << std::endl;  

      cudaEventRecord(start, 0);

      // execute
      if (cufftExecZ2Z(plan_fft, ampAdj, ffti, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2Z Forward failed");
        return 0;
      }

      cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&dt_ms, start, stop);
      printf ("Elapsed time: %f s\n", dt_ms/1000.0);

      /*  --------------------------------------------------------------------------------------------*/   


      /* -------------------------------------------------------------------------------------------- */
      /* ##################### step iii - spectral adjustement ############################################# */
      /* -------------------------------------------------------------------------------------------- */


      /* ---  SPECTRAL ADJ  ------------------------------- */

      std::cout << ">> iii - spectral adjustement" << std::endl;  

      cudaEventRecord(start, 0);      

      // spectral adj

      get_fftAdj<<<blocks,threads>>>(fftAdj,fft_norm,ffti,len_out);      

      cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&dt_ms, start, stop);
      printf ("Elapsed time: %f s\n", dt_ms/1000.0);

      std::cout << ">> iii - FFT" << std::endl;  

      cudaEventRecord(start, 0);

      // execute ifft
      if (cufftExecZ2Z(plan_fft, fftAdj, LCadj, CUFFT_INVERSE) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2Z Inverse failed");
        return 0;
      }

      cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&dt_ms, start, stop);
      printf ("Elapsed time: %f s\n", dt_ms/1000.0);


      /*  --------------------------------------------------------------------------------------------*/      

      /* -------------------------------------------------------------------------------------------- */
      /* ##################### step iv - amplitude adjustement ############################################# */
      /* -------------------------------------------------------------------------------------------- */


      /* ---  SORT ------------------------------- */

      std::cout << ">> iv - sort" << std::endl; 

      cudaEventRecord(start, 0);       

      // Define arrays to sort

      prepare_sort<<<blocks,threads>>>(LCadj, LCadj_sorted, samples_sorted, samples_sorted_it, global_i_sorted, len_out);

      cudaDeviceSynchronize();

      // Sort samples together
      thrust::sort_by_key(thrust::device, LCadj_sorted, LCadj_sorted+len_out, global_i_sorted);
      cudaDeviceSynchronize();
      thrust::sort_by_key(thrust::device, global_i_sorted, global_i_sorted+len_out, samples_sorted_it);
      cudaDeviceSynchronize();

      set_ampAdj_ssort<<<blocks,threads>>>(ampAdj, samples_sorted_it, len_out);

      cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&dt_ms, start, stop);
      printf ("Elapsed time: %f s\n", dt_ms/1000.0);

      cudaDeviceSynchronize();
      cudaEventRecord(stop_it, 0);
      cudaEventSynchronize(stop_it);
      cudaEventElapsedTime(&dt_ms_it, start_it, stop_it);
      printf (">> Total elapsed time for iteration: %f s\n", dt_ms_it/1000.0);

      /* --------------------------------------------------------------------------------------------*/

  }




  std::cout << ">> save results" << std::endl;  

  cudaEventRecord(start, 0);

  //copy
  complex_to_real<<<blocks,threads>>>(ampAdj,final_output,len_out);

  cudaDeviceSynchronize();

  FILE * pFile;
  pFile = fopen (outputFolder.c_str(), "wb");
  fwrite (final_output , sizeof(cufftDoubleReal), len_out, pFile);
  fclose (pFile);
  
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  /* --------------------------------------------------------------------------------------------*/

 
  --------------------------------------------------------------------------------------------*/



  std::cout << "# --------------------------------------------------"<< std::endl;
  std::cout << "# FREE MEMORY" << std::endl;  
  std::cout << "# --------------------------------------------------" << std::endl;

  cudaEventRecord(start, 0);

  cufftDestroy(plan_fft);

  cudaFree(znoisypowerlaw);
  cudaFree(longlightcurve);
  cudaFree(fft_norm);
  cudaFree(samples);
  cudaFree(ampAdj);
  cudaFree(ffti);
  cudaFree(fftAdj);
  cudaFree(LCadj);
  cudaFree(LCadj_sorted);
  cudaFree(samples_sorted);
  cudaFree(samples_sorted_it);
  cudaFree(global_i_sorted);

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&dt_ms, start, stop);
  printf ("Elapsed time: %f s\n", dt_ms/1000.0);

  cudaDeviceSynchronize();
  cudaEventRecord(stop_tot, 0);
  cudaEventSynchronize(stop_tot);
  cudaEventElapsedTime(&dt_ms_tot, start_tot, stop_tot);
  std::cout << "# --------------------------------------------------"<< std::endl;
  printf ("TOTAL ELAPSED TIME: %f s\n", dt_ms_tot/1000.0);
  std::cout << "# --------------------------------------------------" << std::endl;
  /* --------------------------------------------------------------------------------------------*/


  return 0;
}

