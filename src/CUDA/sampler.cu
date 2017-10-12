// BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
// Copyright (C) 2017 Samuel Powell

// sampler.cu: PTX module
#include <cuda.h>
#include <stdio.h>

#include "sampler.h"

#ifdef BS_DEBUG
#define DPRINT(args...) if(id==0) { printf("Thread %i: ", id); printf(args); printf("\n"); }
#else
#define DPRINT(args...)
#endif

__device__ __noinline__
void sample(sampler_t s, float zeta, unsigned int *k, float *w)
{
  #ifdef BS_DEBUG
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  #endif

  int i = 0;
    
  while(s.cdf[i] < zeta)
  {
    if(++i >= s.fn)
      break;
  }
  DPRINT("Drew random %f, sampled CDF at 0-index %i", zeta, i);

  if(i >= s.fn)
  {
    printf("Sampling CDF of length %i at index %i", s.fn, i);
    printf("\tCDF[i-1] = %.9g\n", s.cdf[i-1]);
    printf("\tzeta = %.9g\n", zeta);

    for(int ii = 0; ii < s.fn; ii++)
      printf("CDF [%i]: %.9g\n", ii, s.cdf[ii]);

    asm("trap;");
  }

  // Get the ONE-based index into the sparse function (the element index)
  *k = s.prm[i];   
  DPRINT("CDF gives permutation 0-index %i", *k);

  // Get the polarity of the function from compressed vector
  *w = s.pol[i] > 0.f ? 1.f : -1.f;
  DPRINT("Function polarity at 0-index %i is %i", *w);

  // Convert to zero based element index
  (*k)--;

  return;
}

__global__
void sample_test(sampler_t s, int ns, float *zeta, unsigned int *k, float *w)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int gr = blockDim.x * gridDim.x;

  for(int sid = id; sid < ns; sid += gr)
  {
    sample(s, zeta[sid], k+sid, w+sid);
    k[sid]++; // Return one-based indices to Julia
  }
  
  return;

}

