// BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
// Copyright (C) 2017 Samuel Powell

// sampler.cu: PTX module
#include <cuda.h>
#include <stdio.h>

#include "sampler.h"

#ifdef DEBUG
#define DPRINT(args...) if(id==0) { printf("Thread %i: ", id); printf(args); printf("\n"); }
#else
#define DPRINT(args...)
#endif

__device__ __noinline__
void sample(sampler_t s, float zeta, unsigned int *k, float *w)
{
  const int id = blockIdx.x * blockDim.x + threadIdx.x;

  int i = 0;

  // for(i = 0; i < s.fn; i++)
  // {
  //   s.cdf[i] < zeta ? continue : break;
  // }

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

  *k = s.prm[i];   // Get the ONE-based index into the sparse function
  DPRINT("CDF gives permutation 0-index %i", *k);

  // Loop over all entries in the sparse matrix, looking for the correct index
  for(int j = 0; j < s.fn; j++)
  {
    // When the index has been located, check if the associated value is positive or
    // negative, and assign accordingly
    if(s.fnzind[j] == *k)
    {
      *w = s.fnzval[j] < 0.f ? -1.f : 1.f;
      DPRINT("Found nonzero at 0-index %i, func value %f, weight %f", j, s.fnzval[j], *w);
      break;
    }
  }

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

