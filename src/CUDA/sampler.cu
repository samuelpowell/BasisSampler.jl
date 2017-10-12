// BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
// Copyright (C) 2017 Samuel Powell

// sampler.cu: PTX module
#include <cuda.h>
#include <stdio.h>

#include "sampler.h"

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

