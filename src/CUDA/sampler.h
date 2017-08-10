// BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
// Copyright (C) 2017 Samuel Powell

// sampler.h: PTX module header

struct sampler {
  const unsigned int fn;        // Sparse function length
  unsigned int * const fnzind;  // Sparse nonzero indices
  float * const fnzval;         // Sparse nonzero values
  float * const cdf;            // Cumulative densitiy function
  unsigned int * const prm;     // Permutation vector
};
typedef struct sampler sampler_t;

extern "C" __device__ void sample(sampler_t s, float zeta, unsigned int *k, float *w);
extern "C" __global__ void sample_test(sampler_t s, int ns, float *zeta, unsigned int *k, float *w);