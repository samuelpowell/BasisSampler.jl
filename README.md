# BasisSampler

[![Build Status](https://travis-ci.org/samuelpowell/BasisSampler.jl.svg?branch=master)](https://travis-ci.org/samuelpowell/BasisSampler.jl)
[![Coverage Status](https://coveralls.io/repos/samuelpowell/BasisSampler.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/samuelpowell/BasisSampler.jl?branch=master)
[![codecov.io](http://codecov.io/github/samuelpowell/BasisSampler.jl/coverage.svg?branch=master)](http://codecov.io/github/samuelpowell/BasisSampler.jl?branch=master)


## Installation

```
julia> Pkg.clone("git@github.com:samuelpowell/BasisSampler.jl.git")
```

```
julia> Pkg.build("BasisSampler")
```

## Usage

Express some arbitrary function with an (optionally sparse) vector of basis coefficients:

```
julia> v = rand(1000)*2-1
```

Build a `Sampler`:

```
julia> s = Sampler(v)
```

Draw unbiased samples representing the index of the basis coefficient, and the polarity of
that sample:

```
julia> sample(s)
(846, -1.0)
```

Reconstruct a function:

```
julia> function recon(n)
  l = length(v)
  r = zeros(l)

  @inbounds for i in 1:n
    e, w = sample(s)
    r[e] += w
  end

  r .*= absnorm(s)/n

  println("Norm: ", norm( r - v )./l)

  end
recon (generic function with 1 method)

julia> recon(1e6)
Norm: 0.0005198154172866841
```

## CUDA Module (experimental)

`BasisSampler.jl` provides a PTX module and helper types which can be used by a CUDA
application. The PTX module is compiled automatically during the package build process if
CUDAdrv.jl is installed, and `nvcc` is available on the path.

To call the kernel function, include `src/CUDA/sampler.h` in your project, link against
`src/CUDA/sampler.ptx` using the CUDA driver JIT features, and pass the appropriate structures
to your CUDA application.

To upoad the neccedary data to the GPU:

```
julia>  ds = CuSampler(s)
```

Then pass a `CuSamplerStruct` type to your kernel, acquired by calling `custruct` on the 
previously created object :

```
julia> cudacall(..., (...,{CuSamplerStruct},...), ..., custruct(ds), ...)
```

Note that the structure contains only an integer and a set of pointers to the GPU, so this
can be safely passed by value.
