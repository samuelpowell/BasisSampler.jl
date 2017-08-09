# BasisSampler

[![Build Status](https://travis-ci.org/samuelpowell/BasisSampler.jl.svg?branch=master)](https://travis-ci.org/samuelpowell/BasisSampler.jl)

[![Coverage Status](https://coveralls.io/repos/samuelpowell/BasisSampler.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/samuelpowell/BasisSampler.jl?branch=master)

[![codecov.io](http://codecov.io/github/samuelpowell/BasisSampler.jl/coverage.svg?branch=master)](http://codecov.io/github/samuelpowell/BasisSampler.jl?branch=master)


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
n = 1e6
l = length(v)
r = zeros(l)

  @inbounds for i in 1:n
    e, w = sample(s)
    r[e] += w
  end

r .*= absnorm(s)/n

println("Norm: ", norm( r - v )./l)
```

