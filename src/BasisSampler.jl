# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell

module BasisSampler

export BasisSampler, sample, value, absnorm, norm

struct BasisSampler{T}
  f::SparseVector{T,Int}      # Store original function projection
  cdf::Vector{T}              # Compressed cumulative density function 
  prm::Vector{Int}            # Permutation vector
  absnorm::T                  # Absolute norm of the function (for rescaling)
end

"""
    BasisSampler(f::SparseVector)

Construct a `BasisSampler` from a sparse vector `f` where row `i` of vector `f` is the value
of `ith` node in some arbitrary basis.
"""
function BasisSampler(f::SparseVector{T,Int}) where T

  # Distributions dont care about polarity
  fa = abs.(f)

  # Sort the non-zero entries into ascending order
  nzperm = fa.nzind[sortperm(nonzeros(fa), rev=true)]
  nzsort = full(fa[nzperm])

  # Form a cumulative density function on the nonzero entries
  fn = sum(nzsort)
  nzcdf = cumsum(nzsort) ./ fn

  BasisSampler(f, nzcdf, nzperm, fn)

end

"""
    sample(f::BasisSampler) -> i, w

Draw an unbiased index `i` of a node from a function descritised in an arbitrary basis, and
weighting to restore the polarity of the function.
"""
function sample(f::BasisSampler{N,T}) where {N,T}

  ζ = rand()
  i = 0
  while i < length(f.prm)
    i += 1
    @inbounds f.cdf[i] < ζ ? continue : break
  end

  @inbounds s = f.prm[i]
  @inbounds w = f.f[s] < 0 ? -one(T) : one(T)

  return s, w

end

"""
    value(f::BasisSampler, i)

Return the projected nodal value of node `i`.
"""
value(f::BasisSampler, i) = f.f[i]

"""
    absnorm(f::BasisSampler)

Return the integral of the absolute value of the function, which can be used to rescale a
sampled function to its original magnitude.
"""
absnorm(f::BasisSampler) = f.absnorm
norm(f::BasisSampler) = absnorm(f)

end # module

