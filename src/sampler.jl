# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell

# sampler.jl: Types and methods to support basis sampling

struct Sampler{T}
  f::SparseVector{T,Int}      # Store original function projection
  cdf::Vector{T}              # Compressed cumulative density function 
  prm::Vector{Int}            # Permutation vector
  pol::Vector{UInt8}          # Compressed function polarity (for sampling)
  absnorm::T                  # Absolute norm of the function (for rescaling)
end

"""
    Sampler(f::AbstractVector)

Construct a `Sampler` from a sparse vector `f` where row `i` of vector `f` is
the value of `ith` node in some arbitrary basis.
"""
Sampler(f::AbstractVector{T}) where {T} = Sampler(sparsevec(f))


function Sampler(f::SparseVector{T,Int}) where {T}

  # Distributions dont care about polarity
  fa = abs.(f)

  # Sort the non-zero entries into ascending order
  nzperm = fa.nzind[sortperm(nonzeros(fa), rev=true)]
  nzsort = full(fa[nzperm])

  # Form a cumulative density function on the nonzero entries
  fn = sum(nzsort)
  nzcdf = cumsum(nzsort) ./ fn

  # From compressed polarity vector
  pol = Vector{UInt8}(length(nzperm))
  for (i, perm) in enumerate(nzperm)
    pol[i] = f[perm] >= 0 ? one(UInt8) : zero(UInt8)
  end

  Sampler(f, nzcdf, nzperm, pol, fn)

end


"""
    sample(f::Sampler) -> i, w

Draw an unbiased index `i` of a node from a function descritised in an arbitrary
basis, and weighting to restore the polarity of the function.
"""
sample(f::Sampler{T}) where {T} = sample(f, rand())

function sample(f::Sampler{T}, ζ::Number) where {T}

  i = 0
  while i < length(f.prm)
    i += 1
    @inbounds f.cdf[i] < ζ ? continue : break
  end

  @inbounds s = f.prm[i]
  @inbounds w = f.pol[i] > 0 ? one(T) : -one(T)
  
  return s, w

end

"""
    value(f::Sampler, i)

Return the projected nodal value of node `i`.
"""
value(f::Sampler, i) = f.f[i]

"""
    absnorm(f::Sampler)

Return the integral of the absolute value of the function, which can be used to
rescale a sampled function to its original magnitude.
"""
absnorm(f::Sampler) = f.absnorm
norm(f::Sampler) = absnorm(f)