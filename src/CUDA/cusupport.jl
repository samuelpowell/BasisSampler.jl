# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell

# cudasupport.jl: CUDA compatible types and methods
export CuSampler, CuSamplerStruct, custruct, cuptx, cuinc

struct CuSamplerStruct
  fn::Cuint
  cdf::Ptr{Void}
  pol::Ptr{Void}
  prm::Ptr{Void}
end

struct CuSampler 
  d_cdf::CuArray{Cfloat}
  d_pol::CuArray{Cuchar}
  d_prm::CuArray{Cuint}
  d_struct::CuSamplerStruct
end

function CuSampler(s::Sampler)

  # Convert and upload to GPU
  fn = Cuint(length(s.cdf))
  d_pol = CuArray(Vector{Cuchar}(s.pol))
  d_cdf = CuArray(Vector{Cfloat}(s.cdf))
  d_prm = CuArray(Vector{Cuint}(s.prm))
  
  # Dirty GPU pointer extraction and storage
  d_struct = CuSamplerStruct(fn, d_cdf.buf.ptr, d_pol.buf.ptr, d_prm.buf.ptr)
  
  # Return structure of pointers
  return CuSampler(d_cdf, d_pol, d_prm, d_struct)

end

custruct(s::CuSampler) = s.d_struct
cuptx() = ptxfn
cuinc() = joinpath(dirname(@__FILE__))

