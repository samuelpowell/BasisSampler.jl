# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell

# cudasupport.jl: CUDA compatible types and methods
export CuSampler, CuSamplerStruct, custruct, cuptx, cuinc

struct CuSamplerStruct
  fn::Cuint
  fnzind::Ptr{Cuint}
  fnzval::Ptr{Cfloat}
  cdf::Ptr{Cfloat}
  prm::Ptr{Cuint}
end

struct CuSampler 
  d_fnzind::CuArray{Cuint}
  d_fnzval::CuArray{Cfloat}
  d_cdf::CuArray{Cfloat}
  d_prm::CuArray{Cuint}
  d_struct::CuSamplerStruct
end

function CuSampler(s::Sampler)

  # Convert and upload to GPU
  fn = Cuint(length(s.cdf))
  d_fnzind = CuArray(Vector{Cuint}(s.f.nzind))
	d_fnzval = CuArray(Vector{Cfloat}(s.f.nzval))
	d_cdf = CuArray(Vector{Cfloat}(s.cdf))
  d_prm = CuArray(Vector{Cuint}(s.prm))
  
  # Dirty GPU pointer extraction and storage
  d_struct = CuSamplerStruct(fn, pointer(pointer(d_fnzind)),
                                 pointer(pointer(d_fnzval)),
                                 pointer(pointer(d_cdf)),
                                 pointer(pointer(d_prm)))
  
  # Return structure of pointers
  return CuSampler(d_fnzind, d_fnzval, d_cdf, d_prm, d_struct)

end

custruct(s::CuSampler) = s.d_struct
cuptx() = ptxfn
cuinc() = joinpath(dirname(@__FILE__), "CUDA")

