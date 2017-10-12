# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell
__precompile__()

module BasisSampler

import Base.norm
export Sampler, sample, value, absnorm, norm

# Include build configuration
try
  include(joinpath(dirname(@__FILE__), "..", "deps", "config.jl"))
catch
  error("Configuration file missing, run 'Pkg.build(\"BasisSampler\")' to configure.")
end

include("sampler.jl")

if cudabuild 
  include("CUDA/cusupport.jl")
end

end # module

