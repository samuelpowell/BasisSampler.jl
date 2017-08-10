# BasisSampler.jl: importance sample from arbitrary sets of basis coefficients 
# Copyright (C) 2017 Samuel Powell

# build.jl: compile CUDA support if available, write configuration to config file

const nvccbin = "nvcc"
const sources = ["sampler.cu"]
const srcdir = joinpath(dirname(@__FILE__), "..", "src", "CUDA")
const nvccsrc = (joinpath(srcdir, srcfile) for srcfile in sources)
const incdir = srcdir

cudabuild = false

try
  using CUDAdrv
  run(`$nvccbin --use_fast_math -ptx -dc -lineinfo -src-in-ptx $nvccsrc`)
  info("BasisSampler.jl CUDA backend built, PTX module available.")
  cudabuild = true
catch
  cudabuild = false
  info("BasisSampler.jl unable to build CUDA backend, PTX module unavailable.")
end

# Write config file
if cudabuild
  open("config.jl", "w") do f
    write(f, """
      const ptxfn = \"$(joinpath(dirname(@__FILE__), "sampler.ptx"))\"
      const cudabuild = true
      using CUDAdrv
      """)
  end
else
  open("config.jl", "w") do f
    write(f, """const cudabuild = false""")
  end
end
