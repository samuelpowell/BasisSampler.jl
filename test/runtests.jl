using BasisSampler
using Base.Test

const nsamples = 1e8
const gausvec = exp.(-linspace(-5,5,100).^2)
const randvec = rand(1000)*2-1
const endvec = [zeros(100); 1.]
const strvec = [1.; zeros(100)]
const servec = [1.; zeros(100); 1.]

const vectors = [[gausvec]; [randvec]; [endvec]; [strvec]; [servec]]

@testset "Reconstruction" begin

  for v in vectors 

    s = Sampler(v)
    l = length(v)
    r = zeros(l)

    @inbounds for i in 1:nsamples
      e, w = sample(s)
      r[e] += w
    end

    r .*= absnorm(s)/nsamples

    @test (norm( r - v )./l) < 1e-4
  
  end

end

cudatest = false

if BasisSampler.cudabuild
  try
    @eval using CUDAdrv
    cudatest = true
  catch
    info("CUDA PTX available, but CUDAdrv cannot be loaded to run tests.")
  end
end

if cudatest
    
  ns = 100000000
  zvec = rand(Float32, ns)

  # Prepare the GPU
  ctx = CuContext(CuDevice(0))
  ptxmod = CuModuleFile(CuSamplerPTX())
  kernel = CuFunction(ptxmod, "sample_test")

  gpurvec = [rand(100)*2-1; zeros(10); 1.];

  # Create a CPU sampler, and upload to GPU
  s = Sampler(gpurvec)
  l = length(gpurvec)
  ds = CuSampler(s)
  
  k = CuArray{Cuint}(ns)
  w = CuArray{Cfloat}(ns)
  z = CuArray(zvec)

  # Run the sampler, return results
  cudacall(kernel, CuDim(4096), CuDim(128), 0, CuDefaultStream(),
    (CuSamplerStruct, Cint, Ptr{Cfloat}, Ptr{Cuint}, Ptr{Cfloat}), 
    ds.d_struct, ns, z, k, w)

  hk = Array(k)
  hw = Array(w)
    
  # Compare against CPU implmentation, and against the function
  rgpu = zeros(l)
  rcpu = zeros(l)
  @inbounds for i in 1:ns
    rgpu[hk[i]] += hw[i]
    kcpu, wcpu = sample(s, zvec[i])
    rcpu[kcpu] += wcpu
  end

  # Test in an absolute sense
  @testset "CUDA backend" begin
    @test norm((rcpu-rgpu)./(ns*l)) < 1e-8
    @test norm( rgpu*absnorm(s)/ns - gpurvec )./l < 7e-5
  end



end

