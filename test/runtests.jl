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

