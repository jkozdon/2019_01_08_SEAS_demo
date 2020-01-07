include("operators.jl")

using GaussQuadrature
using OrdinaryDiffEq
using LinearAlgebra
using UnicodePlots

struct SpectralOP{T}
  x  :: Array{T, 1}
  ω  :: Array{T, 1}
  D  :: Array{T, 2}
  e0 :: Array{T, 1}
  eN :: Array{T, 1}
  function SpectralOP{T}(N) where T
    Nq = N + 1
    x, ω = legendre(T, Nq, both)
    x = (x .+ 1) / 2
    ω /= 2
    D = derivative(x)
    e0 = [1; zeros(T, N)]
    eN = [zeros(T, N); 1]
    new{T}(x, ω, D, e0, eN)
  end
end

function spectralodefun!(dq, q, p, t)
  o = p.o
  f = p.f
  Nq = length(o.x)
  v,  u  = @views  q[1:Nq],  q[Nq+1:2Nq]
  dv, du = @views dq[1:Nq], dq[Nq+1:2Nq]

  D = o.D
  ω = o.ω
  e0 = o.e0
  dv .= (-D' * (ω .* D * u) - e0 * f(v[1])) ./ ω
  du .= v
end


let
  N = 100
  T = Float64

  tspan = (T(0), T(3//10))

  Nq = N + 1
  o = SpectralOP{T}(N)
  q = [zeros(T, Nq);sin.(2o.x * π)]
  f(v) = asinh(v)

  p = (o=o, f=f)
  prob = ODEProblem(spectralodefun!, q, tspan, p)
  sol = solve(prob, Tsit5();
              saveat=range(tspan[1], stop=tspan[2], length = 4),
              atol = 1e-6, rtol = 1e-3,
              internalnorm=(x, _)->norm(x, Inf))

  for n = 1:length(sol.t)
    display(lineplot(o.x, sol.u[n][Nq+1:2Nq], width=100, xlim=(0,1),
                     title="t = $(sol.t[n])"))
  end
end
