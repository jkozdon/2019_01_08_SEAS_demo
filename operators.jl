"""
    baryweights(r)

returns the barycentric weights associated with the array of points `r`

Reference:
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function baryweights(r)
  T = eltype(r)
  Np = length(r)
  wb = ones(T, Np)

  for j = 1:Np
    for i = 1:Np
      if i != j
        wb[j] = wb[j] * (r[j] - r[i])
      end
    end
    wb[j] = one(T) / wb[j]
  end
  wb
end

"""
    derivative(r, wb=baryweights(r))

returns the spectral differentiation matrix for a polynomial defined on the
points `r` with associated barycentric weights `wb`

Reference:
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function derivative(r, wb=baryweights(r))
  T = promote_type(eltype(r), eltype(wb))
  Np = length(r)
  @assert Np == length(wb)
  D = zeros(T, Np, Np)

  for k = 1:Np
    for j = 1:Np
      if k == j
        for l = 1:Np
          if l != k
            D[j, k] = D[j, k] + one(T) / (r[k] - r[l])
          end
        end
      else
        D[j, k] = (wb[k] / wb[j]) / (r[j] - r[k])
      end
    end
  end
  D
end

"""
    interpolation(rsrc, rdst, wbsrc=baryweights(rsrc))

returns the polynomial interpolation matrix for interpolating between the points
`rsrc` (with associated barycentric weights `wbsrc`) and `rdst`

Reference:
  Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation",
  SIAM Review 46 (2004), pp. 501-517.
  <https://doi.org/10.1137/S0036144502417715>
"""
function interpolation(rsrc, rdst, wbsrc=baryweights(rsrc))
  T = promote_type(eltype(rsrc), eltype(rdst), eltype(wbsrc))
  Npdst = length(rdst)
  Npsrc = length(rsrc)
  @assert Npsrc == length(wbsrc)
  I = zeros(T, Npdst, Npsrc)

  for k = 1:Npdst
    for j = 1:Npsrc
      I[k, j] = wbsrc[j] / (rdst[k] - rsrc[j])
      if !isfinite(I[k,j])
        I[k, :] .= zero(T)
        I[k, j] = one(T)
        break
      end
    end
    d = sum(I[k, :])
    I[k, :] = I[k, :] / d
  end
  I
end
