#!/usr/bin/env julia
using NPZ
using IterativeSolvers
using LinearAlgebra

A  = npzread("./data.npz")["A"]
P  = npzread("./data.npz")["P"]
x0 = npzread("./data.npz")["x0"]

N, m = size(x0)
λreff, vreff = eigen(Symmetric(A))
λref = λreff[1:m]
vref = vreff[:, 1:m]

println("Problem properties:")
println("N:                 ", N)
println("m:                 ", m)
println("λref:              ", λref)
println("λapprox:           ", eigvals(Symmetric(x0' * A * x0)))
println("A λ dist:          ", abs.(λreff[m+1] - λreff[m]))

λPAm = vreff[:, m]' * P * A * vreff[:, m]
λPAm1 = vreff[:, m + 1]' * P * A * vreff[:, m + 1]
println("P*A ritz dist:     ", abs.(λPAm1 - λPAm))

# Unfortunately missing in standard library
LinearAlgebra.ldiv!(Y, M::Diagonal, X) = Y = ldiv!(M, copy(X))

function test_lobpcg(;read_guess=true, tol=1e-5)
    # P is a diagonal matrix that contains the preconditioner
    # to be applied. Julia wants to apply P^{-1}
    Prec = Diagonal(1 ./ diag(P))
    res = try
        if read_guess
            lobpcg(A, false, x0, P=Prec, maxiter=100, tol=tol)
        else
            lobpcg(A, false, m, P=Prec, maxiter=100, tol=tol)
        end
    catch y
        println("ERROR: ", y)
        return nothing
    end
    @assert maximum(abs.(λref - res.λ)) < tol
    nothing
end

println()
println("read_guess=true")
test_lobpcg(read_guess=true)

println()
println("read_guess=false")
test_lobpcg(read_guess=false)
