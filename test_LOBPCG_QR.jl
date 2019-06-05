# Adapted from https://gist.github.com/antoine-levitt/f8ac3637c5d5d778a448780172e63e28
using LinearAlgebra
using NPZ
using LinearAlgebra

function ortho(X)
    Array(qr(X).Q)
end

function align(X,Y)
    # returns an orthogonal combination X * c with c unitary
    # such that X*c is closest to Y
    proj = X'Y
    U,S,V = svd(proj)
    any(x -> x <= .1, S) && error("Not enough overlap")
    U*V'
end

function LOBPCG(A, X, tol=1e-16, maxiter=100, do_align=true; Prec=I)
    N = size(X,1)
    M = size(X,2)
    X = ortho(X)
    Y = zeros(N,3M)
    P = zeros(N,M)
    R = zeros(N,M)
    niter = 1
    while true
        Y[:,1:M] = X
        R .= Prec \ (A*X - X*(X'*A*X))
        Y[:,M+1:2M] = R
        Y[:,2M+1:3M] = P
        Y .= ortho(Y)
        c = eigvecs(Symmetric(Y'*A*Y))[:,1:M]
        new_X = Y*c
        e = Array{Float64}(I,3M,M)
        do_align && (e .= e*align(e,c))
        P .= Y*(c - e)
        println(niter, " ", norm(R))
        X .= new_X
        norm(R) < tol && break
        niter = niter + 1
        niter <= maxiter || break
    end

    λ = diag(X'*A*X)
    λ, X
end

function check_with_reference(λ, v, tol)
    m = length(λ)
    A  = npzread("./data.npz")["A"]
    λref, vref = eigen(Symmetric(A))
    @assert maximum(abs.(λref[1:m] - λ)) < tol
end


function run()
    A  = npzread("./data.npz")["A"]
    X = npzread("./data.npz")["x0"]

    tol = 1e-14
    λ, v = LOBPCG(A, X, tol, 300, false)
    check_with_reference(λ, v, tol)
end

function run_preconditioner()
    A  = npzread("./data.npz")["A"]
    P  = npzread("./data.npz")["P"]
    X = npzread("./data.npz")["x0"]

    tol = 1e-14
    Prec = Diagonal(1 ./ diag(P))
    λ, v = LOBPCG(A, X, tol, 300, false, Prec=Prec)
    check_with_reference(λ, v, tol)
end
