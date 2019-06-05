# Adapted from https://gist.github.com/antoine-levitt/f8ac3637c5d5d778a448780172e63e28
using LinearAlgebra
using NPZ
using LinearAlgebra

# Unfortunately missing in standard library
LinearAlgebra.ldiv!(Y, M::Diagonal, X) = Y .= ldiv!(M, copy(X))

function LOBPCG(A, X0, maxiter=100; Prec=I, tol=2size(A,2)*eps(eltype(A)))
    ortho(X) = Array(qr(X).Q)

    N = size(A, 2)
    m = size(X0, 2)
    T = eltype(A)

    # Allocate containers for subspace data
    Y = similar(X0, T, N, 3m)
    X = view(Y, :, 1:m)
    R = similar(X0, T, N, m)
    P = view(Y, :, 2m+1:3m) .= 0

    # Storage for A*Y
    Ay = similar(Y, T)

    # Orthogonalise X0 and apply A to it.
    X .= ortho(X0)
    rvals = zeros(m, m)
    mul!(view(Ay, :, 1:m), A, X)
    for niter in 1:maxiter
        # Compute residuals and Ritz values
        rvals .= X' * Ay[:, 1:m]
        R .= Ay[:, 1:m] - X * rvals
        println(niter, " ", norm(R))
        norm(R) < tol && break
        niter += 1

        # Update the residual slot of Y
        if Prec == I
            Y[:, m+1:2m] .= R
        else
            ldiv!(view(Y, :, m+1:2m), Prec, R)
        end

        # Orthogonalise Y and solve Rayleigh-Ritz step
        Y .= ortho(Y)
        mul!(Ay, A, Y)

        c = eigvecs(Symmetric(Y'*Ay))[:, 1:m]
        new_X = Y*c
        P .= new_X - X
        X .= new_X
        Ay[:, 1:m] .= Ay * c
    end

    diag(rvals), X
end

function check_with_reference(λ, v, tol)
    m = length(λ)
    A  = npzread("./data.npz")["A"]
    λref, vref = eigen(Symmetric(A))
    println("maxerror:  $(maximum(abs.(λref[1:m] - λ)))")
    @assert maximum(abs.(λref[1:m] - λ)) < tol
end


function run()
    A = npzread("./data.npz")["A"]
    X = npzread("./data.npz")["x0"]

    tol = 1e-13
    λ, v = LOBPCG(A, X, 300, tol=tol)
    check_with_reference(λ, v, tol)
end

function run_preconditioner()
    A  = npzread("./data.npz")["A"]
    P  = npzread("./data.npz")["P"]
    X = npzread("./data.npz")["x0"]

    tol = 1e-13
    Prec = Diagonal(1 ./ diag(P))
    λ, v = LOBPCG(A, X, 300, tol=tol, Prec=Prec)
    check_with_reference(λ, v, tol)
end
