using ZSL
using Base.Test

X = [1 3 4; 5 6 7]
λ = 0
M = ZSL.train_x2y(X, X, λ)
@test_approx_eq_eps sum(M - eye(size(X,1),size(X,1))) .0 1e-6
W = ZSL.train_y2x(X, X, λ)
@test_approx_eq_eps sum(W - eye(size(X,1),size(X,1))) .0 1e-6

#############################################

Y = [1 2 4; 32 2 1; 3 3 3]

M = ZSL.train_x2y(X, Y, λ)
@test size(M,1) == size(Y,1)
@test size(M,2) == size(X,1)

W = ZSL.train_y2x(X, Y, λ)
@test size(W,1) == size(X,1)
@test size(W,2) == size(Y,1)

