using Generator
using Base.Test

# Set dataset
d = [3000, 300, 300]
n = 10000
X, Y, gold = Generator.synthetic(d, n)

@test d[2] == size(X,1)
@test n == size(X,2)
@test d[3] == size(Y,1)
@test n == size(Y,2)
@test size(gold,1) == n
@test size(gold,2) == 2

J = Generator.gold2matrix(gold)

@test size(J,1) == n
@test size(J,2) == n
