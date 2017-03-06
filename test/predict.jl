using ZSL
using Base.Test

X = [1 2 3; 1 2 3; 1 2 3]
M = eye(3,3)

@test ZSL.predict_x2y(X, X, M) == [1; 2; 3]
#= @test ZSL.predict_x2y(X[:,1], X, M) == 1 =#
#= @test ZSL.predict_x2y(X[:,2], X, M) == 2 =#
#= @test ZSL.predict_x2y(X[:,3], X, M) == 3 =#

@test ZSL.predict_y2x(X, X, M) == [1; 2; 3]
#= @test ZSL.predict_y2x(X[:,1], X, M) == 1 =#
#= @test ZSL.predict_y2x(X[:,2], X, M) == 2 =#
#= @test ZSL.predict_y2x(X[:,3], X, M) == 3 =#


