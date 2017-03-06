
#Usage:
#julia demo_synthetic.jl

##########################################

push!(LOAD_PATH,"../source/")
using Generator, ZSL

###########################################

println("Generating data")
d = [3000, 300, 300]
n = 10000
X, Y, gold = synthetic(d, n)

###########################################

println("Trainining model")
λ = 0.1
M = train_x2y(X[:,gold[1:8000,1]], Y[:,gold[1:8000,2]], λ)
W = train_y2x(X[:,gold[1:8000,1]], Y[:,gold[1:8000,2]], λ)

###########################################

println("1-NN accuracy (training data)")
println("  x2y: ", mean(predict_x2y(X[:,gold[1:8000,1]], Y[:,gold[1:8000,2]], M) .== collect(1:8000)))
println("  y2x: ", mean(predict_y2x(X[:,gold[1:8000,1]], Y[:,gold[1:8000,2]], W) .== collect(1:8000)))

println("1-NN accuracy (test data)")
println("  x2y: ", mean(predict_x2y(X[:,gold[8001:end,1]], Y[:,gold[8001:end,2]], M) .== collect(1:2000)))
println("  y2x: ", mean(predict_y2x(X[:,gold[8001:end,1]], Y[:,gold[8001:end,2]], W) .== collect(1:2000)))
